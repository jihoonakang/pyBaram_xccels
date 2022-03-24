# -*- coding: utf-8 -*-
import functools as fc
import numpy as np

from pybaram.geometry import get_geometry
from pybaram.utils.np import chop, npeval


class BaseElements:
    name = 'base'

    def __init__(self, be, cfg, name, eles, vcon):
        # Argument save
        self.be = be
        self.cfg = cfg
        self.name = name
        self.eles = eles
        self._vcon = vcon

        # Dimension 설정
        self.nvtx, self.neles, self.ndims = self.eles.shape

        # Geometry 설정
        self.geom = get_geometry(name)
        self.nface = nface = self.geom.nface

        self.order = order = cfg.getint('solver', 'order', 1)

        if order > 1:
            self.dxc = self.xc - self.xf

        # ifpts
        #if cfg.get('solver-time-integrator', 'stepper') == 'simple-point-implicit':
        self.nei_ele = np.ones((nface, self.neles), dtype=np.int)*-1

    def coloring(self):
        # Multi-Coloring
        #TODO: 계산 시간 확인 필요
        color = np.zeros(self.neles, dtype=np.int)
        max_color = 1
        is_colored = np.empty(32, dtype=np.int)

        # Search Coloring (Search along hyperplane)
        xn = np.sum(self.xc, axis=1)
        for idx in np.argsort(xn):
            # 이 컬러 번호가 주변에 있는지 확인용 자료
            is_colored[:max_color+1] = 0

            for jdx in range(self.nface):
                # 인접 셀 탐색 후 컬러 번호가 있는지 저장
                nei = self.nei_ele[jdx, idx]
                if nei > 0:
                    nei_color = color[nei]
                    is_colored[nei_color] = 1
            
            # 주변에 없는 가장 작은 컬러 번호 찾기
            is_found = False
            for k in range(1, max_color+1):
                if is_colored[k] == 0:
                    if not is_found:
                        c = k
                        is_found = True
                    else:
                        c = min(c, k)

            if is_found:
                # 주변에 없으면서 가장 작은 컬러 번호 부여
                color[idx] = c
            else:
                # Color level 증가 후 컬러 번호 기록
                max_color += 1
                color[idx] = max_color

        ele_idx = np.arange(self.neles, dtype=np.int)

        # Linked List 형식 저장
        ncolor = np.cumsum([sum(color==i) for i in range(max_color+1)])
        icolor = np.concatenate([ele_idx[color==i] for i in range(max_color+1)])
        return ncolor, icolor, color

    def reordering(self):
        try:
            # Use Scipy sparse packages
            from scipy import sparse
            from scipy.sparse.csgraph import reverse_cuthill_mckee

            # Convert nei_ele to csr sparse matrix
            mask = self.nei_ele > 0
            index = (np.ones(self.nface, dtype=int)[:, None]* np.arange(self.neles)[None, :])

            col = np.concatenate([index[mask], np.arange(self.neles)])
            row = np.concatenate([self.nei_ele[mask], np.arange(self.neles)])
            data = np.ones_like(row)

            mtx = sparse.csr_matrix((data, (row, col)), shape=(self.neles, self.neles))

            # reverse Cuthill MacKee reordering
            mapping = reverse_cuthill_mckee(mtx)
            unmapping = np.argsort(mapping)

        except:
            mapping = np.arange(self.neles, dtype=int)
            unmapping = np.arange(self.neles, dtype=int)

        return mapping, unmapping

    def set_ics_from_cfg(self):
        xc = self.geom.xc(self.eles).T

        # Initialize
        subs = dict(zip('xyz', xc))
        ics = [npeval(self.cfg.getexpr('soln-ics', v, self._const), subs)
               for v in self.primevars]
        ics = self.prim_to_conv(ics, self.cfg)

        # Allocate and copy
        self._ics = np.empty((self.nvars, self.neles))
        for i in range(self.nvars):
            self._ics[i] = ics[i]

    def set_ics_from_sol(self, sol):
        self._ics = sol

    @property
    @fc.lru_cache()
    def _vol(self):
        return np.abs(self.geom.vol(self.eles))

    @property
    @fc.lru_cache()
    def tot_vol(self):
        return np.sum(self._vol)

    @property
    @fc.lru_cache()
    def rcp_vol(self):
        return 1/np.abs(self._vol)

    @fc.lru_cache()
    def _gen_snorm_fpts(self):
        sign = np.sign(self.geom.vol(self.eles))[..., None]
        snorm = self.geom.snorm(self.eles)
        mag = np.einsum('...i,...i', snorm, snorm)
        mag = np.sqrt(mag)
        vec = snorm / mag[..., None]*sign
        return mag, vec

    @property
    def _mag_snorm_fpts(self):
        return self._gen_snorm_fpts()[0]

    @property
    def _vec_snorm_fpts(self):
        return self._gen_snorm_fpts()[1]

    @property
    def mag_fnorm(self):
        return self._mag_snorm_fpts

    @property
    @fc.lru_cache()
    def vec_fnorm(self):
        return self._vec_snorm_fpts.swapaxes(1, 2).copy()

    @property
    def _perimeter(self):
        return np.sum(self._mag_snorm_fpts, axis=0)

    @property
    @fc.lru_cache()
    def le(self):
        return 1/(self.rcp_vol * self._perimeter)

    @property
    @fc.lru_cache()
    def xc(self):
        return self.geom.xc(self.eles)

    @property
    @fc.lru_cache()
    def xf(self):
        return self.geom.xf(self.eles)

    @property
    @fc.lru_cache()
    @chop
    def _prelsq(self):
        dxc = np.rollaxis(self.dxc, 2)

        # TODO:Inverse distance weight
        w = 1.0 / np.linalg.norm(dxc, axis=0)
        dxc = dxc * w

        # Least square matrix [dx*dy] and its inverse
        lsq = np.array([[np.einsum('ij,ij->j', x, y)
                         for y in dxc] for x in dxc])
        invlsq = np.linalg.inv(np.rollaxis(lsq, 2))

        # Final form: lsq^-1*dx
        return np.einsum('kij,jmk->imk', invlsq, dxc*w)

    @property
    @fc.lru_cache()
    def dxf(self):
        return self.geom.dxf(self.eles).swapaxes(1, 2)

    @property
    @fc.lru_cache()
    def dxv(self):
        return self.geom.dxv(self.eles).swapaxes(1, 2)
