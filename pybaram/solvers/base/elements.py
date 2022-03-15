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
        if cfg.get('solver-time-integrator', 'stepper') == 'lu-sgs':
            self.nei_ele = -np.ones((nface, self.neles), dtype=np.int)

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
