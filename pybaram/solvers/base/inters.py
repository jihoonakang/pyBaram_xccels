# -*- coding: utf-8 -*-
import numpy as np
import numba as nb
import re


class BaseInters:
    name = 'base'

    def __init__(self, cfg, elemap, lhs):
        # cfg 저장
        self.cfg = cfg

        # Dimension 저장
        self.nfpts = len(lhs)
        self.ele0 = ele0 = elemap[next(iter(elemap))]
        self.ndims, self.nvars = ele0.ndims, ele0.nvars
        self.primevars = ele0.primevars
        self._const = cfg.items('constants')

        self.order = cfg.getint('solver', 'order', 1)

        # Normal Vector
        self._mag_snorm = self._get_fpts('_mag_snorm_fpts', elemap, lhs)[0]
        self._vec_snorm = self._get_fpts('_vec_snorm_fpts', elemap, lhs)

    def _get_fpts(self, meth, elemap, lhs):
        arr = [getattr(elemap[t], meth)[f, e] for t, e, f, z in lhs]
        arr = np.vstack(arr).T
        return arr.copy()

    def _get_upts(self, meth, elemap, lhs):
        arr = [getattr(elemap[t], meth)[e] for t, e, f, z in lhs]
        arr = np.vstack(arr).T
        return arr.copy()

    def _get_index(self, elemap, lhs):
        cell_nums = {c: i for i, c in enumerate(elemap)}
        return np.array([[cell_nums[t], e, f] for t, e, f, z in lhs]).T.copy()


class BaseIntInters(BaseInters):
    def __init__(self, cfg, elemap, lhs, rhs):
        super().__init__(cfg, elemap, lhs)

        self._lidx = self._get_index(elemap, lhs)
        self._ridx = self._get_index(elemap, rhs)

        if self.order > 1:
            # Delx
            dxc = [cell.dxc for cell in elemap.values()]
            self._compute_dxc(*dxc)

        # ifpts
        if cfg.get('solver-time-integrator', 'stepper') == 'lu-sgs':
            self._compute_nei_ele(elemap)

    def _compute_dxc(self, *dx):
        nface, ndims = self.nfpts, self.ndims
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx

        # Connecting vector from adjacent elements
        self._dx_adj = np.empty((ndims, nface))

        @nb.jit(nopython=True, fastmath=True)
        def compute_dxc(dx_adj, *dxc):
            for idx in range(nface):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]

                for jdx in range(ndims):
                    xl = dxc[lti][lfi, lei, jdx]
                    xr = dxc[rti][rfi, rei, jdx]

                    dx = xr - xl
                    dx_adj[jdx, idx] = dx

                    dxc[lti][lfi, lei, jdx] = dx
                    dxc[rti][rfi, rei, jdx] = -dx

        compute_dxc(self._dx_adj, *dx)

    def _compute_nei_ele(self, elemap):
        nei_ele = [ele.nei_ele for ele in elemap.values()]

        lt, le, lf = self._lidx
        rt, re, rf = self._ridx

        for idx in range(self.nfpts):
            lti, lfi, lei = lt[idx], lf[idx], le[idx]
            rti, rfi, rei = rt[idx], rf[idx], re[idx]

            if rti == lti:
                nei_ele[lti][lfi, lei] = rei
                nei_ele[rti][rfi, rei] = lei


class BaseBCInters(BaseInters):
    _reqs = None

    def __init__(self, cfg, elemap, lhs, bctype):
        super().__init__(cfg, elemap, lhs)
        self.bctype = bctype

        self._lidx = self._get_index(elemap, lhs)

        if self.order > 1:
            # Delx
            dxc = [cell.dxc for cell in elemap.values()]
            self._compute_dxc(*dxc)

        self.xf = self._get_fpts('xf', elemap, lhs)
        #xf = self._get_fpts('xf', elemap, lhs)
        #xc = self._get_upts('xc', elemap, lhs)
        # self.xf = xc + np.einsum('ij,ij->j', xf - xc,
        #                         self._vec_snorm)*self._vec_snorm

    def _compute_dxc(self, *dx):
        nface, ndims = self.nfpts, self.ndims
        lt, le, lf = self._lidx

        nf = self._vec_snorm

        # Connecting vector from adjacent elements
        self._dx_adj = np.empty((ndims, nface))

        @nb.jit(nopython=True, fastmath=True)
        def compute_dxc(dx_adj, *dxc):
            for idx in range(nface):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                # dxn = (xf - xc)*nf
                dxn = 0
                for jdx in range(ndims):
                    dxn += -dxc[lti][lfi, lei, jdx]*nf[jdx, idx]

                for jdx in range(ndims):
                    dx = 2*dxn*nf[jdx, idx]
                    dxc[lti][lfi, lei, jdx] = dx
                    dx_adj[jdx, idx] = dx

        compute_dxc(self._dx_adj, *dx)


class BaseMPIInters(BaseInters):
    def __init__(self, cfg, elemap, lhs, dest):
        super().__init__(cfg, elemap, lhs)
        self._dest = dest

        self._lidx = self._get_index(elemap, lhs)

        if self.order > 1:
            dxc = [cell.dxc for cell in elemap.values()]
            self._compute_dxc(*dxc)

    def _compute_dxc(self, *dx):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

        nface, ndims = self.nfpts, self.ndims
        lt, le, lf = self._lidx
        buf = np.empty((nface, ndims), dtype=np.float)

        # Connecting vector from adjacent elements
        self._dx_adj = np.empty((ndims, nface))

        @nb.jit(nopython=True, fastmath=True)
        def pack(buf, *dxc):
            for idx in range(nface):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                for jdx in range(ndims):
                    buf[idx, jdx] = dxc[lti][lfi, lei, jdx]

        @nb.jit(nopython=True, fastmath=True)
        def compute_dxc(dx_adj, buf, *dxc):
            for idx in range(nface):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                for jdx in range(ndims):
                    xl = dxc[lti][lfi, lei, jdx]
                    xr = buf[idx, jdx]

                    dx = xr - xl

                    dxc[lti][lfi, lei, jdx] = dx
                    dx_adj[jdx, idx] = dx

        pack(buf, *dx)
        comm.Sendrecv_replace(buf, dest=self._dest, source=self._dest)
        compute_dxc(self._dx_adj, buf, *dx)
