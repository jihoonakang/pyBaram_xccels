# -*- coding: utf-8 -*-
import numpy as np


class BaseInters:
    name = 'base'

    def __init__(self, be, cfg, elemap, lhs):
        # Save arguments
        self.be = be
        self.cfg = cfg

        # Dimensions
        self.nfpts = len(lhs)
        self.ele0 = ele0 = elemap[next(iter(elemap))]
        self.ndims, self.nvars, self.nfvars = ele0.ndims, ele0.nvars, ele0.nfvars

        # Primitive variables
        self.primevars = ele0.primevars

        # Collect constants
        self._const = cfg.items('constants')

        # Order of spatial discretization
        self.order = cfg.getint('solver', 'order', 1)

        # Normal Vector of face
        self._mag_snorm = self._get_fpts('_mag_snorm_fpts', elemap, lhs)[0]
        self._vec_snorm = self._get_fpts('_vec_snorm_fpts', elemap, lhs)

    def _get_fpts(self, meth, elemap, lhs):
        # Get element property at face and sort it
        arr = [getattr(elemap[t], meth)[f, e] for t, e, f, z in lhs]
        arr = np.vstack(arr).T
        return arr.copy()

    def _get_upts(self, meth, elemap, lhs):
        # Get element property and sort it
        arr = [getattr(elemap[t], meth)[e] for t, e, f, z in lhs]
        arr = np.vstack(arr).T
        return arr.copy()

    def _get_index(self, elemap, lhs):
        # Parse index of elements and make index of each face point
        cell_nums = {c: i for i, c in enumerate(elemap)}
        return np.array([[cell_nums[t], e, f] for t, e, f, z in lhs]).T.copy()


class BaseIntInters(BaseInters):
    def __init__(self, be, cfg, elemap, lhs, rhs):
        super().__init__(be, cfg, elemap, lhs)

        self._lidx = self._get_index(elemap, lhs)
        self._ridx = self._get_index(elemap, rhs)

        if self.order > 1:
            # Delx = xc2 - xc1 across face
            dxc = [cell.dxc for cell in elemap.values()]
            self._compute_dxc(*dxc)

        # Construct neighboring element within current Elements
        self._compute_nei_ele(elemap)

    def _compute_dxc(self, *dx):
        nface, ndims = self.nfpts, self.ndims
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx

        # Connecting vector from adjacent elements
        self._dx_adj = np.empty((ndims, nface))

        def compute_dxc(i_begin, i_end, dx_adj, *dxc):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]

                for jdx in range(ndims):
                    xl = dxc[lti][lfi, lei, jdx]
                    xr = dxc[rti][rfi, rei, jdx]

                    dx = xr - xl
                    dx_adj[jdx, idx] = dx

                    dxc[lti][lfi, lei, jdx] = dx
                    dxc[rti][rfi, rei, jdx] = -dx

        # Compute dx_adj
        self.be.make_loop(nface, compute_dxc)(self._dx_adj, *dx)

    def _compute_nei_ele(self, elemap):
        # List of nei_ele
        nei_ele = [ele.nei_ele for ele in elemap.values()]

        lt, le, lf = self._lidx
        rt, re, rf = self._ridx

        for idx in range(self.nfpts):
            lti, lfi, lei = lt[idx], lf[idx], le[idx]
            rti, rfi, rei = rt[idx], rf[idx], re[idx]

            if rti == lti:
                # If same Elements, save neigboring element
                nei_ele[lti][lfi, lei] = rei
                nei_ele[rti][rfi, rei] = lei


class BaseBCInters(BaseInters):
    _reqs = None

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs)
        self.bctype = bctype

        self._lidx = self._get_index(elemap, lhs)

        if self.order > 1:
            # Delx across face
            dxc = [cell.dxc for cell in elemap.values()]
            self._compute_dxc(*dxc)

        # Compute face center at boundary
        self.xf = self._get_fpts('xf', elemap, lhs)

    def _compute_dxc(self, *dx):
        nface, ndims = self.nfpts, self.ndims
        lt, le, lf = self._lidx

        nf = self._vec_snorm

        # Connecting vector from adjacent elements
        self._dx_adj = np.empty((ndims, nface))

        def compute_dxc(i_begin, i_end, dx_adj, *dxc):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                # Compute normal component of (xf - xc) as dxn
                dxn = 0
                for jdx in range(ndims):
                    dxn += -dxc[lti][lfi, lei, jdx]*nf[jdx, idx]

                for jdx in range(ndims):
                    dx = 2*dxn*nf[jdx, idx]
                    dxc[lti][lfi, lei, jdx] = dx
                    dx_adj[jdx, idx] = dx

        # Compute dx_adj
        self.be.make_loop(nface, compute_dxc)(self._dx_adj, *dx)


class BaseMPIInters(BaseInters):
    def __init__(self, be, cfg, elemap, lhs, dest):
        super().__init__(be, cfg, elemap, lhs)
        self._dest = dest

        self._lidx = self._get_index(elemap, lhs)

        if self.order > 1:
            # Delx = xc2 - xc1 across face
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

        def pack(i_begin, i_end, buf, *dxc):
            # Save dxc to buf for communication
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                for jdx in range(ndims):
                    buf[idx, jdx] = dxc[lti][lfi, lei, jdx]

        def compute_dxc(i_begin, i_end, dx_adj, buf, *dxc):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                for jdx in range(ndims):
                    xl = dxc[lti][lfi, lei, jdx]
                    xr = buf[idx, jdx]

                    dx = xr - xl

                    dxc[lti][lfi, lei, jdx] = dx
                    dx_adj[jdx, idx] = dx

        # Pack dx
        self.be.make_loop(nface, pack)(buf, *dx)

        # Exchange halo
        comm.Sendrecv_replace(buf, dest=self._dest, source=self._dest)

        # Compute dxc
        self.be.make_loop(nface, compute_dxc)(self._dx_adj, buf, *dx)
