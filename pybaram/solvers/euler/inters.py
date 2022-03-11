# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvec import BaseAdvecIntInters, BaseAdvecBCInters, BaseAdvecMPIInters
from pybaram.solvers.euler.rsolvers import get_rsolver
from pybaram.solvers.euler.bcs import get_bc

import numpy as np
import numba as nb


class EulerIntInters(BaseAdvecIntInters):
    def _make_flux(self):
        nface, ndims, nvars = self.nfpts, self.ndims, self.nvars
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        nf, sf = self._vec_snorm, self._mag_snorm

        scheme = self.cfg.get('solver-interfaces', 'riemann-solver')
        pre, flux = get_rsolver(scheme, ndims, nvars, **self._const)

        @nb.jit(nopython=True, fastmath=True)
        def comm_flux(*uf):
            ul, ur = np.empty(nvars), np.empty(nvars)
            ftmp = pre()
            fn = np.empty(nvars)
            nfi = np.empty(ndims)

            for idx in range(nface):
                for jdx in range(ndims):
                    nfi[jdx] = nf[jdx, idx]

                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]
                for jdx in range(nvars):
                    ul[jdx] = uf[lti][lfi, jdx, lei]
                    ur[jdx] = uf[rti][rfi, jdx, rei]

                flux(ul, ur, nfi, fn, *ftmp)

                for jdx in range(nvars):
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]
                    uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

        return comm_flux


class EulerMPIInters(BaseAdvecMPIInters):
    def _make_flux(self):
        nface, ndims, nvars = self.nfpts, self.ndims, self.nvars
        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm

        scheme = self.cfg.get('solver-interfaces', 'riemann-solver')
        pre, flux = get_rsolver(scheme, ndims, nvars, **self._const)

        @nb.jit(nopython=True, fastmath=True)
        def comm_flux(rhs, *uf):
            ul, ur = np.empty(nvars), np.empty(nvars)
            ftmp = pre()
            fn = np.empty(nvars)
            nfi = np.empty(ndims)

            for idx in range(nface):
                for jdx in range(ndims):
                    nfi[jdx] = nf[jdx, idx]

                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                for jdx in range(nvars):
                    ul[jdx] = uf[lti][lfi, jdx, lei]
                    ur[jdx] = rhs[jdx, idx]

                flux(ul, ur, nfi, fn, *ftmp)

                for jdx in range(nvars):
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return comm_flux


class EulerBCInters(BaseAdvecBCInters):
    _get_bc = get_bc

    def _make_flux(self):
        nface, ndims, nvars = self.nfpts, self.ndims, self.nvars

        scheme = self.cfg.get('solver-interfaces', 'riemann-solver')
        pre, flux = get_rsolver(scheme, ndims, nvars, **self._const)

        bc = self.bc

        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm,

        @nb.jit(nopython=True, fastmath=True)
        def bc_flux(*uf):
            ul, ur = np.empty(nvars), np.empty(nvars)
            ftmp = pre()
            fn = np.empty(nvars)
            nfi = np.empty(ndims)

            for idx in range(nface):
                for jdx in range(ndims):
                    nfi[jdx] = nf[jdx, idx]

                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                for jdx in range(nvars):
                    ul[jdx] = uf[lti][lfi, jdx, lei]

                bc(ul, ur, nfi)

                flux(ul, ur, nfi, fn, *ftmp)

                for jdx in range(nvars):
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return bc_flux


class EulerSupOutBCInters(EulerBCInters):
    name = 'sup-out'


class EulerSlipWallBCInters(EulerBCInters):
    name = 'slip-wall'


class EulerSupInBCInters(EulerBCInters):
    name = 'sup-in'

    def __init__(self, cfg, elemap, lhs, bctype):
        super().__init__(cfg, elemap, lhs, bctype)

        self._reqs = self.primevars


class EulerFarInBCInters(EulerBCInters):
    name = 'far'

    def __init__(self, cfg, elemap, lhs, bctype):
        super().__init__(cfg, elemap, lhs, bctype)

        self._reqs = self.primevars


class EulerSubOutPBCInters(EulerBCInters):
    name = 'sub-outp'
    _reqs = ['p']

    def __init__(self, cfg, elemap, lhs, bctype):
        super().__init__(cfg, elemap, lhs, bctype)
