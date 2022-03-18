# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvec import BaseAdvecIntInters, BaseAdvecBCInters, BaseAdvecMPIInters
from pybaram.solvers.euler.rsolvers import get_rsolver
from pybaram.solvers.euler.bcs import get_bc

import numpy as np


class EulerIntInters(BaseAdvecIntInters):
    def _make_flux(self):
        ndims, nvars = self.ndims, self.nvars
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        nf, sf = self._vec_snorm, self._mag_snorm

        scheme = self.cfg.get('solver-interfaces', 'riemann-solver')
        pre, flux = get_rsolver(scheme, ndims, nvars, **self._const)

        def comm_flux(i_begin, i_end, *uf):
            ul, ur = np.empty(nvars), np.empty(nvars)
            ftmp = pre()
            fn = np.empty(nvars)
            nfi = np.empty(ndims)

            for idx in range(i_begin, i_end):
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

        return self.be.make_loop(self.nfpts, comm_flux)


class EulerMPIInters(BaseAdvecMPIInters):
    def _make_flux(self):
        ndims, nvars = self.ndims, self.nvars
        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm

        scheme = self.cfg.get('solver-interfaces', 'riemann-solver')
        pre, flux = get_rsolver(scheme, ndims, nvars, **self._const)

        def comm_flux(i_begin, i_end, rhs, *uf):
            ul, ur = np.empty(nvars), np.empty(nvars)
            ftmp = pre()
            fn = np.empty(nvars)
            nfi = np.empty(ndims)

            for idx in range(i_begin, i_end):
                for jdx in range(ndims):
                    nfi[jdx] = nf[jdx, idx]

                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                for jdx in range(nvars):
                    ul[jdx] = uf[lti][lfi, jdx, lei]
                    ur[jdx] = rhs[jdx, idx]

                flux(ul, ur, nfi, fn, *ftmp)

                for jdx in range(nvars):
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)


class EulerBCInters(BaseAdvecBCInters):
    _get_bc = get_bc

    def _make_flux(self):
        ndims, nvars = self.ndims, self.nvars

        scheme = self.cfg.get('solver-interfaces', 'riemann-solver')
        pre, flux = get_rsolver(scheme, ndims, nvars, **self._const)

        bc = self.bc

        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm,

        def bc_flux(i_begin, i_end, *uf):
            ul, ur = np.empty(nvars), np.empty(nvars)
            ftmp = pre()
            fn = np.empty(nvars)
            nfi = np.empty(ndims)

            for idx in range(i_begin, i_end):
                for jdx in range(ndims):
                    nfi[jdx] = nf[jdx, idx]

                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                for jdx in range(nvars):
                    ul[jdx] = uf[lti][lfi, jdx, lei]

                bc(ul, ur, nfi)

                flux(ul, ur, nfi, fn, *ftmp)

                for jdx in range(nvars):
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, bc_flux)


class EulerSupOutBCInters(EulerBCInters):
    name = 'sup-out'


class EulerSlipWallBCInters(EulerBCInters):
    name = 'slip-wall'


class EulerSupInBCInters(EulerBCInters):
    name = 'sup-in'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = self.primevars


class EulerFarInBCInters(EulerBCInters):
    name = 'far'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = self.primevars


class EulerSubOutPBCInters(EulerBCInters):
    name = 'sub-outp'
    _reqs = ['p']

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)
