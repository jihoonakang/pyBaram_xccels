# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvecdiff import BaseAdvecDiffIntInters, BaseAdvecDiffBCInters, BaseAdvecDiffMPIInters
from pybaram.solvers.euler.rsolvers import get_rsolver
from pybaram.solvers.navierstokes.bcs import get_bc
from pybaram.solvers.navierstokes.visflux import make_visflux

import numpy as np


class NavierStokesIntInters(BaseAdvecDiffIntInters):
    def _make_flux(self):
        ndims, nvars = self.ndims, self.nvars
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        nf, sf = self._vec_snorm, self._mag_snorm

        # Compile Arguments
        cplargs = {
            'flux' : self.ele0.inviscid_flux_container(),
            'to_primevars' : self.ele0.to_flow_primevars(),
            'compute_mu' : self.ele0.mu_container(),
            'ndims' : ndims,
            'nvars' : nvars,
            **self._const
        }

        scheme = self.cfg.get('solver-interfaces', 'riemann-solver')
        pre, flux = get_rsolver(scheme, self.be, cplargs)
        visflux = make_visflux(self.be, cplargs)

        def comm_flux(i_begin, i_end, gradf, *uf):
            um = np.empty(nvars)
            ftmp = pre()
            fn = np.empty(nvars)

            for idx in range(i_begin, i_end):
                nfi = nf[:, idx]

                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]
                ul = uf[lti][lfi, :, lei]
                ur = uf[rti][rfi, :, rei]
                gf = gradf[:, :, idx]

                for jdx in range(nvars):
                    um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                flux(ul, ur, nfi, fn, *ftmp)
                visflux(um, gf, nfi, fn)

                for jdx in range(nvars):
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]
                    uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)


class NavierStokesMPIInters(BaseAdvecDiffMPIInters):
    def _make_flux(self):
        ndims, nvars = self.ndims, self.nvars
        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm

        # Compile Arguments
        cplargs = {
            'flux' : self.ele0.inviscid_flux_container(),
            'to_primevars' : self.ele0.to_flow_primevars(),
            'compute_mu' : self.ele0.mu_container(),
            'ndims' : ndims,
            'nvars' : nvars,
            **self._const
        }

        scheme = self.cfg.get('solver-interfaces', 'riemann-solver')
        pre, flux = get_rsolver(scheme, self.be, cplargs)
        visflux = make_visflux(self.be, cplargs)

        def comm_flux(i_begin, i_end, gradf, rhs, *uf):
            um = np.empty(nvars)
            ftmp = pre()
            fn = np.empty(nvars)

            for idx in range(i_begin, i_end):
                nfi = nf[:, idx]

                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]
                ur = rhs[:, idx]
                gf = gradf[:, :, idx]

                for jdx in range(nvars):
                    um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                flux(ul, ur, nfi, fn, *ftmp)
                visflux(um, gf, nfi, fn)

                for jdx in range(nvars):
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)


class NavierStokesBCInters(BaseAdvecDiffBCInters):
    _get_bc = get_bc

    def _make_flux(self):
        ndims, nvars = self.ndims, self.nvars
        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm

        # Compile Arguments
        cplargs = {
            'flux' : self.ele0.inviscid_flux_container(),
            'to_primevars' : self.ele0.to_flow_primevars(),
            'compute_mu' : self.ele0.mu_container(),
            'ndims' : ndims,
            'nvars' : nvars,
            **self._const
        }

        scheme = self.cfg.get('solver-interfaces', 'riemann-solver')
        pre, flux = get_rsolver(scheme, self.be, cplargs)
        visflux = make_visflux(self.be, cplargs)

        bc = self.bc

        def comm_flux(i_begin, i_end, gradf, *uf):
            ur, um = np.empty(nvars), np.empty(nvars)
            ftmp = pre()
            fn = np.empty(nvars)

            for idx in range(i_begin, i_end):
                nfi = nf[:, idx]

                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]
                gf = gradf[:, :, idx]

                bc(ul, ur, nfi)
                for jdx in range(nvars):
                    um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                flux(ul, ur, nfi, fn, *ftmp)
                visflux(um, gf, nfi, fn)

                for jdx in range(nvars):
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)


class NavierStokesSlipWallBCInters(NavierStokesBCInters):
    name = 'slip-wall'


class NavierStokesAdiaWallBCInters(NavierStokesBCInters):
    name = 'adia-wall'


class NavierStokesIsothermWallBCInters(NavierStokesBCInters):
    name = 'isotherm-wall'
    _reqs = ['cptw']

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)


class NavierStokesSupOutBCInters(NavierStokesBCInters):
    name = 'sup-out'


class NavierStokesSupInBCInters(NavierStokesBCInters):
    name = 'sup-in'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = self.primevars


class NavierStokesFarInBCInters(NavierStokesBCInters):
    name = 'far'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = self.primevars


class NavierStokesSubOutPBCInters(NavierStokesBCInters):
    name = 'sub-outp'
    _reqs = ['p']

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)


class NavierStokesSubInvBCInters(NavierStokesBCInters):
    name = 'sub-inv'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = ['rho'] + ['u', 'v', 'w'][:self.ndims]
