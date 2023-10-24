# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvecdiff import BaseAdvecDiffIntInters, BaseAdvecDiffBCInters, BaseAdvecDiffMPIInters
from pybaram.solvers.euler.rsolvers import get_rsolver
from pybaram.solvers.navierstokes.bcs import get_bc
from pybaram.solvers.navierstokes.visflux import make_visflux

import numpy as np


class NavierStokesIntInters(BaseAdvecDiffIntInters):
    def _make_flux(self):
        ndims, nfvars = self.ndims, self.nfvars
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        nf, sf = self._vec_snorm, self._mag_snorm

        # Compiler arguments
        array = self.be.local_array()
        cplargs = {
            'flux' : self.ele0.flux_container(),
            'to_primevars' : self.ele0.to_flow_primevars(),
            'ndims' : ndims,
            'nfvars' : nfvars,
            'array' : array,
            **self._const
        }

        # Get numerical schems from `rsolvers.py`
        scheme = self.cfg.get('solver', 'riemann-solver')
        flux = get_rsolver(scheme, self.be, cplargs)

        # Get compiled function of viscosity and viscous flux
        compute_mu = self.ele0.mu_container()
        visflux = make_visflux(self.be, cplargs)

        def comm_flux(i_begin, i_end, gradf, *uf):
            for idx in range(i_begin, i_end):
                fn = array(nfvars)
                um = array(nfvars)

                # Normal vector
                nfi = nf[:, idx]

                # Left and right solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]
                ul = uf[lti][lfi, :, lei]
                ur = uf[rti][rfi, :, rei]
                
                # Gradient and solution at face
                gf = gradf[:, :, idx]

                for jdx in range(nfvars):
                    um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                # Compute approixmate Riemann solver
                flux(ul, ur, nfi, fn)

                # Compute viscosity and viscous flux
                mu = compute_mu(um)
                visflux(um, gf, nfi, mu, fn)

                for jdx in range(nfvars):
                    # Save it at left and right solution array
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]
                    uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)


class NavierStokesMPIInters(BaseAdvecDiffMPIInters):
    def _make_flux(self):
        ndims, nfvars = self.ndims, self.nfvars
        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm

        # Compiler arguments
        array = self.be.local_array()
        cplargs = {
            'flux' : self.ele0.flux_container(),
            'to_primevars' : self.ele0.to_flow_primevars(),
            'ndims' : ndims,
            'nfvars' : nfvars,
            'array' : array,
            **self._const
        }

        # Get numerical schems from `rsolvers.py`
        scheme = self.cfg.get('solver', 'riemann-solver')
        flux = get_rsolver(scheme, self.be, cplargs)

        # Get compiled function of viscosity and viscous flux
        compute_mu = self.ele0.mu_container()
        visflux = make_visflux(self.be, cplargs)

        def comm_flux(i_begin, i_end, gradf, rhs, *uf):
            for idx in range(i_begin, i_end):
                fn = array(nfvars)
                um = array(nfvars)

                # Normal vector
                nfi = nf[:, idx]

                # Left and right solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]
                ur = rhs[:, idx]

                # Gradient and solution at face
                gf = gradf[:, :, idx]

                for jdx in range(nfvars):
                    um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                # Compute approixmate Riemann solver
                flux(ul, ur, nfi, fn)
                
                # Compute viscosity and viscous flux
                mu = compute_mu(um)
                visflux(um, gf, nfi, mu, fn)

                for jdx in range(nfvars):
                    # Save it at left solution array
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)


class NavierStokesBCInters(BaseAdvecDiffBCInters):
    _get_bc = get_bc

    def _make_flux(self):
        ndims, nfvars = self.ndims, self.nfvars
        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm

        # Compiler arguments
        array = self.be.local_array()
        cplargs = {
            'flux' : self.ele0.flux_container(),
            'to_primevars' : self.ele0.to_flow_primevars(),
            'ndims' : ndims,
            'nfvars' : nfvars,
            'array' : array,
            **self._const
        }

        # Get numerical schems from `rsolvers.py`
        scheme = self.cfg.get('solver', 'riemann-solver')
        flux = get_rsolver(scheme, self.be, cplargs)

        # Get compiled function of viscosity and viscous flux
        compute_mu = self.ele0.mu_container()
        visflux = make_visflux(self.be, cplargs)

        # Get bc function (`self.bc` was defined at `baseadvec.inters`)
        bc = self.bc

        def comm_flux(i_begin, i_end, gradf, *uf):
            for idx in range(i_begin, i_end):
                ur = array(nfvars)
                um = array(nfvars)
                fn = array(nfvars)

                # Normal vector
                nfi = nf[:, idx]

                # Left solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]

                # Gradient at face
                gf = gradf[:, :, idx]

                # Compute BC
                bc(ul, ur, nfi)

                # Solution at face
                for jdx in range(nfvars):
                    um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                # Compute approixmate Riemann solver
                flux(ul, ur, nfi, fn)
                
                # Compute viscosity and viscous flux
                mu = compute_mu(um)
                visflux(um, gf, nfi, mu, fn)

                for jdx in range(nfvars):
                    # Save it at left solution array
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)


class NavierStokesSlipWallBCInters(NavierStokesBCInters):
    name = 'slip-wall'


class NavierStokesAdiaWallBCInters(NavierStokesBCInters):
    name = 'adia-wall'


class NavierStokesIsothermWallBCInters(NavierStokesBCInters):
    name = 'isotherm-wall'
    _reqs = ['cptw']


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


class NavierStokesSubInvBCInters(NavierStokesBCInters):
    name = 'sub-inv'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = ['rho'] + ['u', 'v', 'w'][:self.ndims]


class NavierStokesSubInpttBCInters(NavierStokesBCInters):
    name = 'sub-inptt'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = ['p0', 'cpt0', 'dir']
