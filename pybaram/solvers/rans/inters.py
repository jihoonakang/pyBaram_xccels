# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvecdiff import BaseAdvecDiffIntInters, BaseAdvecDiffBCInters, BaseAdvecDiffMPIInters
from pybaram.solvers.rans.visflux import make_visflux
from pybaram.solvers.euler.rsolvers import get_rsolver
from pybaram.utils.np import npeval

import numpy as np
import re


class RANSIntInters(BaseAdvecDiffIntInters):
    def construct_kernels(self, elemap):
        # Wall distance at face
        ydistf = [cell.ydist for cell in elemap.values()]
        self.ydist = np.array([ydistf[t][e]  for (t, e, _) in self._lidx.T])
        
        # Call Parent method
        super().construct_kernels(elemap)

    def _make_flux(self):
        ndims, nvars, nfvars = self.ndims, self.nvars, self.nfvars

        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        nf, sf = self._vec_snorm, self._mag_snorm
        ydist = self.ydist

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
        compute_mut = self.ele0.mut_container()
        visflux = make_visflux(self.be, cplargs)

        # Get turbulence flux from `turbulent.py`
        tflux = self._make_turb_flux()

        def comm_flux(i_begin, i_end, gradf, *uf):
            for idx in range(i_begin, i_end):
                fn = array((nvars,))
                um = array((nvars,))

                # Normal vector and wall distance (ydns)
                nfi = nf[:, idx]
                ydnsi = ydist[idx]

                # Left and right solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]
                ul = uf[lti][lfi, :, lei]
                ur = uf[rti][rfi, :, rei]

                # Gradient and solution at face
                gf = gradf[:, :, idx]

                for jdx in range(nvars):
                    um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                # Compute approixmate Riemann solver
                flux(ul, ur, nfi, fn)
                
                # Compute viscosity and viscous flux
                mu = compute_mu(um)
                mut = compute_mut(um, gf, mu, ydnsi)
                visflux(um, gf, nfi, mu, mut, fn)

                # Compute turbulent flux
                tflux(ul, ur, um, gf, nfi, ydnsi, mu, mut, fn)

                for jdx in range(nvars):
                    # Save it at left and right solution array
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]
                    uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)


class RANSMPIInters(BaseAdvecDiffMPIInters):
    def construct_kernels(self, elemap):
        # Wall distance at face
        ydistf = [cell.ydist for cell in elemap.values()]
        self.ydist = np.array([ydistf[t][e]  for (t, e, _) in self._lidx.T])
        
        # Call Parent method
        super().construct_kernels(elemap)

    def _make_flux(self):
        ndims, nvars, nfvars = self.ndims, self.nvars, self.nfvars

        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm
        ydist = self.ydist

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
        compute_mut = self.ele0.mut_container()
        visflux = make_visflux(self.be, cplargs)

        # Get turbulence flux from `turbulent.py`
        tflux = self._make_turb_flux()

        def comm_flux(i_begin, i_end, gradf, rhs, *uf):
            for idx in range(i_begin, i_end):
                fn = array((nvars,))
                um = array((nvars,))

                # Normal vector and wall distance (ydns)
                nfi = nf[:, idx]
                ydnsi = ydist[idx]

                # Left and right solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]
                ur = rhs[:, idx]

                # Gradient and solution at face
                gf = gradf[:, :, idx]

                for jdx in range(nvars):
                    um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                # Compute approixmate Riemann solver
                flux(ul, ur, nfi, fn)
                
                # Compute viscosity and viscous flux
                mu = compute_mu(um)
                mut = compute_mut(um, gf, mu, ydnsi)
                visflux(um, gf, nfi, mu, mut, fn)

                # Compute turbulent flux
                tflux(ul, ur, um, gf, nfi, ydnsi, mu, mut, fn)

                for jdx in range(nvars):
                    # Save it at left solution array
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)


class RANSBCInters(BaseAdvecDiffBCInters):
    is_vis_wall = False

    def construct_bc(self):
        # Parse BC function name
        bcf = re.sub('-', '_', self.name)

        # Constants for BC function
        if self._reqs:
            bcsect = 'soln-bcs-{}'.format(self.bctype)
            bcc = {k: npeval(self.cfg.getexpr(bcsect, k, self._const))
                   for k in self._reqs}
        else:
            bcc = {}

        bcc['ndims'], bcc['nvars'], bcc['nfvars'] = self.ndims, self.nvars, self.nfvars

        bcc.update(self._const)
        bcc.update(self._turb_coeffs)

        # Get bc from `bcs.py` (in rans...) and compile them
        self.bc = self._get_bc(self.be, bcf, bcc)

    def construct_kernels(self, elemap):
        # Wall distance at face
        ydistf = [cell.ydist for cell in elemap.values()]
        self.ydist = np.array([ydistf[t][e]  for (t, e, _) in self._lidx.T])
        
        # Call Parent method
        super().construct_kernels(elemap)

    def _make_delu(self):
        nvars, ndims = self.nvars, self.ndims
        lt, le, lf = self._lidx
        nf = self._vec_snorm
        ydist = self.ydist

        # Compile functions
        array = self.be.local_array()
        compute_mu = self.ele0.mu_container()

        bc = self.bc

        def compute_delu(i_begin, i_end, *uf):
            for idx in range(i_begin, i_end):
                ur = array((nvars,))
                nfi = nf[:, idx]

                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                ul = uf[lti][lfi, :, lei]
                
                mul = compute_mu(ul)
                bc(ul, ur, nfi, mul, ydist[idx])

                for jdx in range(nvars):
                    du = ur[jdx] - ul[jdx]
                    uf[lti][lfi, jdx, lei] = du

        return self.be.make_loop(self.nfpts, compute_delu)
        
    def _make_flux(self):
        ndims, nvars, nfvars = self.ndims, self.nvars, self.nfvars

        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm
        ydist = self.ydist

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
        compute_mut = self.ele0.mut_container()
        visflux = make_visflux(self.be, cplargs)

        # Get turbulence flux from `turbulent.py`
        tflux = self._make_turb_flux()

        # Get bc function (`self.bc` was defined at `baseadvec.inters`)
        bc = self.bc

        def comm_flux(i_begin, i_end, gradf, *uf):
            for idx in range(i_begin, i_end):
                fn = array((nvars,))
                um = array((nvars,))
                ur = array((nvars,))

                # Normal vector and wall distance (ydns)
                nfi = nf[:, idx]
                ydnsi = ydist[idx]

                # Left solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]

                # Gradient at face
                gf = gradf[:, :, idx]

                # Viscosity from left solution
                mul = compute_mu(ul)

                # Compute BC
                bc(ul, ur, nfi, mul, ydnsi)

                # Solution at face
                for jdx in range(nvars):
                    um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                # Compute approixmate Riemann solver
                flux(ul, ur, nfi, fn)
                
                # Compute viscosity and viscous flux
                mu = compute_mu(um)
                mut = compute_mut(um, gf, mu, ydnsi)
                visflux(um, gf, nfi, mu, mut, fn)

                # Compute turbulent flux
                tflux(ul, ur, um, gf, nfi, ydnsi, mu, mut, fn)

                for jdx in range(nvars):
                    # Save it at left solution array
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)


class RANSSlipWallBCInters(RANSBCInters):
    name = 'slip-wall'


class RANSAdiaWallBCInters(RANSBCInters):
    name = 'adia-wall'
    is_vis_wall = True


class RANSIsothermWallBCInters(RANSBCInters):
    name = 'isotherm-wall'
    is_vis_wall = True
    _reqs = ['cptw']


class RANSSupOutBCInters(RANSBCInters):
    name = 'sup-out'


class RANSSupInBCInters(RANSBCInters):
    name = 'sup-in'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = self.primevars


class RANSFarBCInters(RANSBCInters):
    name = 'far'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = self.primevars


class RANSSubOutPBCInters(RANSBCInters):
    name = 'sub-outp'
    _reqs = ['p']


class RANSSubInvBCInters(RANSBCInters):
    name = 'sub-inv'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = ['rho'] + ['u', 'v', 'w'][:self.ndims] + self.primevars[self.nfvars:]


class RANSSubInpttBCInters(RANSBCInters):
    name = 'sub-inptt'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = ['p0', 't0', 'dir', 'R'] + self.primevars[self.nfvars:]