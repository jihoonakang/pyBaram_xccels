# -*- coding: utf-8 -*-
from pybaram.solvers.base.inters import BaseInters
from pybaram.solvers.baseadvecdiff import BaseAdvecDiffIntInters, BaseAdvecDiffBCInters, BaseAdvecDiffMPIInters
from pybaram.solvers.ranssa.visflux import make_visflux
from pybaram.solvers.ranssa.bcs import get_bc
from pybaram.solvers.euler.rsolvers import get_rsolver
from pybaram.utils.nb import dot
from pybaram.utils.np import npeval

import numpy as np
import re


class RANSSAInters(BaseInters):
    def __init__(self, be, cfg, elemap, *args, **kwargs):
        super().__init__(be, cfg, elemap, *args, **kwargs)

        self._turb_coeffs = self.ele0._turb_coeffs
        self.nturbvars = self.ele0.nturbvars

    def _make_turb_flux(self):
        ndims, nvars = self.ndims, self.nvars

        sigma = self._turb_coeffs['sigma']

        def tflux(ul, ur, mu, gf, nf, fn):
            # Convective flux
            contral = dot(ul, nf, ndims, 1)/ul[0]
            contrar = dot(ur, nf, ndims, 1)/ur[0]
            contram = 0.5*(contral + contrar)

            contrap = 0.5*(contram + abs(contram))
            contram = 0.5*(contram - abs(contram))

            # Upwind
            fn[nvars-1] = contrap*ul[nvars-1] + contram*ur[nvars-1]

            nu = mu / (ul[0] + ur[0])
            nut = 0.5*(ul[nvars-1] + ur[nvars-1])

            tau = dot(gf[:, nvars-1], nf, ndims)

            fn[nvars-1] -= 1/sigma*(nu + nut)*tau

        return self.be.compile(tflux)


class RANSSAIntInters(BaseAdvecDiffIntInters, RANSSAInters):
    def _make_flux(self):
        ndims, nvars, nfvars = self.ndims, self.nvars, self.nfvars

        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        nf, sf = self._vec_snorm, self._mag_snorm

        # Compile Arguments
        cplargs = {
            'flux' : self.ele0.flux_container(),
            'to_primevars' : self.ele0.to_flow_primevars(),
            'ndims' : ndims,
            'nfvars' : nfvars,
            **self._const
        }

        scheme = self.cfg.get('solver-interfaces', 'riemann-solver')
        pre, flux = get_rsolver(scheme, self.be, cplargs)
        compute_mu = self.ele0.mu_container()
        compute_mut = self.ele0.mut_container()
        visflux = make_visflux(self.be, cplargs)
        tflux = self._make_turb_flux()

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

                mu = compute_mu(um)
                mut = compute_mut(um, mu)
                visflux(um, gf, nfi, mu, mut, fn)

                tflux(ul, ur, mu, gf, nfi, fn)

                for jdx in range(nvars):
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]
                    uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)


class RANSSAMPIInters(BaseAdvecDiffMPIInters, RANSSAInters):
    def _make_flux(self):
        ndims, nvars, nfvars = self.ndims, self.nvars, self.nfvars

        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm

        # Compile Arguments
        cplargs = {
            'flux' : self.ele0.flux_container(),
            'to_primevars' : self.ele0.to_flow_primevars(),
            'ndims' : ndims,
            'nfvars' : nfvars,
            **self._const
        }

        scheme = self.cfg.get('solver-interfaces', 'riemann-solver')
        pre, flux = get_rsolver(scheme, self.be, cplargs)
        compute_mu = self.ele0.mu_container()
        compute_mut = self.ele0.mut_container()
        visflux = make_visflux(self.be, cplargs)
        tflux = self._make_turb_flux()

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

                mu = compute_mu(um)
                mut = compute_mut(um, mu)
                visflux(um, gf, nfi, mu, mut, fn)

                tflux(ul, ur, mu, gf, nfi, fn)

                for jdx in range(nvars):
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)


class RANSSABCInters(BaseAdvecDiffBCInters, RANSSAInters):
    _get_bc = get_bc

    def construct_bc(self):
        # BC 함수
        bcf = re.sub('-', '_', self.name)

        # BC constant
        if self._reqs:
            bcsect = 'soln-bcs-{}'.format(self.bctype)
            bcc = {k: npeval(self.cfg.getexpr(bcsect, k, self._const))
                   for k in self._reqs}
        else:
            bcc = {}

        bcc['ndims'], bcc['nvars'], bcc['nfvars'] = self.ndims, self.nvars, self.nfvars

        bcc.update(self._const)
        bcc.update(self._turb_coeffs)

        self.bc = self._get_bc(self.be, bcf, bcc)

    def _make_flux(self):
        ndims, nvars, nfvars = self.ndims, self.nvars, self.nfvars

        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm

        # Compile Arguments
        cplargs = {
            'flux' : self.ele0.flux_container(),
            'to_primevars' : self.ele0.to_flow_primevars(),
            'ndims' : ndims,
            'nfvars' : nfvars,
            **self._const
        }

        scheme = self.cfg.get('solver-interfaces', 'riemann-solver')
        pre, flux = get_rsolver(scheme, self.be, cplargs)
        compute_mu = self.ele0.mu_container()
        compute_mut = self.ele0.mut_container()
        visflux = make_visflux(self.be, cplargs)
        tflux = self._make_turb_flux()
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

                mu = compute_mu(um)
                mut = compute_mut(um, mu)
                visflux(um, gf, nfi, mu, mut, fn)

                tflux(ul, ur, mu, gf, nfi, fn)

                for jdx in range(nvars):
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)


class RANSSASupOutBCInters(RANSSABCInters):
    name = 'sup-out'


class RANSSASlipWallBCInters(RANSSABCInters):
    name = 'slip-wall'


class RANSSAAdiaWallBCInters(RANSSABCInters):
    name = 'adia-wall'


class RANSSAIsothermWallBCInters(RANSSABCInters):
    name = 'isotherm-wall'
    _reqs = ['cptw']


class RANSASSupInBCInters(RANSSABCInters):
    name = 'sup-in'

    def __init__(self, cfg, elemap, lhs, bctype):
        super().__init__(cfg, elemap, lhs, bctype)

        self._reqs = self.primevars


class RANSSAFarInBCInters(RANSSABCInters):
    name = 'far'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = self.primevars


class RANSSASubOutPBCInters(RANSSABCInters):
    name = 'sub-outp'
    _reqs = ['p']


class RANSSASubInvBCInters(RANSSABCInters):
    name = 'sub-inv'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = ['rho'] + ['u', 'v', 'w'][:self.ndims] + ['nut']
