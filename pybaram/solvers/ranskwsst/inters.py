# -*- coding: utf-8 -*-
from pybaram.solvers.base.inters import BaseInters
from pybaram.solvers.rans.inters import RANSIntInters, RANSBCInters, RANSMPIInters
from pybaram.solvers.rans.inters import (RANSSlipWallBCInters, RANSAdiaWallBCInters, RANSIsothermWallBCInters,
                                         RANSSupOutBCInters, RANSSupInBCInters, RANSFarBCInters,
                                         RANSSubOutPBCInters, RANSSubInvBCInters)
from pybaram.solvers.ranskwsst.bcs import get_bc
from pybaram.utils.nb import dot


class RANSKWSSTInters(BaseInters):
    def __init__(self, be, cfg, elemap, *args, **kwargs):
        super().__init__(be, cfg, elemap, *args, **kwargs)

        self._turb_coeffs = self.ele0._turb_coeffs
        self.nturbvars = self.ele0.nturbvars

    def _make_turb_flux(self):
        from pybaram.solvers.ranskwsst.turbulent import make_blendingF1

        ndims, nvars = self.ndims, self.nvars
        sigmak1, sigmak2 = self._turb_coeffs['sigmak1'], self._turb_coeffs['sigmak2']
        sigmaw1, sigmaw2 = self._turb_coeffs['sigmaw1'], self._turb_coeffs['sigmaw2']

        cplargs = {'ndims' : ndims, 'nvars' : nvars, **self._turb_coeffs}
        _f1 = make_blendingF1(self.be, cplargs)

        def tflux(ul, ur, um, gf, nf, ydist, mu, mut, fn):
             # Convective flux
            contral = dot(ul, nf, ndims, 1)/ul[0]
            contrar = dot(ur, nf, ndims, 1)/ur[0]
            contram = 0.5*(contral + contrar)

            contrap = 0.5*(contram + abs(contram))
            contram = 0.5*(contram - abs(contram))

            # Upwind
            fn[nvars-2] = contrap*ul[nvars-2] + contram*ur[nvars-2]
            fn[nvars-1] = contrap*ul[nvars-1] + contram*ur[nvars-1]

            # Viscous
            f1 = _f1(um, gf, mu, ydist)
            sigmak = f1*sigmak1 + (1-f1)*sigmak2
            sigmaw = f1*sigmaw1 + (1-f1)*sigmaw2

            tauk, tauw = 0, 0
            rho = um[0]
            for i in range(ndims):
                rho_x = gf[i][0]
                k_x = (gf[i][nvars-2] - um[nvars-2]*rho_x/rho)/rho
                w_x = (gf[i][nvars-1] - um[nvars-1]*rho_x/rho)/rho

                tauk += k_x*nf[i]
                tauw += w_x*nf[i]

            fn[nvars-2] -= (mu + sigmak*mut)*tauk
            fn[nvars-1] -= (mu + sigmaw*mut)*tauw
        
        return self.be.compile(tflux)


class RANSKWSSTIntInters(RANSIntInters, RANSKWSSTInters):
    pass


class RANSKWSSTMPIInters(RANSMPIInters, RANSKWSSTInters):
    pass


class RANSKWSSTBCInters(RANSBCInters, RANSKWSSTInters):
    _get_bc = get_bc

    def _make_delu(self):
        nvars, ndims = self.nvars, self.ndims
        lt, le, lf = self._lidx
        nf = self._vec_snorm
        ydist = self.ydist

        # Compile functions
        compute_mu = self.ele0.mu_container()

        bc = self.bc

        def compute_delu(i_begin, i_end, *uf):
            ur = np.empty(nvars)

            for idx in range(i_begin, i_end):
                nfi = nf[:, idx]

                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                ul = uf[lti][lfi, :, lei]
                
                mul = compute_mu(ul)
                bc(ul, ur, nfi, mul, ydist[idx])

                for jdx in range(nvars):
                    du = ur[jdx] - ul[jdx]
                    uf[lti][lfi, jdx, lei] = du

        return self.be.make_loop(self.nfpts, compute_delu)


class RANSKWSSTSlipWallBCInters(RANSKWSSTBCInters, RANSSlipWallBCInters):
    pass


class RANSKWSSTAdiaWallBCInters(RANSKWSSTBCInters, RANSAdiaWallBCInters):
    pass


class RANSKWSSTIsothermWallBCInters(RANSKWSSTBCInters, RANSIsothermWallBCInters):
    pass


class RANSKWSSTSupOutBCInters(RANSKWSSTBCInters, RANSSupOutBCInters):
    pass


class RANSKWSSTSupInBCInters(RANSKWSSTBCInters, RANSSupInBCInters):
    pass


class RANSKWSSTFarBCInters(RANSKWSSTBCInters, RANSFarBCInters):
    pass


class RANSKWSSTSubOutPBCInters(RANSKWSSTBCInters, RANSSubOutPBCInters):
    pass


class RANSKWSSTSubInvBCInters(RANSKWSSTBCInters, RANSSubInvBCInters):
    pass