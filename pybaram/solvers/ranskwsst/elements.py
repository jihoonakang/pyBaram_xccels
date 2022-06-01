# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvecdiff import BaseAdvecDiffElements
from pybaram.solvers.navierstokes import ViscousFluidElements
from pybaram.backends.types import Kernel
from pybaram.utils.nb import dot
from pybaram.utils.np import eps

import numpy as np


class RANSKWSSTFluidElements(ViscousFluidElements):
    name = 'rans-kwsst'
    nturbvars = 2

    @property
    def auxvars(self):
        return ['ydns', 'mu', 'mut']

    @property
    def primevars(self):
        return super().primevars + ['k', 'omega']

    @property
    def conservars(self):
        return super().conservars + ['rhok', 'rhoomega']

    def prim_to_conv(self, pri, cfg):
        return super().prim_to_conv(pri, cfg) + [pri[-2]*pri[0], pri[-1]*pri[0]]

    def conv_to_prim(self, con, cfg):
        return super().conv_to_prim(con, cfg) + [con[-2]/con[0], con[-1]/con[0]]    

    def mut_container(self):
        from pybaram.solvers.ranskwsst.turbulent import make_blendingF2, make_vorticity
        
        cplargs = {'ndims' : self.ndims, 'nvars' : self.nvars, 
                    **self._turb_coeffs}

        # Functions
        _vorticity = make_vorticity(self.be, cplargs)
        _f2 = make_blendingF2(self.be, cplargs)

        a1 = self._turb_coeffs['a1']

        def mut(uc, gc, mu, d):
            w = uc[-1] / uc[0]
            rk = uc[-2]

            omega = _vorticity(uc, gc)
            f2 = _f2(uc, mu, d)

            return a1*rk / max(a1*w, f2*omega)

        return self.be.compile(mut)

    def tflux_container(self):
        ndims, nvars = self.ndims, self.nvars

        def tflux(u, nf, f):
            # Convective flux for turbulent variables
            rho = u[0]
            contrav = dot(u, nf, ndims, 1)/rho

            f[0] = u[nvars-2]*contrav
            f[1] = u[nvars-1]*contrav

        return self.be.compile(tflux)

    def turb_src_container(self):
        from pybaram.solvers.ranskwsst.turbulent import make_blendingF1, make_vorticity

        cplargs = {'ndims' : self.ndims, 'nvars' : self.nvars, 
                    **self._turb_coeffs}

        # Functions
        _vorticity = make_vorticity(self.be, cplargs)
        _f1 = make_blendingF1(self.be, cplargs)

        # Constants
        nvars, ndims = self.nvars, self.ndims
        betast = self._turb_coeffs['betast']
        beta1, beta2 = self._turb_coeffs['beta1'], self._turb_coeffs['beta2']
        tgamma1, tgamma2 = self._turb_coeffs['tgamma1'], self._turb_coeffs['tgamma2']
        sigmaw2 = self._turb_coeffs['sigmaw2']
        
        def src(uc, gc, mu, mut, d, rhs, dsrc):
            rho = uc[0]
            k = uc[nvars-2] / rho
            w = uc[nvars-1] / rho
            nut = mut / rho

            # Compute dk/dx_i dw/dx_i
            kwcross = 0
            for i in range(ndims):
                rho_x = gc[i][0]
                k_x = (gc[i][nvars-2] - k*rho_x)/rho
                w_x = (gc[i][nvars-1] - w*rho_x)/rho
                kwcross += k_x*w_x

            # Vorticity
            omega = _vorticity(uc, gc)

            # SST-Vm
            bigP = mut*omega**2

            # Blending function
            f1 = _f1(uc, gc, mu, d)
            tgamma = f1*tgamma1 + (1-f1)*tgamma2
            beta = f1*beta1 + (1-f1)*beta2

            prodk = min(bigP, 20*betast*rho*w*k)
            ddestk = betast*w 
            destk = ddestk*rho*k

            prodw = tgamma / nut * prodk
            crossw = 2*(1-f1)*rho*sigmaw2/w*kwcross 
            ddestw = 2*beta*w + max(crossw, 0)/(rho*w)
            destw = beta*rho*w**2 - crossw

            rhs[nvars-2] += prodk - destk
            rhs[nvars-1] += prodw - destw

            dsrc[nvars-2] = max(ddestk, 0)
            dsrc[nvars-1] = max(ddestw, 0)

        return self.be.compile(src)

    def fix_nonPys_container(self):
        gamma, pmin = self._const['gamma'], self._const['pmin']
        ndims, nfvars, nvars = self.ndims, self.nfvars, self.nvars

        # Adhoc

        def fix_nonPhy(u):
            rho, et = u[0], u[nfvars-1]
            if rho < 0:
                u[0] = rho = eps

            p = (gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho)

            if p < pmin:
                u[nfvars - 1] = pmin/(gamma-1) + 0.5*dot(u, u, ndims, 1, 1)/rho
            
            u[nvars-2] = max(eps, u[nvars-2])
            u[nvars-1] = max(eps, u[nvars-1])

        return self.be.compile(fix_nonPhy)

class RANSKWSSTElements(BaseAdvecDiffElements, RANSKWSSTFluidElements):
    def __init__(self, be, cfg, name, eles, vcon):
        super().__init__(be, cfg, name, eles, vcon)
        self.nvars = len(self.primevars)
        self.nfvars = self.nvars - self.nturbvars

        # Constants
        cfg.get('constants', 'pmin', '1e-15')
        self._const = cfg.items('constants')

        # KW-SST Constants
        # See https://turbmodels.larc.nasa.gov/sst.html
        sect = 'solver-turbulence-coefficients'
        cfg.get(sect, 'sigmak1', '0.85')
        cfg.get(sect, 'sigmaw1', '0.5')
        cfg.get(sect, 'beta1', '0.075')
        cfg.get(sect, 'sigmak2', '1.0')
        cfg.get(sect, 'sigmaw2', '0.856')
        cfg.get(sect, 'beta2', '0.0828')
        cfg.get(sect, 'betast', '0.09')
        cfg.get(sect, 'kappa', '0.41')
        cfg.get(sect, 'a1', '0.31')
        
        self._turb_coeffs = cfg.items(sect)

        # Compute gamma1, gamma2
        beta1 = self._turb_coeffs['beta1']
        beta2 = self._turb_coeffs['beta2']
        betast = self._turb_coeffs['betast']
        kappa = self._turb_coeffs['kappa']
        sigmaw1 = self._turb_coeffs['sigmaw1']
        sigmaw2 = self._turb_coeffs['sigmaw2']

        self._turb_coeffs['tgamma1'] = beta1/betast - sigmaw1*kappa**2/np.sqrt(betast)
        self._turb_coeffs['tgamma2'] = beta2/betast - sigmaw2*kappa**2/np.sqrt(betast)

    def construct_kernels(self, vertex, xw, nreg):
        # Aux array
        nauxvars = len(self.auxvars)
        self.aux = aux = np.empty((nauxvars, self.neles))

        # 벽면까지 길이 계산
        self.ydist = aux[0]
        #self.ydist[:] = self._wall_distance(xw)
        self._wall_distance(xw, self.ydist)

        super().construct_kernels(vertex, xw, nreg)

        # Viscosity
        self.mu, self.mut = aux[1], aux[2]

        # Kernel argument 조절 및 Aux variable 초기화
        self.post.update_args(self.upts_in, self.grad, self.mu, self.mut)
        self.post()

        self.div_upts.update_args(
            self.upts_out, self.fpts, self.upts_in, self.grad,
            self.dsrc, self.mu, self.mut
        )

        # Timestep Kernel argument 조절
        self.timestep = Kernel(self._make_timestep(),
                               self.upts_in, self.mu, self.mut, self.dt)

    def _wall_distance(self, xw, ydist):
        import time
        print('Wall distance started')
        t0 = time.time()
        eles = self.eles.swapaxes(0, 1)[:,:,None]
        xw = xw[None, :]
        ydist[:] = np.array([np.average(np.linalg.norm(xc - xw, axis=2).min(axis=1)) for xc in eles])
        print('Completed Wall distance', time.time() - t0)

    def _make_timestep(self):
        ndims, nface = self.ndims, self.nface
        nflvars = self.nfvars
        vol = self._vol
        smag, svec = self._gen_snorm_fpts()
        gamma, pmin = self._const['gamma'], self._const['pmin']
        pr, prt = self._const['pr'], self._const['prt']

        def timestep(i_begin, i_end, u, mu, mut, dt, cfl):
            for idx in range(i_begin, i_end):
                rho = u[0, idx]
                et = u[nflvars-1, idx]
                rv2 = dot(u[:, idx], u[:, idx], ndims, 1, 1)/rho

                p = max((gamma - 1)*(et - 0.5*rv2), pmin)
                c = np.sqrt(gamma*p/rho)

                sum_lamdf = 0.0
                for jdx in range(nface):
                    lamdf = abs(dot(u[:, idx], svec[jdx, idx], ndims, 1)) + c
                    lamdf += (1/rho*max(4/3, gamma)*(mu[idx]/pr + mut[idx]/prt)*
                              smag[jdx, idx]/vol[idx])
                    sum_lamdf += lamdf*smag[jdx, idx]

                dt[idx] = cfl*vol[idx] / sum_lamdf

        return self.be.make_loop(self.neles, timestep)

    def make_wave_speed(self):
        ndims, nfvars = self.ndims, self.nfvars
        gamma, pmin = self._const['gamma'], self._const['pmin']
        pr, prt = self._const['pr'], self._const['prt']

        def _lambdaf(u, nf, dx, idx, mu, mut):
            rho, et = u[0], u[nfvars-1]

            contra = dot(u, nf, ndims, 1)/rho
            p = max((gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho), pmin)
            c = np.sqrt(gamma*p/rho)

            return abs(contra) + c + 1/dx/rho * max(4/3, gamma)*(mu[idx]/pr + mut[idx]/prt)

        return self.be.compile(_lambdaf)

    def make_turb_wave_speed(self):
        ndims, nvars = self.ndims, self.nvars
        sigma = 1.0

        def _lambdaf(u, nf, dx, idx, mu, mut):
            rho = u[0]
            contra = dot(u, nf, ndims, 1)/rho

            return abs(contra) + 1/dx*(mu[idx] + mut[idx])/rho/sigma

        return self.be.compile(_lambdaf)

    def _make_recon(self):
        nface, ndims, neles = self.nface, self.ndims, self.neles
        nvars, nfvars = self.nvars, self.nfvars
        op = self.dxf

        def _cal_recon(i_begin, i_end, upts, grad, lim, fpts):
            for i in range(i_begin, i_end):
                for l in range(nfvars):
                    for k in range(nface):
                        tmp = 0
                        for j in range(ndims):
                            tmp += op[k, j, i]*grad[j, l, i]
                        fpts[k, l, i] = upts[l, i] + lim[l, i]*tmp

                # First order
                for l in range(nfvars, nvars):
                    for k in range(nface):
                        fpts[k, l, i] = upts[l, i]

        return self.be.make_loop(self.neles,_cal_recon)

    def _make_div_upts(self):
        nvars = self.nvars
        nface, neles = self.nface, self.neles

        rcp_vol = self.rcp_vol
        ydist = self.ydist

        turb_src = self.turb_src_container()

        def _div_upts(i_begin, i_end, rhs, fpts, upts, grad, dsrc, mu, mut, t=0):
            for idx in range(i_begin, i_end):
                rcp_voli = rcp_vol[idx]
                for jdx in range(nvars):
                    tmp = 0.0
                    for kdx in range(nface):
                        tmp += fpts[kdx, jdx, idx]

                    rhs[jdx, idx] = -rcp_voli*tmp

                # Turbulence source term
                turb_src(upts[:, idx], grad[:, :, idx], mu[idx], mut[idx],
                         ydist[idx], rhs[:, idx], dsrc[:, idx])

        return self.be.make_loop(self.neles, _div_upts)

    def _make_post(self):
        _fix_nonPys = self.fix_nonPys_container()
        _compute_mu = self.mu_container()
        _compute_mut = self.mut_container()

        ydist = self.ydist
        muf = self._const['mu']

        def post(i_begin, i_end, upts, grad, mu, mut):
            # Update
            for idx in range(i_begin, i_end):
                _fix_nonPys(upts[:, idx])
                mu[idx] = _compute_mu(upts[:, idx])
                mut[idx] = _compute_mut(
                    upts[:, idx], grad[:,:,idx], mu[idx], ydist[idx]
                )
                mut[idx] = min(mut[idx], 100000*muf)

        return self.be.make_loop(self.neles, post)