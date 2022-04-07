# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvecdiff import BaseAdvecDiffElements
from pybaram.solvers.navierstokes import ViscousFluidElements
from pybaram.backends.types import Kernel
from pybaram.utils.nb import dot

import numpy as np


class RANSSAFluidElements(ViscousFluidElements):
    name = 'rans-sa'
    nturbvars = 1

    @property
    def auxvars(self):
        return ['ydns', 'mu', 'mut']

    @property
    def primevars(self):
        return super().primevars + ['nut']

    @property
    def conservars(self):
        return super().conservars + ['nut']

    def prim_to_conv(self, pri, cfg):
        return super().prim_to_conv(pri, cfg) + [pri[-1]]

    def conv_to_prim(self, con, cfg):
        return super().conv_to_prim(con, cfg) + [con[-1]]

    def mut_container(self):
        nvars = self.nvars

        # Constants
        cv1 = self._turb_coeffs['cv1']
        cv13 = cv1**3

        def mut(u, mu):
            # Dynamic viscosity
            rho, nut = u[0], u[nvars-1]
            nu = mu/rho

            # functions
            xi = nut / nu
            fv1 = xi**3 / (xi**3 + cv13)

            return max(rho*nut*fv1, 0)

        return self.be.compile(mut)

    def tflux_container(self):
        ndims, nvars = self.ndims, self.nvars

        def tflux(u, nf, f):
            rho = u[0]
            contrav = dot(u, nf, ndims, 1)/rho

            f[0] = u[nvars-1]*contrav

        return self.be.compile(tflux)

    def turb_src_container(self):
        from pybaram.solvers.ranssa.turbulent import make_vorticity
        from pybaram.utils.np import eps
        
        ndims, nvars = self.ndims, self.nvars

        # Turbulent constants
        cv1 = self._turb_coeffs['cv1']
        cb1, cb2 = self._turb_coeffs['cb1'], self._turb_coeffs['cb2']
        cw2, cw3 = self._turb_coeffs['cw2'], self._turb_coeffs['cw3']
        sigma, kappa = self._turb_coeffs['sigma'], self._turb_coeffs['kappa']

        cv13 = cv1**3
        cw1 = cb1/kappa**2 + (1 + cb2)/sigma
        
        # Functions
        cplargs = {'ndims' : self.ndims, 'nvars' : self.nvars, 
                    **self._turb_coeffs}
        _vorticity = make_vorticity(self.be, cplargs)

        def src(uc, gc, mu, d, rhs, dsrc):
            # nut and dnut
            nut = uc[nvars-1]
            dnut2 = 0
            for i in range(ndims):
                dnut2 += gc[i][nvars-1]**2

            # Magnitude of vorticity
            omega = _vorticity(uc, gc)

            # SA functions
            nu = mu / uc[0]
            xi = nut / nu
            fv1 = xi**3 / (xi**3 + cv13)
            fv2 = 1 - nut / (nu + nut*fv1)
            Shat = omega + nut/(kappa*d)**2*fv2
            Shat = max(Shat, 1000*eps)
        
             # Production
            prod = cb1*Shat*nut

            # Destruction
            r = min(nut/(Shat*(kappa*d)**2), 10)
            g = r + cw2*(r**6 - r)
            glim = ((1 + cw3**6)/(g**6 + cw3**6))**(1/6)
            fw = g*glim
            dest = cw1*fw*(nut/d)**2

            # Difference
            diff = cb2/sigma*dnut2

            # Implicit term
            ddest = cw1*2*fw*nut/d**2

            rhs[nvars-1] += prod - dest + diff
            dsrc[nvars-1] = max(ddest, 0)  # - dprod
        
        return self.be.compile(src)


class RANSSAElements(BaseAdvecDiffElements, RANSSAFluidElements):
    def __init__(self, be, cfg, name, eles, vcon):
        super().__init__(be, cfg, name, eles, vcon)
        self.nvars = len(self.primevars)
        self.nfvars = self.nvars - self.nturbvars

        # Constants
        cfg.get('constants', 'pmin', '1e-4')

        # SA Constants
        # See https://turbmodels.larc.nasa.gov/spalart.html#sa
        sect = 'solver-turbulence-coefficients'
        cfg.get(sect, 'cv1', '7.1')
        cfg.get(sect, 'cb1', '0.1355')
        cfg.get(sect, 'cb2', '0.622')
        cfg.get(sect, 'sigma', '2/3')
        cfg.get(sect, 'kappa', '0.41')
        cfg.get(sect, 'cw2', '0.3')
        cfg.get(sect, 'cw3', '2')
        cfg.get(sect, 'ct3', '1.2')
        cfg.get(sect, 'ct4', '0.5')

        self._const = cfg.items('constants')
        self._turb_coeffs = cfg.items(sect)

    def construct_kernels(self, vertex, xw, nreg):
        # Aux array
        nauxvars = len(self.auxvars)
        self.aux = aux = np.empty((nauxvars, self.neles))

        # 벽면까지 길이 계산
        self.ydist = aux[0]
        self.ydist[:] = self._wall_distance(xw)

        # Viscosity
        self.mu, self.mut = aux[1], aux[2]

        super().construct_kernels(vertex, xw, nreg)

        # Kernel argument 조절 및 Aux variable 초기화
        self.post.update_args(self.upts_in, self.mu, self.mut)
        self.post()

        self.div_upts.update_args(
            self.upts_out, self.fpts, self.upts_in, self.grad,
            self.mu, self.dsrc
        )

        # Timestep Kernel argument 조절
        self.timestep = Kernel(self._make_timestep(),
                               self.upts_in, self.mu, self.mut, self.dt)

    def _wall_distance(self, xw):
        # 벽면에서 부터 길이 계산
        return np.array([
            (np.linalg.norm(xw - xi[:, None], axis=0).min()) for xi in self.xc]
        )

    def _make_div_upts(self):
        ndims, nvars, nface = self.ndims, self.nvars, self.nface

        rcp_vol = self.rcp_vol
        ydist = self.ydist

        turb_src = self.turb_src_container()

        def _div_upts(i_begin, i_end, rhs, fpts, upts, grad, mu, dsrc, t=0):
            for idx in range(i_begin, i_end):
                rcp_voli = rcp_vol[idx]
                for jdx in range(nvars):
                    tmp = 0.0
                    for kdx in range(nface):
                        tmp += fpts[kdx, jdx, idx]

                    rhs[jdx, idx] = -rcp_voli*tmp

                # Turbulence source term
                turb_src(upts[:, idx], grad[:, :, idx], mu[idx],
                         ydist[idx], rhs[:, idx], dsrc[:, idx])

        return self.be.make_loop(self.neles, _div_upts)

    def _make_post(self):
        _fix_nonPys = self.fix_nonPys_container()
        _compute_mu = self.mu_container()
        _compute_mut = self.mut_container()

        def post(i_begin, i_end, upts, mu, mut):
            # Update
            for idx in range(i_begin, i_end):
                _fix_nonPys(upts[:, idx])
                mu[idx] = _compute_mu(upts[:, idx])
                mut[idx] = _compute_mut(upts[:, idx], mu[idx])

        return self.be.make_loop(self.neles, post)   

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
        sigma = self._turb_coeffs['sigma']

        def _lambdaf(u, nf, dx, idx, mu, *args):
            rho = u[0]
            contra = dot(u, nf, ndims, 1)/rho

            nu = mu[idx]/rho
            nut = u[nvars-1]

            return abs(contra) + 1/dx*(nu + nut)/sigma

        return self.be.compile(_lambdaf)