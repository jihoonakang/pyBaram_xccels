# -*- coding: utf-8 -*-
from pybaram.solvers.euler.elements import FluidElements
from pybaram.solvers.baseadvecdiff import BaseAdvecDiffElements
from pybaram.backends.types import Kernel
from pybaram.utils.nb import dot

import numpy as np


class ViscousFluidElements(FluidElements):
    name = 'navier-stokes'
    
    @property
    def auxvars(self):
        return ['mu']

    def mu_container(self):
        mu = self._const['mu']

        # Constant viscosity
        def compute_mu(*args):
            return mu

        return self.be.compile(compute_mu)


class NavierStokesElements(BaseAdvecDiffElements, ViscousFluidElements):
    def __init__(self, be, cfg, name, eles, vcon):
        super().__init__(be, cfg, name, eles, vcon)
        self.nvars = len(self.primevars)
        self.nfvars = self.nvars

        # Constants
        cfg.get('constants', 'pmin', '1e-15')
        self._const = cfg.items('constants')

    def construct_kernels(self, vertex, xw, nreg):
        super().construct_kernels(vertex, xw, nreg)

        # Aux array
        nauxvars = len(self.auxvars)
        self.aux = aux = np.empty((nauxvars, self.neles))

        # Viscosity
        self.mu = aux[0]

        # Kernel argument 조절 및 Aux variable 초기화
        self.post.update_args(self.upts_in, self.mu)
        self.post()

        # Timestep Kernel argument 조절
        self.timestep = Kernel(self._make_timestep(),
                               self.upts_in, self.mu, self.dt)

    def _make_timestep(self):
        ndims, nface, nfvars = self.ndims, self.nface, self.nfvars
        vol = self._vol
        smag, svec = self._gen_snorm_fpts()
        gamma, pmin = self._const['gamma'], self._const['pmin']
        pr = self._const['pr']

        def timestep(i_begin, i_end, u, mu, dt, cfl):
            for idx in range(i_begin, i_end):
                rho = u[0, idx]
                et = u[nfvars-1, idx]
                rv2 = dot(u[:, idx], u[:, idx], ndims, 1, 1)/rho

                p = max((gamma - 1)*(et - 0.5*rv2), pmin)
                c = np.sqrt(gamma*p/rho)

                sum_lamdf = 0.0
                for jdx in range(nface):
                    lamdf = abs(dot(u[:, idx], svec[jdx, idx], ndims, 1)) + c
                    lamdf += (1/rho*max(4/3, gamma)*mu[idx]/pr *
                              smag[jdx, idx]/vol[idx])
                    sum_lamdf += lamdf*smag[jdx, idx]

                dt[idx] = cfl*vol[idx] / sum_lamdf

        return self.be.make_loop(self.neles, timestep)

    def make_wave_speed(self):
        ndims, nfvars = self.ndims, self.nfvars
        gamma, pmin = self._const['gamma'], self._const['pmin']
        pr = self._const['pr']

        def _lambdaf(u, nf, dx, idx, mu, *args):
            rho, et = u[0], u[nfvars-1]

            contra = dot(u, nf, ndims, 1)/rho
            p = max((gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho), pmin)
            c = np.sqrt(gamma*p/rho)

            return abs(contra) + c + 1/dx/rho * max(4/3, gamma)*mu[idx]/pr

        return self.be.compile(_lambdaf)

    def _make_post(self):
        _fix_nonPys = self.fix_nonPys_container()
        _compute_mu = self.mu_container()

        def post(i_begin, i_end, upts, mu):
            # Update
            for idx in range(i_begin, i_end):
                _fix_nonPys(upts[:, idx])
                mu[idx] = _compute_mu(upts[:, idx])

        return self.be.make_loop(self.neles, post)