# -*- coding: utf-8 -*-
import numpy as np
import numba as nb

from pybaram.solvers.baseadvec import BaseAdvecElements
from pybaram.solvers.euler.rsolvers import flux
from pybaram.utils.kernels import Kernel
from pybaram.utils.nb import dot
from pybaram.utils.np import eps


class FluidElements:   
    @property
    def primevars(self):
        return ['rho', 'p'] + [k for k in 'uvw'[:self.ndims]]

    @property
    def conservars(self):
        pri = self.primevars
        # rho,rhou,rhov,rhow,E
        return [pri[0]] + [pri[0] + v for v in pri[2:self.ndims+2]] + ['E']

    def prim_to_conv(self, pri, cfg):
        (rho, p), v = pri[:2], pri[2:self.ndims+2]
        rhov = [rho * u for u in v]
        gamma = cfg.getfloat('constants', 'gamma')
        et = p / (gamma - 1) + 0.5 * rho * sum(u * u for u in v)
        return [rho] + rhov + [et]

    def conv_to_prim(self, con, cfg):
        rho, et = con[0], con[self.ndims+1]
        v = [rhov / rho for rhov in con[1:self.ndims+1]]
        gamma = cfg.getfloat('constants', 'gamma')
        p = (gamma - 1) * (et - 0.5 * rho * sum(u * u for u in v))
        return [rho, p] + v

    def flux_container(self):
        gamma, pmin = self._const['gamma'], self._const['pmin']
        ndims, nvars = self.ndims, self.nvars

        @nb.jit(nopython=True, fastmath=True)
        def flux(u, nf, f):
            rho, et = u[0], u[nvars-1]

            contrav = dot(u, nf, ndims, 1)/rho

            p = (gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho)
            if p < pmin:
                p = pmin
                u[nvars - 1] = et = p/(gamma-1) + 0.5 * \
                    dot(u, u, ndims, 1, 1)/rho

            ht = et + p

            f[0] = rho*contrav
            for i in range(ndims):
                f[i + 1] = u[i + 1]*contrav + nf[i]*p
            f[nvars-1] = ht*contrav

        return flux

    def fix_nonPys_container(self):
        gamma, pmin = self._const['gamma'], self._const['pmin']
        ndims, nvars = self.ndims, self.nvars

        @nb.jit(nopython=True, fastmath=True)
        def fix_nonPhy(u):
            rho, et = u[0], u[nvars-1]
            if rho < 0:
                u[0] = rho = eps

            p = (gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho)

            if p < pmin:
                u[nvars - 1] = pmin/(gamma-1) + 0.5*dot(u, u, ndims, 1, 1)/rho

        return fix_nonPhy


class EulerElements(BaseAdvecElements, FluidElements):
    def __init__(self, cfg, name, eles, vcon):
        super().__init__(cfg, name, eles, vcon)
        self.nvars = len(self.primevars)

        # Constants
        cfg.get('constants', 'pmin', '1e-4')

        self._const = cfg.items('constants')

    def construct_kernels(self, vertex, xw, nreg):
        super().construct_kernels(vertex, xw, nreg)

        self.timestep = Kernel(self._make_timestep(),
                               self.upts_in, self.dt, gamma=1.4)

    def _make_timestep(self):
        neles, ndims, nface = self.neles, self.ndims, self.nface
        vol = self._vol
        smag, svec = self._gen_snorm_fpts()
        pmin = self.cfg.getfloat('constants', 'pmin')

        @nb.jit(nopython=True, fastmath=True)
        def timestep(u, dt, cfl, gamma):
            for idx in range(neles):
                rho = u[0, idx]
                et = u[-1, idx]
                rv2 = dot(u[:, idx], u[:, idx], ndims, 1, 1)/rho

                p = max((gamma - 1)*(et - 0.5*rv2), pmin)
                c = np.sqrt(gamma*p/rho)

                sum_lamdf = 0.0
                for jdx in range(nface):
                    lamdf = abs(dot(u[:, idx], svec[jdx, idx], ndims, 1)) + c
                    sum_lamdf += lamdf*smag[jdx, idx]

                dt[idx] = cfl*vol[idx] / sum_lamdf

        return timestep

    def make_wave_speed(self):
        ndims, nvars = self.ndims, self.nvars
        gamma, pmin = self._const['gamma'], self._const['pmin']

        @nb.jit(nopython=True, fastmath=True)
        def _lambdaf(u, nf, *args):
            rho, et = u[0], u[nvars-1]

            contra = dot(u, nf, ndims, 1)/rho
            p = max((gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho), pmin)
            c = np.sqrt(gamma*p/rho)

            return abs(contra) + c

        return _lambdaf
