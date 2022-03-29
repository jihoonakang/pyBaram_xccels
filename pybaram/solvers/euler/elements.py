# -*- coding: utf-8 -*-
import numpy as np

from pybaram.solvers.baseadvec import BaseAdvecElements
from pybaram.backends.types import Kernel
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
        ndims, nfvars = self.ndims, self.nfvars

        def flux(u, nf, f):
            rho, et = u[0], u[nfvars-1]

            contrav = dot(u, nf, ndims, 1)/rho

            p = (gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho)
            if p < pmin:
                p = pmin
                u[nfvars - 1] = et = p/(gamma-1) + 0.5 * \
                    dot(u, u, ndims, 1, 1)/rho

            ht = et + p

            f[0] = rho*contrav
            for i in range(ndims):
                f[i + 1] = u[i + 1]*contrav + nf[i]*p
            f[nfvars-1] = ht*contrav

            return p, contrav

        return self.be.compile(flux)

    def to_flow_primevars(self):
        gamma, pmin = self._const['gamma'], self._const['pmin']
        ndims, nfvars = self.ndims, self.nfvars

        def to_primevars(u, v):
            rho, et = u[0], u[nfvars-1]

            for i in range(ndims):
                v[i] = u[i + 1] / rho

            p = (gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho)
            if p < pmin:
                p = pmin
                u[nfvars - 1] = p/(gamma-1) + 0.5*dot(u, u, ndims, 1, 1)/rho

            return p

        return self.be.compile(to_primevars)

    def fix_nonPys_container(self):
        gamma, pmin = self._const['gamma'], self._const['pmin']
        ndims, nfvars = self.ndims, self.nfvars

        def fix_nonPhy(u):
            rho, et = u[0], u[nfvars-1]
            if rho < 0:
                u[0] = rho = eps

            p = (gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho)

            if p < pmin:
                u[nfvars - 1] = pmin/(gamma-1) + 0.5*dot(u, u, ndims, 1, 1)/rho

        return self.be.compile(fix_nonPhy)


class EulerElements(BaseAdvecElements, FluidElements):
    def __init__(self, be, cfg, name, eles, vcon):
        super().__init__(be, cfg, name, eles, vcon)
        self.nvars = len(self.primevars)
        self.nfvars = self.nvars

        # Constants
        cfg.get('constants', 'pmin', '1e-15')
        self._const = cfg.items('constants')

    def construct_kernels(self, vertex, xw, nreg):
        super().construct_kernels(vertex, xw, nreg)

        self.timestep = Kernel(self._make_timestep(),
                               self.upts_in, self.dt)

    def _make_timestep(self):
        ndims, nface = self.ndims, self.nface
        vol = self._vol
        smag, svec = self._gen_snorm_fpts()
        gamma, pmin = self._const['gamma'], self._const['pmin']

        def timestep(i_begin, i_end, u, dt, cfl):
            for idx in range(i_begin, i_end):
                rho = u[0, idx]
                et = u[-1, idx]
                rv2 = dot(u[:, idx], u[:, idx], ndims, 1, 1)/rho

                p = max((gamma - 1)*(et - 0.5*rv2), pmin)
                c = np.sqrt(gamma*p/rho)

                # Wave speed * surface area의 합
                sum_lamdf = 0.0
                for jdx in range(nface):
                    lamdf = abs(dot(u[:, idx], svec[jdx, idx], ndims, 1)) + c
                    sum_lamdf += lamdf*smag[jdx, idx]

                dt[idx] = cfl*vol[idx] / sum_lamdf

        return self.be.make_loop(self.neles, timestep)

    def make_wave_speed(self):
        ndims, nfvars = self.ndims, self.nfvars
        gamma, pmin = self._const['gamma'], self._const['pmin']

        def _lambdaf(u, nf, *args):
            rho, et = u[0], u[nfvars-1]

            contra = dot(u, nf, ndims, 1)/rho
            p = max((gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho), pmin)
            c = np.sqrt(gamma*p/rho)

            return abs(contra) + c

        return self.be.compile(_lambdaf)
