# -*- coding: utf-8 -*-
from pybaram.solvers.euler.elements import FluidElements
from pybaram.solvers.baseadvecdiff import BaseAdvecDiffElements
from pybaram.backends.types import Kernel
from pybaram.utils.np import npeval
from pybaram.utils.nb import dot

import functools as fc
import numpy as np


class ViscousFluidElements(FluidElements):
    name = 'navier-stokes'
    
    @property
    def auxvars(self):
        return ['mu']

    @fc.lru_cache()
    def mu_container(self):
        viscosity = self.cfg.get('solver', 'viscosity', 'constant')

        if viscosity == 'constant':
            mu = self._const['mu']

            def compute_mu(*args):
                # Constant viscosity
                return mu
            
        elif viscosity == 'sutherland':
            # Constants and dimensions
            gamma, pmin = self._const['gamma'], self._const['pmin']
            ndims, nfvars = self.ndims, self.nfvars

            # Minimum value of enthalpy
            Hmin = gamma/(gamma-1)*pmin               

            # All default values are given in MKS units.
            sect = 'solver-viscosity-sutherland'
            Ts = npeval(self.cfg.getexpr(sect, 'Ts', self._const, 110.4))
            Tref = npeval(self.cfg.getexpr(sect, 'Tref', self._const, 288.15))

            # Get reference viscosity
            muref_expr = self.cfg.getexpr(sect, 'muref', self._const)
            if muref_expr == 'none':
                # If mu_ref is not given, calculate from Sutherland Law
                c1 = npeval(self.cfg.getexpr(sect, 'c1', self._const, 1.458e-6))
                muref = c1 * Tref**1.5 / (Tref + Ts)
            else:
                try:
                    muref = npeval(muref_expr)
                except:
                    raise ValueError("Invalid reference viscosty: {}".fromat(muref_expr))
                
            # Free-stream enthalpy
            CpTf = self.cfg.getexpr(sect, 'cptf', self._const,'none')
            if CpTf == 'none':
                KeyError("Free-stream enthalpy is not given (CpTf)")
            else:
                CpTf = npeval(CpTf)

            # Non-dimensionalized Ts
            TsTref = Ts/Tref

            def compute_mu(u):
                rho, et = u[0], u[nfvars-1]
                inv_rho = 1/rho

                # Specific Enthalpy
                CpT = max(
                    gamma*inv_rho*(et - 0.5*inv_rho*dot(u, u, ndims, 1, 1)), 
                    Hmin*inv_rho
                )
                
                # Temperature ratio
                Tratio = CpT / CpTf

                # Sutherland Law (non-dimensionalized form)
                return muref*Tratio**1.5*(1 + TsTref) / (Tratio + TsTref)
                
        else:
            raise ValueError("Unspported viscosity")
        

        return self.be.compile(compute_mu)


class NavierStokesElements(BaseAdvecDiffElements, ViscousFluidElements):
    def __init__(self, be, cfg, name, eles):
        super().__init__(be, cfg, name, eles)
        self.nvars = len(self.primevars)
        self.nfvars = self.nvars

        # Get constants
        cfg.get('constants', 'pmin', '1e-15')
        self._const = cfg.items('constants')

    def construct_kernels(self, vertex, nreg, impl_op):
        # Call paraent method
        super().construct_kernels(vertex, nreg)

        # Aux array
        nauxvars = len(self.auxvars)
        self.aux = aux = np.empty((nauxvars, self.neles))
        
        # Viscosity
        self.mu = aux[0]

        if impl_op == 'spectral-radius':
            # Spectral radius
            self.fspr = np.empty((self.nface, self.neles))
        elif impl_op == 'approx-jacobian':
            # Jacobian matrix on face
            self.jmat = np.empty((2, self.nfvars, self.nfvars, \
                                  self.nface, self.neles))

        # Update arguments of post kerenl
        self.post.update_args(self.upts_in, self.mu)

        # Initialize viscosity
        self.post()

        # Kernel to compute timestep
        self.timestep = Kernel(self._make_timestep(),
                               self.upts_in, self.mu, self.dt)

    def _make_timestep(self):
        # Dimensions
        ndims, nface, nfvars = self.ndims, self.nface, self.nfvars

        # Static variables
        vol = self._vol
        smag, svec = self._gen_snorm_fpts()

        # Constants
        gamma, pmin = self._const['gamma'], self._const['pmin']
        pr = self._const['pr']

        def timestep(i_begin, i_end, u, mu, dt, cfl):
            for idx in range(i_begin, i_end):
                rho = u[0, idx]
                et = u[nfvars-1, idx]
                rv2 = dot(u[:, idx], u[:, idx], ndims, 1, 1)/rho

                p = max((gamma - 1)*(et - 0.5*rv2), pmin)
                c = np.sqrt(gamma*p/rho)

                # Sum of Wave speed * surface area
                sum_lamdf = 0.0
                for jdx in range(nface):
                    # Wave speed abs(Vn) + c + max(4/3 \gamma) mu/rho/pr/length
                    lamdf = abs(dot(u[:, idx], svec[jdx, idx], ndims, 1))/rho + c
                    lamdf += (1/rho*max(4/3, gamma)*mu[idx]/pr *
                              smag[jdx, idx]/vol[idx])
                    sum_lamdf += lamdf*smag[jdx, idx]

                # Time step : CFL * vol / sum(lambda_f S_f)
                dt[idx] = cfl*vol[idx] / sum_lamdf

        return self.be.make_loop(self.neles, timestep)

    def make_wave_speed(self):
        # Dimensions and constants
        ndims, nfvars = self.ndims, self.nfvars
        gamma, pmin = self._const['gamma'], self._const['pmin']
        pr = self._const['pr']

        def _lambdaf(u, nf, rcp_dx, mu):
            rho, et = u[0], u[nfvars-1]

            contra = dot(u, nf, ndims, 1)/rho
            p = max((gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho), pmin)
            c = np.sqrt(gamma*p/rho)

            # Wave speed abs(Vn) + c + 1/dx/rho * max(4/3 \gamma) mu/pr
            return abs(contra) + c + rcp_dx/rho*max(4/3, gamma)*mu/pr

        return self.be.compile(_lambdaf)

    def _make_post(self):
        # Get post-process function
        _fix_nonPys = self.fix_nonPys_container()
        _compute_mu = self.mu_container()

        def post(i_begin, i_end, upts, mu):
            # Apply the function over eleemnts
            for idx in range(i_begin, i_end):
                _fix_nonPys(upts[:, idx])
                mu[idx] = _compute_mu(upts[:, idx])

        return self.be.make_loop(self.neles, post)