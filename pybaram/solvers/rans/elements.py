# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvecdiff import BaseAdvecDiffElements
from pybaram.backends.types import Kernel
from pybaram.utils.nb import dot

import numpy as np


class RANSElements(BaseAdvecDiffElements):
    def __init__(self, be, cfg, name, eles, vcon):
        super().__init__(be, cfg, name, eles, vcon)
        self.nvars = len(self.primevars)
        self.nfvars = self.nvars - self.nturbvars

        # Constants
        cfg.get('constants', 'pmin', '1e-15')
        self._const = cfg.items('constants')

    def construct_kernels(self, vertex, xw, nreg):
        # Aux array
        nauxvars = len(self.auxvars)
        self.aux = aux = np.empty((nauxvars, self.neles))

        # 벽면까지 길이 계산
        self.ydist = aux[0]
        self._wall_distance(xw, self.ydist)

        super().construct_kernels(vertex, nreg)

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

    def _wall_distance(self, xw, wdist):
        # _wdist 함수 상수
        nf, ne, nd = self.eles.shape
        nw = xw.shape[0]
        eles = self.eles
        rcp_nf = 1.0 / nf
        xmax = 2*(eles.max() - eles.min())

        def _cal_wdist(i_begin, i_end, wdist):
            # Brute-force searching
            for idx in range(i_begin, i_end):
                wd_ele = 0
                for jdx in range(nf):
                    # for all node points
                    xc = eles[jdx, idx]
                    
                    # Compute minimum wall distance for each node
                    wd_node = xmax
                    for kdx in range(nw):
                        xwi = xw[kdx]                      
                        
                        # 길이 계산
                        dx = 0
                        for i in range(nd):
                            dx += (xwi[i] - xc[i])**2

                        dx = np.sqrt(dx)
                        wd_node = min(dx, wd_node)

                    # Averaging for cell
                    wd_ele += wd_node

                wd_ele *= rcp_nf
                wdist[idx] = wd_ele

        self.be.make_loop(ne, _cal_wdist)(wdist)

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

    def _make_recon(self):
        nface, ndims = self.nface, self.ndims
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
        nvars, nface = self.nvars, self.nface

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