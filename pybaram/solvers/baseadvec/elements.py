# -*- coding: utf-8 -*-
import numpy as np
import re

from pybaram.solvers.base import BaseElements
from pybaram.backends.types import ArrayBank, Kernel, NullKernel
from pybaram.utils.np import eps


class BaseAdvecElements(BaseElements):
    def construct_kernels(self, vertex, xw, nreg):
        self.vertex = vertex

        self.coloring()    

        # Upts : Solution vector
        self.upts = upts = [self._ics.copy() for i in range(nreg)]
        del(self._ics)

        # Solution vector bank and assign upts index
        self.upts_in = upts_in = ArrayBank(upts, 0)
        self.upts_out = upts_out = ArrayBank(upts, 1)

        # Array 선언
        self.fpts = fpts = np.empty((self.nface, self.nvars, self.neles))
        self.dt = np.empty(self.neles)
        self.dsrc = np.zeros((self.nvars, self.neles))

        if self.order > 1:
            self.grad = grad = np.empty((self.ndims, self.nvars, self.neles))
            lim = np.ones((self.nvars, self.neles))
            limiter = self.cfg.get('solver', 'limiter', 'none')

            # Prepare vertex array
            vpts = vertex.make_array(limiter)

        # Build kernels
        self.compute_fpts = Kernel(self._make_compute_fpts(), upts_in, fpts)
        self.div_upts = Kernel(self._make_div_upts(), upts_out, fpts)
        self.compute_resid = Kernel(self._make_compute_resid(), self.upts_out)

        if self.order > 1:
            self.compute_grad = Kernel(self._make_grad(), fpts, grad)
            self.compute_recon = Kernel(
                self._make_recon(), upts_in, grad, lim, fpts)

            if limiter != 'none':
                self.compute_mlp_u = Kernel(
                    self._make_mlp_u(limiter), upts_in, grad, vpts, lim)
            else:
                self.compute_mlp_u = NullKernel
        else:
            self.compute_grad = NullKernel
            self.compute_recon = NullKernel
            self.compute_mlp_u = NullKernel

        self.post = Kernel(self._make_post(), upts_in)

    def _make_compute_resid(self):
        import numba as nb

        vol = self._vol
        neles, nvars = self.neles, self.nvars

        def run(upts):
            resid = np.empty(nvars)
            for j in range(nvars):
                s = 0
                for i in nb.prange(neles):
                    s += upts[j,i]**2*vol[i]
                resid[j] = s

            return resid

        return self.be.compile(run, outer=True)

    def _make_compute_fpts(self):
        nvars, nface = self.nvars, self.nface

        def _compute_fpts(i_begin, i_end, upts, fpts):
            for idx in range(i_begin, i_end):
                for j in range(nvars):
                    tmp = upts[j, idx]
                    for k in range(nface):
                        fpts[k, j, idx] = tmp
        
        return self.be.make_loop(self.neles, _compute_fpts)

    def _make_div_upts(self):
        # Global variables
        gvars = {"np": np, "rcp_vol": self.rcp_vol}

        # Parse Source term
        subs = {x: 'xc[{0}, idx]'.format(i)
                for i, x in enumerate('xyz'[:self.ndims])}
        subs.update(self._const)
        subs.update({'sin': 'np.sin', 'cos': 'np.cos',
                     'exp': 'np.exp', 'tanh': 'np.tanh'})

        src = [self.cfg.getexpr('solver-source-terms', k, subs, default=0.0)
               for k in self.conservars]

        if any([re.search(r'xc\[.*?\]', s) for s in src]):
            gvars.update({"xc": self.xc.T})

        # Construct function text
        f_txt = (
            f"def _div_upts(i_begin, i_end, rhs, fpts, t=0):\n"
            f"    for idx in range(i_begin, i_end): \n"
            f"        rcp_voli = rcp_vol[idx]\n"
        )
        for j, s in enumerate(src):
            subtxt = "+".join("fpts[{},{},idx]".format(i, j)
                              for i in range(self.nface))
            f_txt += "        rhs[{}, idx] = -rcp_voli*({}) + {}\n".format(
                j, subtxt, s)

        # Compile function
        lvars = {}
        exec(f_txt, gvars, lvars)

        return self.be.make_loop(self.neles, lvars["_div_upts"])

    def _make_grad(self):
        nface, ndims, nvars = self.nface, self.ndims, self.nvars
        op = self._prelsq

        def _cal_grad(i_begin, i_end, fpts, grad):
            for i in range(i_begin, i_end):
                for l in range(nvars):
                    for k in range(ndims):
                        tmp = 0
                        for j in range(nface):
                            tmp += op[k, j, i]*fpts[j, l, i]
                        grad[k, l, i] = tmp

        return self.be.make_loop(self.neles, _cal_grad)       

    def _make_recon(self):
        nface, ndims, nvars = self.nface, self.ndims, self.nvars
        op = self.dxf

        def _cal_recon(i_begin, i_end, upts, grad, lim, fpts):
            for i in range(i_begin, i_end):
                for l in range(nvars):
                    for k in range(nface):
                        tmp = 0
                        for j in range(ndims):
                            tmp += op[k, j, i]*grad[j, l, i]
                        fpts[k, l, i] = upts[l, i] + lim[l, i]*tmp

        return self.be.make_loop(self.neles, _cal_recon)

    def _make_mlp_u(self, limiter):
        nvtx, ndims, nvars = self.nvtx, self.ndims, self.nvars

        dx = self.dxv
        cons = self._vcon.T

        def u1(dup, dum, ee2):
            return min(1.0, dup/dum)

        def u2(dup, dum, ee2):
            dup2 = dup**2
            dum2 = dum**2
            dupm = dup*dum
            return ((dup2 + ee2)*dum + 2*dum2*dup)/(dup2 + 2*dum2 + dupm + ee2)/dum

        # x_i^1.5
        le32 = self.le**1.5

        if limiter == 'mlp-u2':
            is_u2 = True
            u2k = self.cfg.getfloat('solver', 'u2k', 5.0)
            limf = self.be.compile(u2)
        else:
            is_u2 = False
            u2k = 0.0
            limf = self.be.compile(u1)

        def _cal_mlp_u(i_begin, i_end, upts, grad, vext, lim):
            for i in range(i_begin, i_end):
                for j in range(nvtx):
                    vi = cons[j, i]
                    for k in range(nvars):
                        duv = 0

                        if is_u2:
                            dvv = vext[0, k, vi] - vext[1, k, vi]
                            ee = dvv / le32[i] / u2k
                            ee2 = u2k*dvv**2/(ee + 1.0)
                        else:
                            ee2 = 0.0

                        for l in range(ndims):
                            duv += dx[j, l, i]*grad[l, k, i]

                        if duv > eps:
                            limj = limf(
                                (vext[0, k, vi] - upts[k, i]), duv, ee2)
                        elif duv < -eps:
                            limj = limf(
                                (vext[1, k, vi] - upts[k, i]), duv, ee2)
                        else:
                            limj = 1.0

                        if j == 0:
                            lim[k, i] = limj
                        else:
                            lim[k, i] = min(lim[k, i], limj)

        return self.be.make_loop(self.neles, _cal_mlp_u)

    def _make_post(self):
        neles = self.neles

        _fix_nonPys = self.fix_nonPys_container()

        def post(i_begin, i_end, upts):
            # Update
            for idx in range(i_begin, i_end):
                _fix_nonPys(upts[:, idx])

        return self.be.make_loop(neles, post)
