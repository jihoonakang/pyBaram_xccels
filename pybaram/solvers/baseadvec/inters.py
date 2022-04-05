# -*- coding: utf-8 -*-
from pybaram.solvers.base import BaseIntInters, BaseBCInters, BaseMPIInters
from pybaram.backends.types import Kernel, NullKernel
from pybaram.utils.np import npeval

import numpy as np
import re


class BaseAdvecIntInters(BaseIntInters):
    def construct_kernels(self, elemap):
        # View of elemenet array
        fpts = [cell.fpts for cell in elemap.values()]

        self.compute_flux = Kernel(self._make_flux(), *fpts)

        if self.order > 1:
            self.compute_delu = Kernel(self._make_delu(), *fpts)
        else:
            self.compute_delu = NullKernel

    def _make_delu(self):
        nvars = self.nvars
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx

        def compute_delu(i_begin, i_end, *uf):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]

                for jdx in range(nvars):
                    ul = uf[lti][lfi, jdx, lei]
                    ur = uf[rti][rfi, jdx, rei]
                    du = ur - ul
                    uf[lti][lfi, jdx, lei] = du
                    uf[rti][rfi, jdx, rei] = -du

        return self.be.make_loop(self.nfpts, compute_delu)


class BaseAdvecMPIInters(BaseMPIInters):
    _tag = 1234

    def construct_kernels(self, elemap):
        lhs = np.empty((self.nvars, self.nfpts))
        rhs = np.empty((self.nvars, self.nfpts))

        # View of elemenet array
        fpts = [cell.fpts for cell in elemap.values()]

        self.compute_flux = Kernel(self._make_flux(), rhs, *fpts)

        if self.order > 1:
            self.compute_delu = Kernel(self._make_delu(), rhs, *fpts)
        else:
            self.compute_delu = NullKernel

        self.pack = Kernel(self._make_pack(), lhs, *fpts)
        self.send, self.sreq = self._make_send(lhs)
        self.recv, self.rreq = self._make_recv(rhs)

    def _make_delu(self):
        nvars = self.nvars
        lt, le, lf = self._lidx

        def compute_delu(i_begin, i_end, rhs, *uf):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                for jdx in range(nvars):
                    ul = uf[lti][lfi, jdx, lei]
                    ur = rhs[jdx, idx]
                    du = ur - ul
                    uf[lti][lfi, jdx, lei] = du

        return self.be.make_loop(self.nfpts, compute_delu)

    def _make_pack(self):
        nvars = self.nvars
        lt, le, lf = self._lidx

        def pack(i_begin, i_end, lhs, *uf):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                for jdx in range(nvars):
                    lhs[jdx, idx] = uf[lti][lfi, jdx, lei]

        return self.be.make_loop(self.nfpts, pack)

    def _sendrecv(self, mpifn, arr):
        req = mpifn(arr, self._dest, self._tag)

        def start(q):
            q.register(req)
            return req.Start()

        return start, req

    def _make_send(self, arr):
        from mpi4py import MPI

        mpifn = MPI.COMM_WORLD.Send_init
        start, req = self._sendrecv(mpifn, arr)

        return start, req

    def _make_recv(self, arr):
        from mpi4py import MPI

        mpifn = MPI.COMM_WORLD.Recv_init
        start, req = self._sendrecv(mpifn, arr)

        return start, req


class BaseAdvecBCInters(BaseBCInters):
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

        self.bc = self._get_bc(self.be, bcf, bcc)

    def construct_kernels(self, elemap):
        self.construct_bc()

        # View of elemenet array
        fpts = [cell.fpts for cell in elemap.values()]

        self.compute_flux = Kernel(self._make_flux(), *fpts)

        if self.order > 1:
            self.compute_delu = Kernel(self._make_delu(), *fpts)
        else:
            self.compute_delu = NullKernel

    def _make_delu(self):
        nvars, ndims = self.nvars, self.ndims
        lt, le, lf = self._lidx
        nf = self._vec_snorm

        bc = self.bc

        def compute_delu(i_begin, i_end, *uf):
            ur = np.empty(nvars)

            for idx in range(i_begin, i_end):
                nfi = nf[:, idx]

                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                ul = uf[lti][lfi, :, lei]
                bc(ul, ur, nfi)

                for jdx in range(nvars):
                    du = ur[jdx] - ul[jdx]
                    uf[lti][lfi, jdx, lei] = du

        return self.be.make_loop(self.nfpts, compute_delu)
