# -*- coding: utf-8 -*-
from pybaram.solvers.base import BaseIntInters, BaseBCInters, BaseMPIInters
from pybaram.utils.kernels import Kernel, NullKernel
from pybaram.utils.np import npeval

import numpy as np
import numba as nb
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
        nface, nvars = self.nfpts, self.nvars
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx

        @nb.jit(nopython=True, fastmath=True)
        def compute_delu(*uf):
            for idx in range(nface):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]

                for jdx in range(nvars):
                    ul = uf[lti][lfi, jdx, lei]
                    ur = uf[rti][rfi, jdx, rei]
                    du = ur - ul
                    uf[lti][lfi, jdx, lei] = du
                    uf[rti][rfi, jdx, rei] = -du

        return compute_delu


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
        nface, nvars = self.nfpts, self.nvars
        lt, le, lf = self._lidx

        @nb.jit(nopython=True, fastmath=True)
        def compute_delu(rhs, *uf):
            for idx in range(nface):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                for jdx in range(nvars):
                    ul = uf[lti][lfi, jdx, lei]
                    ur = rhs[jdx, idx]
                    du = ur - ul
                    uf[lti][lfi, jdx, lei] = du

        return compute_delu

    def _make_pack(self):
        nvars, nface = self.nvars, self.nfpts
        lt, le, lf = self._lidx

        @nb.jit(nopython=True, fastmath=True)
        def pack(lhs, *uf):
            for idx in range(nface):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                for jdx in range(nvars):
                    lhs[jdx, idx] = uf[lti][lfi, jdx, lei]

        return pack

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
        ndims, nvars = self.ndims, self.nvars

        # BC constant
        if self._reqs:
            bcsect = 'soln-bcs-{}'.format(self.bctype)
            bcc = {k: npeval(self.cfg.getexpr(bcsect, k, self._const))
                   for k in self._reqs}
        else:
            bcc = {}

        bcc.update(self._const)

        self.bc = self._get_bc(bcf, ndims, nvars, **bcc)

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
        nface, nvars, ndims = self.nfpts, self.nvars, self.ndims
        lt, le, lf = self._lidx
        nf = self._vec_snorm

        bc = self.bc

        @nb.jit(nopython=True, fastmath=True)
        def compute_delu(*uf):
            ul, ur = np.empty(nvars), np.empty(nvars)
            nfi = np.empty(ndims)

            for idx in range(nface):
                for jdx in range(ndims):
                    nfi[jdx] = nf[jdx, idx]

                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                for jdx in range(nvars):
                    ul[jdx] = uf[lti][lfi, jdx, lei]

                bc(ul, ur, nfi)

                for jdx in range(nvars):
                    du = ur[jdx] - ul[jdx]
                    uf[lti][lfi, jdx, lei] = du

        return compute_delu
