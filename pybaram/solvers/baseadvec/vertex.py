# -*- coding: utf-8 -*-
from pybaram.utils.misc import ProxyList
from pybaram.utils.kernels import Kernel, NullKernel
from pybaram.solvers.base import BaseVertex

import numpy as np
import numba as nb


class BaseAdvecVertex(BaseVertex):
    _tag = 2314

    def make_array(self, limiter):
        if limiter == 'none':
            self.vpts = None
        else:
            if not hasattr(self, 'vpts'):
                self.vpts = np.empty((2, self.nvars, self.nvtx))

        return self.vpts

    def construct_kernels(self, elemap):
        order = self.cfg.getint('solver', 'order', 1)
        limiter = self.cfg.get('solver', 'limiter', 'none')

        if order > 1 and limiter != 'none':
            upts_in = [ele.upts_in for ele in elemap.values()]
            self.compute_extv = Kernel(self._make_extv(), self.vpts, *upts_in)

            if self._neivtx:
                self.mpi = True
                self._construct_neighbors(self._neivtx)
            else:
                self.mpi = False
        else:
            self.compute_extv = NullKernel
            self.mpi = False

    def _make_extv(self):
        ivtx = self._ivtx
        t, e, _ = self._idx
        nvtx, nvars = self.nvtx, self.nvars

        @nb.jit(nopython=True, fastmath=True)
        def cal_extv(vext, *upts):
            for i in range(nvtx):
                for idx in range(ivtx[i], ivtx[i+1]):
                    ti, ei = t[idx], e[idx]
                    for jdx in range(nvars):
                        if idx == ivtx[i]:
                            vext[0, jdx, i] = upts[ti][jdx, ei]
                            vext[1, jdx, i] = upts[ti][jdx, ei]
                        else:
                            vext[0, jdx, i] = max(
                                vext[0, jdx, i], upts[ti][jdx, ei])
                            vext[1, jdx, i] = min(
                                vext[1, jdx, i], upts[ti][jdx, ei])

        return cal_extv

    def _construct_neighbors(self, neivtx):
        from mpi4py import MPI

        sbufs, rbufs = [], []
        packs, unpacks = [], []
        sreqs, rreqs = [], []

        nvars = self.nvars
        for p, v in neivtx.items():
            # Make buffer
            n = len(v)
            sbuf = np.empty((2, nvars, n), dtype=np.float)
            rbuf = np.empty((2, nvars, n), dtype=np.float)

            sbufs.append(sbuf)
            rbufs.append(rbuf)

            packs.append(self._make_pack(v))
            unpacks.append(self._make_unpack(v))
            sreqs.append(self._make_send(sbuf, p))
            rreqs.append(self._make_recv(rbuf, p))

        def _communicate(reqs):
            def runall(q):
                q.register(*reqs)
                MPI.Prequest.Startall(reqs)

            return runall

        self.send = _communicate(sreqs)
        self.recv = _communicate(rreqs)

        self.pack = lambda: [pack(self.vpts, buf)
                             for pack, buf in zip(packs, sbufs)]
        self.unpack = lambda: [unpack(self.vpts, buf)
                               for unpack, buf in zip(unpacks, rbufs)]

        self.rbufs = ProxyList(rbufs)

    def _make_pack(self, ivtx):
        n = len(ivtx)
        nvars = self.nvars

        @nb.jit(nopython=True, fastmath=True)
        def pack(vext, buf):
            for idx in range(n):
                iv = ivtx[idx]
                for jdx in range(nvars):
                    buf[0, jdx, idx] = vext[0, jdx, iv]
                    buf[1, jdx, idx] = vext[1, jdx, iv]

        return pack

    def _make_unpack(self, ivtx):
        n = len(ivtx)
        nvars = self.nvars

        @nb.jit(nopython=True, fastmath=True)
        def unpack(vext, buf):
            for idx in range(n):
                iv = ivtx[idx]
                for jdx in range(nvars):
                    vext[0, jdx, iv] = max(vext[0, jdx, iv], buf[0, jdx, idx])
                    vext[1, jdx, iv] = min(vext[1, jdx, iv], buf[1, jdx, idx])

        return unpack

    def _make_send(self, buf, dest):
        from mpi4py import MPI

        mpifn = MPI.COMM_WORLD.Send_init
        return mpifn(buf, dest, self._tag)

    def _make_recv(self, buf, dest):
        from mpi4py import MPI

        mpifn = MPI.COMM_WORLD.Recv_init
        return mpifn(buf, dest, self._tag)
