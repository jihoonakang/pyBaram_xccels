# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvec.system import BaseAdvecSystem
from pybaram.solvers.baseadvecdiff import BaseAdvecDiffElements, BaseAdvecDiffIntInters, BaseAdvecDiffMPIInters, BaseAdvecDiffBCInters


class BaseAdvecDiffSystem(BaseAdvecSystem):
    name = 'baseadvec'
    _elements_cls = BaseAdvecDiffElements
    _intinters_cls = BaseAdvecDiffIntInters
    _bcinters_cls = BaseAdvecDiffBCInters
    _mpiinters_cls = BaseAdvecDiffMPIInters

    def rhside(self, idx_in=0, idx_out=1, t=0, is_norm=False):
        self.eles.upts_in.idx = idx_in
        self.eles.upts_out.idx = idx_out

        q = self._queue

        # Face에서 값 계산
        self.eles.compute_fpts()

        if self.mpiint:
            self.mpiint.pack()
            self.mpiint.send(q)
            self.mpiint.recv(q)

        # Face에서 차이
        self.iint.compute_delu()
        self.bint.compute_delu()

        if self.mpiint:
            q.sync()
            self.mpiint.compute_delu()

        # 꼭지점에서 최대/최소 계산
        self.vertex.compute_extv()

        if self.vertex.mpi:
            self.vertex.pack()
            self.vertex.send(q)
            self.vertex.recv(q)

        # Element에서 Gradient 계산
        self.eles.compute_grad()

        if self.vertex.mpi:
            q.sync()
            self.vertex.unpack()

        # Face에서 Gradient 계산
        self.iint.compute_grad_at()
        self.bint.compute_grad_at()

        if self.mpiint:
            self.mpiint.pack_grad()
            self.mpiint.send_grad(q)
            self.mpiint.recv_grad(q)

        # 기울기 제한자 계산 및 경계값 계산
        self.eles.compute_mlp_u()

        if self.mpiint:
            q.sync()
            self.mpiint.compute_grad_at()

        self.eles.compute_recon()

        if self._is_recon and self.mpiint:
            self.mpiint.pack()
            self.mpiint.send(q)
            self.mpiint.recv(q)

        # Flux 계산
        self.iint.compute_flux()
        self.bint.compute_flux()

        if self.mpiint:
            q.sync()
            self.mpiint.compute_flux()

        # div(u) 계산 
        self.eles.div_upts(t)

        if is_norm:
            resid = sum(self.eles.compute_resid())
            return resid
        else:
            return 'none'
