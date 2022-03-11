# -*- coding: utf-8 -*-
from pybaram.solvers.base.system import BaseSystem
from pybaram.solvers.baseadvec import BaseAdvecElements, BaseAdvecIntInters, BaseAdvecMPIInters, BaseAdvecBCInters, BaseAdvecVertex


class BaseAdvecSystem(BaseSystem):
    name = 'baseadvec'
    _elements_cls = BaseAdvecElements
    _intinters_cls = BaseAdvecIntInters
    _bcinters_cls = BaseAdvecBCInters
    _mpiinters_cls = BaseAdvecMPIInters
    _vertex_cls = BaseAdvecVertex

    def rhside(self, idx_in=0, idx_out=1, t=0, is_norm=False):
        self.eles.upts_in.idx = idx_in
        self.eles.upts_out.idx = idx_out

        q = self._queue

        self.eles.compute_fpts()

        if self.mpiint:
            self.mpiint.pack()
            self.mpiint.send(q)
            self.mpiint.recv(q)

        self.iint.compute_delu()
        self.bint.compute_delu()

        if self.mpiint:
            q.sync()
            self.mpiint.compute_delu()

        self.vertex.compute_extv()

        if self.vertex.mpi:
            self.vertex.pack()
            self.vertex.send(q)
            self.vertex.recv(q)

        self.eles.compute_grad()

        if self.vertex.mpi:
            q.sync()
            self.vertex.unpack()

        self.eles.compute_mlp_u()
        self.eles.compute_recon()

        if self._is_recon and self.mpiint:
            self.mpiint.pack()
            self.mpiint.send(q)
            self.mpiint.recv(q)

        self.iint.compute_flux()
        self.bint.compute_flux()

        if self.mpiint:
            q.sync()
            self.mpiint.compute_flux()

        self.eles.div_upts(t)

        if is_norm:
            resid = sum(self.eles.compute_resid())
            return resid
        else:
            return 'none'

    def timestep(self, cfl, idx_in=0):
        self.eles.upts_in.idx = idx_in
        self.eles.timestep(cfl)

    def post(self, idx_in=0):
        self.eles.upts_in.idx = idx_in
        self.eles.post()
