# -*- coding: utf-8 -*-
import numpy as np
import re

from collections import OrderedDict
from pybaram.solvers.base import BaseElements, BaseIntInters, BaseBCInters, BaseMPIInters, BaseVertex
from pybaram.utils.misc import ProxyList, subclass_by_name
from pybaram.backends.types import Queue


class BaseSystem:
    name = 'base'
    _elements_cls = BaseElements
    _intinters_cls = BaseIntInters
    _mpiinters_cls = BaseMPIInters
    _bcinters_cls = BaseBCInters
    _vertex_cls = BaseVertex

    def __init__(self, be, cfg, msh, soln, comm, nreg):
        # Save parallel infos
        self._comm = comm
        self.rank = rank = comm.rank

        # Load elements
        self.eles, elemap = self.load_elements(msh, soln, be, cfg, rank)
        self.ndims = next(iter(self.eles)).ndims

        # load interfaces
        self.iint = self.load_int_inters(msh, be, cfg, rank, elemap)

        # load bc
        self.bint = self.load_bc_inters(msh, be, cfg, rank, elemap)

        # load mpiint
        self.mpiint = self.load_mpi_inters(msh, be, cfg, rank, elemap)

        # Load vertex
        self.vertex = vertex = self.load_vertex(msh, be, cfg, rank, elemap)

        # Construct kerenls
        self.eles.construct_kernels(vertex, nreg)
        self.iint.construct_kernels(elemap)
        self.bint.construct_kernels(elemap)

        # Check reconstructed or not
        self._is_recon = (cfg.getint('solver', 'order', 1) > 1)

        if self.mpiint:
            from mpi4py import MPI

            # Construct MPI kernels
            self.mpiint.construct_kernels(elemap)

        # Construct Vertex kernels
        self.vertex.construct_kernels(elemap)

        # Construct queue
        self._queue = Queue()

    def load_elements(self, msh, soln, be, cfg, rank):
        elemap = OrderedDict()
        eles = ProxyList()

        for key in msh:
            m = re.match(r'spt_([a-z]*)_p{}$'.format(rank), key)

            if m:
                etype = m.group(1)
                spt = msh[m.group(0)]

                # Load elements
                ele = self._elements_cls(be, cfg, etype, spt)
                elemap[etype] = ele
                eles.append(ele)

        # Get initial solution
        if soln:
            for k, ele in elemap.items():
                sol = soln['soln_{}_p{}'.format(k, rank)]
                ele.set_ics_from_sol(sol)
        else:
            eles.set_ics_from_cfg()

        return eles, elemap

    def load_int_inters(self, msh, be, cfg, rank, elemap):
        key = 'con_p{0}'.format(rank)
        lhs, rhs = msh[key].astype('U4,i4,i1,i1').tolist()
        iint = self._intinters_cls(be, cfg, elemap, lhs, rhs)

        return iint

    def load_mpi_inters(self, msh, be, cfg, rank, elemap):
        mpiint = ProxyList()

        for key in msh:
            m = re.match(r'con_p{}p(\d+)$'.format(rank), key)

            if m:
                lhs = msh[m.group(0)].astype('U4,i4,i1,i1').tolist()
                mpiint.append(self._mpiinters_cls(
                    be, cfg, elemap, lhs, int(m.group(1))))
        return mpiint

    def load_bc_inters(self, msh, be, cfg, rank, elemap):
        bint = ProxyList()
        for key in msh:
            m = re.match(r'bcon_([a-z_\d]+)_p{}$'.format(rank), key)

            if m:
                lhs = msh[m.group(0)].astype('U4,i4,i1,i1').tolist()

                bcsect = 'soln-bcs-{}'.format(m.group(1))
                bctype = cfg.get(bcsect, 'type')

                bint.append(
                    subclass_by_name(self._bcinters_cls, bctype)
                    (be, cfg, elemap, lhs, m.group(1))
                )

        return bint

    def load_vertex(self, msh, be, cfg, rank, elemap):
        nei_vtx = {}

        for key in msh:
            m = re.match(r'nvtx_p{}p(\d+)$'.format(rank), key)

            if m:
                p = int(m.group(1))
                nei_vtx.update({p: msh[key]})

        vtx = msh['vtx_p{}'.format(rank)].astype('U4,i4,i1,i1').tolist()
        ivtx = msh['ivtx_p{}'.format(rank)]
        vertex = self._vertex_cls(be, cfg, elemap, vtx, ivtx, nei_vtx)

        return vertex
