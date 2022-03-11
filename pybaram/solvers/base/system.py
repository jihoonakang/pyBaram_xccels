# -*- coding: utf-8 -*-
import numpy as np
import re

from collections import OrderedDict
from pybaram.solvers.base import BaseElements, BaseIntInters, BaseBCInters, BaseMPIInters, BaseVertex
from pybaram.utils.misc import ProxyList, subclass_by_name
from pybaram.utils.kernels import Queue


def get_spts(nodepts, etype, cell):
    _etype_ndim = {'tri': 2, 'quad': 2,
                   'tet': 3, 'hex': 3, 'pri': 3, 'pyr': 3}

    ndim = _etype_ndim[etype]

    arr = np.array([[nodepts[i] for i in nn] for nn in cell])
    arr = arr.swapaxes(0, 1)
    return arr[..., :ndim]


class BaseSystem:
    name = 'base'
    _elements_cls = BaseElements
    _intinters_cls = BaseIntInters
    _mpiinters_cls = BaseMPIInters
    _bcinters_cls = BaseBCInters
    _vertex_cls = BaseVertex

    def __init__(self, cfg, msh, soln, comm, nreg):
        self._comm = comm
        self.rank = rank = comm.rank

        # Load elements
        self.eles, elemap = self.load_elements(msh, soln, cfg, rank)
        self.ndims = next(iter(self.eles)).ndims

        # load interfaces
        self.iint = self.load_int_inters(msh, cfg, rank, elemap)

        # load bc
        self.bint = bint = self.load_bc_inters(msh, cfg, rank, elemap)

        # load mpiint
        self.mpiint = self.load_mpi_inters(msh, cfg, rank, elemap)

        # Load vertex
        self.vertex = vertex = self.load_vertex(msh, cfg, rank, elemap)

        # Compute wall boundary
        if hasattr(self, 'compute_bc_wall'):
            xw = self.compute_bc_wall(bint)
        else:
            xw = None

        # Construct kerenls
        self.eles.construct_kernels(vertex, xw, nreg)
        self.iint.construct_kernels(elemap)
        self.bint.construct_kernels(elemap)

        # Check reconstructed or not
        order = cfg.getint('solver', 'order', 1)
        if order > 1:
            self._is_recon = True
        else:
            self._is_recon = False

        if self.mpiint:
            from mpi4py import MPI

            # Construct MPI kernels
            self.mpiint.construct_kernels(elemap)

        self.vertex.construct_kernels(elemap)

        # Construct queue
        self._queue = Queue()

    def load_elements(self, msh, soln, cfg, rank):
        elemap = OrderedDict()
        eles = ProxyList()

        for key in msh:
            m = re.match(r'elm_([a-z]*)_p{}$'.format(rank), key)

            if m:
                etype = m.group(1)
                node = msh['node_p{}'.format(rank)]
                nmap = msh['nmap_p{}'.format(rank)]
                nodepts = dict(zip(nmap, node))
                cons = msh[m.group(0)]
                spts = get_spts(nodepts, etype, cons)

                # Local vertex connecivity
                vmap = dict(zip(nmap, np.arange(len(nmap))))
                vcon = np.array([[vmap[i] for i in e] for e in cons])

                ele = self._elements_cls(cfg, etype, spts, vcon)
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

    def load_int_inters(self, msh, cfg, rank, elemap):
        key = 'con_p{0}'.format(rank)
        lhs, rhs = msh[key].astype('U4,i4,i1,i1').tolist()
        iint = self._intinters_cls(cfg, elemap, lhs, rhs)

        return iint

    def load_mpi_inters(self, msh, cfg, rank, elemap):
        mpiint = ProxyList()

        for key in msh:
            m = re.match(r'con_p{}p(\d+)$'.format(rank), key)

            if m:
                lhs = msh[m.group(0)].astype('U4,i4,i1,i1').tolist()
                mpiint.append(self._mpiinters_cls(
                    cfg, elemap, lhs, int(m.group(1))))
        return mpiint

    def load_bc_inters(self, msh, cfg, rank, elemap):
        bint = ProxyList()
        for key in msh:
            m = re.match(r'bcon_([a-z_\d]+)_p{}$'.format(rank), key)

            if m:
                lhs = msh[m.group(0)].astype('U4,i4,i1,i1').tolist()

                bcsect = 'soln-bcs-{}'.format(m.group(1))
                bctype = cfg.get(bcsect, 'type')

                bint.append(
                    subclass_by_name(self._bcinters_cls, bctype)
                    (cfg, elemap, lhs, m.group(1))
                )

        return bint

    def load_vertex(self, msh, cfg, rank, elemap):
        nei_vtx = {}

        for key in msh:
            m = re.match(r'nvtx_p{}p(\d+)$'.format(rank), key)

            if m:
                p = int(m.group(1))
                nei_vtx.update({p: msh[key]})

        vtx = msh['vtx_p{}'.format(rank)].astype('U4,i4,i1,i1').tolist()
        ivtx = msh['ivtx_p{}'.format(rank)]
        vertex = self._vertex_cls(cfg, elemap, vtx, ivtx, nei_vtx)

        return vertex
