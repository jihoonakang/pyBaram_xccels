# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvecdiff.system import BaseAdvecDiffSystem
from pybaram.solvers.ranskwsst import RANSKWSSTElements, RANSKWSSTIntInters, RANSKWSSTBCInters, RANSKWSSTMPIInters
from pybaram.backends.types import Queue

import numpy as np
import re


class RANSKWSSTSystem(BaseAdvecDiffSystem):
    name = 'rans-kwsst'
    _elements_cls = RANSKWSSTElements
    _intinters_cls = RANSKWSSTIntInters
    _bcinters_cls = RANSKWSSTBCInters
    _mpiinters_cls = RANSKWSSTMPIInters

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

        # Load bnode
        bnode = self.load_bnode(msh, cfg, rank)

        # Construct kerenls
        self.eles.construct_kernels(vertex, bnode, nreg)
        self.iint.construct_kernels(elemap)
        self.bint.construct_kernels(elemap)

        # Check reconstructed or not
        self._is_recon = (cfg.getint('solver', 'order', 1) > 1)

        if self.mpiint:
            # Construct MPI kernels
            self.mpiint.construct_kernels(elemap)

        # Construct Vertex kernels
        self.vertex.construct_kernels(elemap)

        # Construct queue
        self._queue = Queue()

    def load_bnode(self, msh, cfg, rank):
        is_loaded = []

        if rank == 0:
            bnode = []
            for key in msh:
                m = re.match(r'bcon_([a-z_\d]+)_p([\d]+)$', key)

                if m:
                    if m.group(0) not in is_loaded:
                        bcsect = 'soln-bcs-{}'.format(m.group(1))
                        bctype = cfg.get(bcsect, 'type')

                        if bctype in ['adia-wall', 'isotherm-wall']:
                            bnode.append(msh['bnode_' + m.group(1)][:,:self.ndims])
            
            bnode = np.vstack(bnode)
        else:
            bnode = None

        bnode = self._comm.bcast(bnode, root=0)

        return bnode