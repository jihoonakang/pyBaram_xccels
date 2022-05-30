# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvecdiff.system import BaseAdvecDiffSystem
from pybaram.solvers.ranskwsst import RANSKWSSTElements, RANSKWSSTIntInters, RANSKWSSTBCInters, RANSKWSSTMPIInters
from pybaram.utils.misc import ProxyList, subclass_by_name

import numpy as np
import re


class RANSKWSSTSystem(BaseAdvecDiffSystem):
    name = 'rans-kwsst'
    _elements_cls = RANSKWSSTElements
    _intinters_cls = RANSKWSSTIntInters
    _bcinters_cls = RANSKWSSTBCInters
    _mpiinters_cls = RANSKWSSTMPIInters

    def load_bc_inters(self, msh, be, cfg, rank, elemap):
        bint = ProxyList()
        for key in msh:
            m = re.match(r'bcon_([a-z_\d]+)_p{}$'.format(rank), key)

            if m:
                lhs = msh[m.group(0)].astype('U4,i4,i1,i1').tolist()

                bcsect = 'soln-bcs-{}'.format(m.group(1))
                bctype = cfg.get(bcsect, 'type')

                bcls = subclass_by_name(self._bcinters_cls, bctype)
                bc = bcls(be, cfg, elemap, lhs, m.group(1))

                if bc.is_vis_wall:
                    bc.xw = self._load_bc_nodes(msh, m.group(1), rank, bc.ndims)

                bint.append(bc)

        return bint

    def _load_bc_nodes(self, msh, name, rank, ndims):
        if rank == 0:
            bnode = np.vstack([
                msh[k].reshape(-1, ndims) for k in msh if k.startswith('bface_' + name)
            ])
        else:
            bnode = None
            
        self._comm.bcast(bnode, root=0)

        return bnode

    def compute_bc_wall(self, bint):
        # 벽면 경계 조건 위치
        return np.vstack([bc.xw for bc in bint if bc.is_vis_wall])[None,:]
