# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvecdiff.system import BaseAdvecDiffSystem
from pybaram.solvers.ranskwsst import RANSKWSSTElements, RANSKWSSTIntInters, RANSKWSSTBCInters, RANSKWSSTMPIInters

import numpy as np


class RANSKWSSTSystem(BaseAdvecDiffSystem):
    name = 'rans-kwsst'
    _elements_cls = RANSKWSSTElements
    _intinters_cls = RANSKWSSTIntInters
    _bcinters_cls = RANSKWSSTBCInters
    _mpiinters_cls = RANSKWSSTMPIInters

    def compute_bc_wall(self, bint):
        # 벽면 경계 조건 위치
        xw_p = [bc.xf for bc in bint if bc.name in [
            'adia-wall', 'isotherm-wall']]

        if len(xw_p) > 0:
            xw_p = np.hstack(xw_p)
        else:
            xw_p = np.array(xw_p)

        xw = np.hstack(
            [x for x in self._comm.allgather(xw_p) if len(x) > 0]
        )

        return xw
