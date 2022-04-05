# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvecdiff.system import BaseAdvecDiffSystem
from pybaram.solvers.ranssa import RANSSAElements, RANSSAIntInters, RANSSAMPIInters, RANSSABCInters

import numpy as np


class RANSSASystem(BaseAdvecDiffSystem):
    name = 'rans-sa'
    _elements_cls = RANSSAElements
    _intinters_cls = RANSSAIntInters
    _bcinters_cls = RANSSABCInters
    _mpiinters_cls = RANSSAMPIInters

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
