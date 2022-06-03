# -*- coding: utf-8 -*-
from pybaram.solvers.rans.system import RANSSystem
from pybaram.solvers.ranssa import RANSSAElements, RANSSAIntInters, RANSSAMPIInters, RANSSABCInters


class RANSSASystem(RANSSystem):
    name = 'rans-sa'
    _elements_cls = RANSSAElements
    _intinters_cls = RANSSAIntInters
    _bcinters_cls = RANSSABCInters
    _mpiinters_cls = RANSSAMPIInters
