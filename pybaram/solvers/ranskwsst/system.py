# -*- coding: utf-8 -*-
from pybaram.solvers.rans.system import RANSSystem
from pybaram.solvers.ranskwsst import RANSKWSSTElements, RANSKWSSTIntInters, RANSKWSSTBCInters, RANSKWSSTMPIInters


class RANSKWSSTSystem(RANSSystem):
    name = 'rans-kwsst'
    _elements_cls = RANSKWSSTElements
    _intinters_cls = RANSKWSSTIntInters
    _bcinters_cls = RANSKWSSTBCInters
    _mpiinters_cls = RANSKWSSTMPIInters