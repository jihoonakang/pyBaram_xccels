# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvecdiff.system import BaseAdvecDiffSystem
from pybaram.solvers.navierstokes import NavierStokesElements, NavierStokesIntInters, NavierStokesMPIInters, NavierStokesBCInters


class NavierStokeSystem(BaseAdvecDiffSystem):
    name = 'navier-stokes'
    _elements_cls = NavierStokesElements
    _intinters_cls = NavierStokesIntInters
    _bcinters_cls = NavierStokesBCInters
    _mpiinters_cls = NavierStokesMPIInters
