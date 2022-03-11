from pybaram.solvers.base.system import BaseSystem
from pybaram.solvers.euler.system import EulerSystem
from pybaram.solvers.euler.elements import FluidElements
from pybaram.utils.misc import subclass_by_name


def get_system(cfg, msh, soln, comm, nreg):
    name = cfg.get('solver', 'system')
    return subclass_by_name(BaseSystem, name)(cfg, msh, soln, comm, nreg)


def get_fluid(name):
    if name in ['euler']:
        return FluidElements()
    else:
        return subclass_by_name(FluidElements, name)()        