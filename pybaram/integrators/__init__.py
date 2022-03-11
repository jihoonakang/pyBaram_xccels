from pybaram.integrators.unsteady import BaseUnsteadyIntegrator
from pybaram.utils.misc import subclass_by_name


def get_integrator(cfg, msh, soln, comm):
    mode = cfg.get('solver-time-integrator', 'mode', 'unsteady')
    stepper = cfg.get('solver-time-integrator', 'stepper', 'tvd-rk3')

    intg = subclass_by_name(BaseUnsteadyIntegrator, stepper)

    return intg(cfg, msh, soln, comm)