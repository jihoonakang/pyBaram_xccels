from mpi4py import MPI
from pybaram.inifile import INIFile
from pybaram.integrators.base import BaseIntegrator

import numpy as np


class BaseUnsteadyIntegrator(BaseIntegrator):
    mode = 'unsteady'

    def __init__(self, cfg, msh, soln, comm):
        self._comm = comm
        self.tlist = eval(cfg.get('solver-time-integrator', 'time'))

        if soln:
            # Restart from solution
            stats = INIFile()
            stats.fromstr(soln['stats'])
            self.tcurr = stats.getfloat('solver-time-integrator', 'tcurr')
            self.iter = stats.getint('solver-time-integrator', 'iter', 0)
        else:
            # Initialize tcurr and iter
            self.tcurr = self.tlist[0]
            self.iter = 0

        super().__init__(cfg, msh, soln, comm)

        self.construct_stages()

        # Configure time step method
        controller = cfg.get('solver-time-integrator', 'controller', 'cfl')
        if controller == 'cfl':
            self.cfl = cfg.getfloat('solver-time-integrator', 'cfl')
            self._timestep = self._dt_cfl
        else:
            dt = cfg.getfloat('solver-time-integrator', 'dt')
            self._timestep = lambda ttag: min(dt, ttag - self.tcurr)

    def add_tlist(self, dt):
        tlist = self.tlist
        tmp = np.arange(tlist[0], tlist[-1], dt)
        self.tlist = np.sort(np.unique(np.concatenate([tlist, tmp])))

    @staticmethod
    def _make_stages(out, *args):
        eq_str = '+'.join('{}*ele.upts[{}]'.format(a, idx)
                          for a, idx in zip(args[::2], args[1::2]))

        def run(ele, dt):
            ele.upts[out] = eval(eq_str)

        return run

    def run(self):
        for t in self.tlist:
            self.advance_to(t)

    def _dt_cfl(self, ttag):
        self.sys.timestep(self.cfl, self._curr_idx)
        dt = min(self.sys.eles.dt.min())

        dtmin = self._comm.allreduce(dt, op=MPI.MIN)

        return min(ttag - self.tcurr, dtmin)

    def advance_to(self, ttag):
        while self.tcurr < ttag:
            self.dt = dt = self._timestep(ttag)
            self._curr_idx = self.step(dt, self.tcurr)
            self.tcurr += dt
            self.iter += 1
            self.completed_handler(self)


class EulerExplicit(BaseUnsteadyIntegrator):
    name = 'eulerexplicit'
    nreg = 2

    def construct_stages(self):
        self._stages = stages = []
        stages.append(self._make_stages(0, 1, 0, 'dt', 1))

    def step(self, dt):
        sys = self.sys
        stages = self._stages

        sys.rhside()
        sys.eles.apply(stages[0], dt)

        return 0


class TVDRK3(BaseUnsteadyIntegrator):
    name = 'tvd-rk3'
    nreg = 3

    def construct_stages(self):
        self._stages = stages = []
        stages.append(self._make_stages(2, 1, 0, 'dt', 1))
        stages.append(self._make_stages(2, 3/4, 0, 1/4, 2, 'dt/4', 1))
        stages.append(self._make_stages(0, 1/3, 0, 2/3, 2, '2*dt/3', 1))

    def step(self, dt, t):
        sys = self.sys
        stages = self._stages

        sys.rhside(t=t)
        sys.eles.apply(stages[0], dt)

        sys.rhside(2, 1, t=t)
        sys.eles.apply(stages[1], dt)

        sys.rhside(2, 1, t=t)
        sys.eles.apply(stages[2], dt)

        return 0