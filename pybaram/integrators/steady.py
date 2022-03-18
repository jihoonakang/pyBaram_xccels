# -*- coding: utf-8 -*-
from mpi4py import MPI
from pybaram.backends.types import Kernel, MetaKernel
from pybaram.inifile import INIFile
from pybaram.integrators.base import BaseIntegrator
from pybaram.utils.misc import ProxyList
from pybaram.utils.np import eps

import numpy as np
import re


class BaseSteadyIntegrator(BaseIntegrator):
    mode = 'steady'

    def __init__(self, be, cfg, msh, soln, comm):
        # MPI communicator 저장
        self._comm = comm

        # Get Iteration
        self.itermax = cfg.getint('solver-time-integrator', 'max-iter')
        self.tol = cfg.getfloat('solver-time-integrator', 'tolerance')

        # Current iteration
        if soln:
            stats = INIFile()
            stats.fromstr(soln['stats'])
            self.iter = stats.getint('solver-time-integrator', 'iter', 0)
            if self.iter > 0:
                self.resid0 = np.array(stats.getlist(
                    'solver-time-integrator', 'resid0'))
        else:
            self.iter = 0

        self.isconv = False

        super().__init__(be, cfg, msh, soln, comm)

        # Get CFL
        self._cfl = cfg.getfloat('solver-time-integrator', 'cfl', 1.0)

        # Residual var
        ele = next(iter(self.sys.eles))
        self.conservars = conservars = ele.conservars
        rvar = cfg.get('solver-time-integrator', 'res-var', 'rho')
        self._res_idx = [i for i, e in enumerate(conservars) if e == rvar][0]

        # Get total volume
        voli = sum(self.sys.eles.tot_vol)
        self.vol = comm.allreduce(voli, op=MPI.SUM)

        # Construct stages
        self.construct_stages()

    def complete_step(self, resid):
        self.resid = resid

        # Check if residual0 is exist or not
        if not hasattr(self, 'resid0'):
            # Avoid zero resid0
            self.resid0 = [r if r != 0 else eps for r in self.resid]

        self.iter += 1
        self.completed_handler(self)

    def _make_stages(self, out, *args):
        # 계산 Kernel 함수 Python 코드 생성
        eq_str = '+'.join('{}*upts[{}][j, idx]'.format(a, i) for a, i in zip(args[::2], args[1::2]))

        # Substitute dt
        eq_str = re.sub('dt', 'dt[idx]', eq_str)

        f_txt =(
            f"def stage(i_begin, i_end, dt, *upts):\n"
            f"  for idx in range(i_begin, i_end):\n"
            f"      for j in range(nvars):\n"
        )
        f_txt += "          upts[{}][j, idx] = {}".format(out, eq_str)

        kernels = []
        for ele in self.sys.eles:
            # Elements 별로 Stage 컴파일
            gvars = {'nvars' : ele.nvars}
            lvars = {}
            exec(f_txt, gvars, lvars)

            # Loop 구성 및 Kernel 생성
            _stage = self.be.make_loop(ele.neles, lvars['stage'])
            kernels.append(Kernel(_stage, ele.dt, *ele.upts))
        
        return MetaKernel(kernels)

    def run(self):
        while self.iter < self.itermax:
            self.advance_to()

            # Check if tolerance is satisfied
            residual = (self.resid / self.resid0)
            if residual[self._res_idx] < self.tol:
                break

        # Fire off plugins
        self.isconv = True
        self.completed_handler(self)
        self.print_res(residual)

    def _local_dt(self):
        self.sys.timestep(self._cfl, self._curr_idx)

    def advance_to(self):
        self._local_dt()
        self._curr_idx, resid = self.step()
        self.complete_step(resid)

    def rhs(self, idx_in=0, idx_out=1, is_norm=False):
        residi = self.sys.rhside(idx_in, idx_out, is_norm=is_norm)

        # Compute L2 norm residual
        if is_norm:
            resid = self._comm.allreduce(residi, op=MPI.SUM)
            return np.sqrt(resid) / self.vol

    def print_res(self, residual):
        idx = self._res_idx
        res = residual[idx]
        if res < self.tol:
            print("Converged : Residual of {} = {:05g} <= {:05g}".format(
                self.conservars[idx], res, self.tol))
        else:
            print("Not converged : Residual of {} = {:05g} > {:05g}".format(
                self.conservars[idx], res, self.tol))


class EulerExplicit(BaseSteadyIntegrator):
    name = 'eulerexplicit'
    nreg = 2

    def construct_stages(self):
        self._stages = [self._make_stages(0, 1, 0, 'dt', 1)]

    def step(self):
        stages = self._stages

        resid = self.rhs(0, 1, is_norm=True)
        stages[0]()

        return 0, resid


class TVDRK3(BaseSteadyIntegrator):
    name = 'tvd-rk3'
    nreg = 3

    def construct_stages(self):
        self._stages = [
            self._make_stages(2, 1, 0, 'dt', 1),
            self._make_stages(2, 3/4, 0, 1/4, 2, 'dt/4', 1),
            self._make_stages(0, 1/3, 0, 2/3, 2, '2*dt/3', 1),
        ]

    def step(self):
        stages = self._stages

        self.rhs()
        stages[0]()

        self.rhs(2, 1)
        stages[1]()

        resid = self.rhs(2, 1, is_norm=True)
        stages[2]()

        return 0, resid


class FiveStageRK(BaseSteadyIntegrator):
    """
    Jameson Multistage scheme
    ref : Blazek book 6.1.1 (Table 6.1)
    """
    name = 'rk5'
    nreg = 3

    def construct_stages(self):
        self._stages = [
            self._make_stages(2, 1, 0, '0.0533*dt', 1),
            self._make_stages(2, 1, 0, '0.1263*dt', 1),
            self._make_stages(2, 1, 0, '0.2375*dt', 1),
            self._make_stages(2, 1, 0, '0.4414*dt', 1),
            self._make_stages(0, 1, 0, 'dt', 1)
        ]

    def step(self):
        stages = self._stages

        self.rhs()
        stages[0]
        
        self.rhs(2, 1)
        stages[1]()

        self.rhs(2, 1)
        stages[2]()

        self.rhs(2, 1)
        stages[3]()

        resid = self.rhs(2, 1, is_norm=True)
        stages[4]()

        return 0, resid
 

class LUSGS(BaseSteadyIntegrator):
    name = 'lu-sgs'
    nreg = 3

    def construct_stages(self):
        from pybaram.integrators.lusgs import make_lusgs

        be = self.be

        # Assign LU-SGS
        for ele in self.sys.eles:
            # Get Coloring result
            ncolor, icolor = ele.coloring()
             
            # diagonal and lambda array
            diag = np.empty(ele.neles)
            lambdaf = np.empty((ele.nface, ele.neles))

            # Sweep Marker (lowered : 1, Uppered : -1)
            sweep_marked = np.zeros(ele.neles, dtype=np.int)

            _flux = ele.flux_container()
            _lambdaf = ele.make_wave_speed()

            # Make LU-SGS
            _pre_lusgs, _lsweep, _usweep, _update = make_lusgs(
                be, ele, icolor, _flux, _lambdaf, factor=1.0
            )

            pre_lusgs = Kernel(
                be.make_loop(ele.neles, _pre_lusgs), 
                ele.upts[0], ele.dt, diag, lambdaf
            )
            
            lsweeps = [
                Kernel(be.make_loop(n0=n0, ne=ne, func=_lsweep),
                sweep_marked, ele.upts[0], ele.upts[1], ele.upts[2], diag, lambdaf
                ) 
                for n0, ne in zip(ncolor[:-1], ncolor[1:])
            ]

            usweeps = [
                Kernel(be.make_loop(n0=n0, ne=ne, func=_usweep),
                sweep_marked, ele.upts[0], ele.upts[1], ele.upts[2], diag, lambdaf
                ) 
                for n0, ne in zip(ncolor[::-1][1:], ncolor[::-1][:-1])
            ]

            ele.lusgs = MetaKernel((pre_lusgs, *lsweeps, *usweeps))
            ele.update = Kernel(be.make_loop(ele.neles, _update),
                ele.upts[0], ele.upts[1]
            )
    
    def step(self):
        resid = self.rhs(0, 1, is_norm=True)
        self.sys.eles.lusgs()
        self.sys.eles.update()

        return 0, resid

                




             

