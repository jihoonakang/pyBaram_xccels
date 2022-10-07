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
        # get MPI_COMM_WORLD
        self._comm = comm

        # Get configurations for iterators
        self.itermax = cfg.getint('solver-time-integrator', 'max-iter')
        self.tol = cfg.getfloat('solver-time-integrator', 'tolerance')

        # Set current iteration
        if soln:
            # Get current iteration from previous result
            stats = INIFile()
            stats.fromstr(soln['stats'])
            self.iter = stats.getint('solver-time-integrator', 'iter', 0)
            if self.iter > 0:
                self.resid0 = np.array(stats.getlist(
                    'solver-time-integrator', 'resid0'))
        else:
            # Initialize iteration
            self.iter = 0

        # Indicator if solution is converted or not
        self.isconv = False

        super().__init__(be, cfg, msh, soln, comm)

        # Get CFL number 
        self._cfl0 = cfg.getfloat('solver-time-integrator', 'cfl', 1.0)

        # Get configuration of CFL linear ramp 
        self._cfl_iter0 = cfg.getint('solver-cfl-ramp', 'iter0', self.itermax)
        self._cfl_itermax = cfg.getint('solver-cfl-ramp', 'max-iter', self.itermax)
        self._cflmax = cfg.getfloat('solver-cfl-ramp', 'max-cfl', self._cfl0)
        
        # Caculate increment of cfl for CFL ramp
        if self._cfl_itermax > self._cfl_iter0:
            self._dcfl = (self._cflmax - self._cfl0) / (self._cfl_itermax - self._cfl_iter0)
        else:
            self._dcfl = 0

        # Specify residual variable for monitoring
        ele = next(iter(self.sys.eles))
        self.conservars = conservars = ele.conservars
        rvar = cfg.get('solver-time-integrator', 'res-var', 'rho')
        self._res_idx = [i for i, e in enumerate(conservars) if e == rvar][0]

        # Get total volume
        voli = sum(self.sys.eles.tot_vol)
        self.vol = comm.allreduce(voli, op=MPI.SUM)

        # Construct kernels
        self.construct_stages()

    @property
    def _cfl(self):
        # Return CFL considering CFL ramp
        if self.iter < self._cfl_iter0:
            return self._cfl0
        elif self.iter > self._cfl_itermax:
            return self._cflmax
        else:
            return self._cfl0 + self._dcfl*(self.iter - self._cfl_iter0)

    def run(self):
        # Run integerator until max iteration
        while self.iter < self.itermax:
            # Compute one iteration
            self.advance_to()

            # Check if tolerance is satisfied
            residual = (self.resid / self.resid0)
            if residual[self._res_idx] < self.tol:
                break

        # Fire off plugins
        self.isconv = True
        self.completed_handler(self)
        self.print_res(residual)

    def complete_step(self, resid):
        self.resid = resid

        # Check if reference residual (resid0) is existed or not
        if not hasattr(self, 'resid0'):
            # Avoid zero in resid0
            self.resid0 = [r if r != 0 else eps for r in self.resid]

        self.iter += 1
        self.completed_handler(self)

    def _make_stages(self, out, *args):
        # Generate formulation of each RK stage 
        eq_str = '+'.join('{}*upts[{}][j, idx]'.format(a, i) for a, i in zip(args[::2], args[1::2]))

        # Substitute 'dt' string as dt array
        eq_str = re.sub('dt', 'dt[idx]', eq_str)

        # Generate Python function for each RK stage
        f_txt =(
            f"def stage(i_begin, i_end, dt, *upts):\n"
            f"  for idx in range(i_begin, i_end):\n"
            f"      for j in range(nvars):\n"
        )
        f_txt += "          upts[{}][j, idx] = {}".format(out, eq_str)

        kernels = []
        for ele in self.sys.eles:
            # Initiate Python function of RK stage for each element
            gvars = {'nvars' : ele.nvars}
            lvars = {}
            exec(f_txt, gvars, lvars)

            # Generate JIT kernel by looping RK stage function
            _stage = self.be.make_loop(ele.neles, lvars['stage'])
            kernels.append(Kernel(_stage, ele.dt, *ele.upts))
        
        # Collect RK stage kernels for elements
        return MetaKernel(kernels)

    def _local_dt(self):
        # Compute timestep of each cell using CFL
        self.sys.timestep(self._cfl, self._curr_idx)

    def advance_to(self):
        # Compute dt
        self._local_dt()

        # Compute one RK step
        self._curr_idx, resid = self.step()

        # Post actions after iteration
        self.complete_step(resid)

    def rhs(self, idx_in=0, idx_out=1, is_norm=False):
        # Compute right hand side
        residi = self.sys.rhside(idx_in, idx_out, is_norm=is_norm)

        # Compute L2 norm residual
        if is_norm:
            resid = self._comm.allreduce(residi, op=MPI.SUM)
            return np.sqrt(resid) / self.vol

    def print_res(self, residual):
        # Print residual result
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

        self.sys.post(0)

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

        self.sys.post(0)

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

        self.sys.post(0)

        return 0, resid
 

class LUSGS(BaseSteadyIntegrator):
    name = 'lu-sgs'
    nreg = 3

    def construct_stages(self):
        from pybaram.integrators.lusgs import make_lusgs_common, make_lusgs_update, make_serial_lusgs

        be = self.be

        # LU-SGS for each elements
        for ele in self.sys.eles:      
            # Get reordering result
            mapping, unmapping = ele.reordering()

            # diagonal and lambda array
            diag = np.empty(ele.neles)
            lambdaf = np.empty((ele.nface, ele.neles))

            # Get Python functions of flux and wave speed
            _flux = ele.flux_container()
            _lambdaf = ele.make_wave_speed()
            nv = (0, ele.nfvars)

            # Get viscosity if exists
            mu = ele.mu if hasattr(ele, 'mu') else None
            mut = ele.mut if hasattr(ele, 'mut') else None

            # Compile LU-SGS functions
            _update = make_lusgs_update(ele)
            _pre_lusgs = make_lusgs_common(ele, _lambdaf, factor=1.0)
            _lsweep, _usweep = make_serial_lusgs(
                be, ele, nv, mapping, unmapping, _flux
            )

            # Initiate LU-SGS kernel objects
            pre_lusgs = Kernel(
                be.make_loop(ele.neles, _pre_lusgs), 
                ele.upts[0], ele.dt, diag, lambdaf, mu, mut
            )            
            lsweeps = Kernel(be.make_loop(ele.neles, func=_lsweep),
                ele.upts[0], ele.upts[1], ele.upts[2], diag, ele.dsrc, lambdaf) 

            usweeps = Kernel(be.make_loop(ele.neles, func=_usweep),
                ele.upts[0], ele.upts[1], ele.upts[2], diag, ele.dsrc, lambdaf)

            kernels = [pre_lusgs, lsweeps, usweeps]

            # LU-SGS for turbulent variables
            if hasattr(ele, 'mut'):
                # Get Python function of flux and wave speed for turbulent variables
                _tflux = ele.tflux_container()
                _tlambdaf = ele.make_turb_wave_speed()
                tnv = (ele.nfvars, ele.nvars)

                # Compile LU-SGS functions for turbulent variables
                _pre_tlusgs = make_lusgs_common(ele, _tlambdaf, factor=1.0)
                _tlsweep, _tusweep = make_serial_lusgs(
                    be, ele, tnv, mapping, unmapping, _tflux
                )

                # Initiate LU-SGS kernel objects for turbulent variables
                pre_tlusgs = Kernel(
                    be.make_loop(ele.neles, _pre_tlusgs), 
                    ele.upts[0], ele.dt, diag, lambdaf, mu, mut
                )                
                tlsweeps = Kernel(be.make_loop(ele.neles, func=_tlsweep),
                    ele.upts[0], ele.upts[1], ele.upts[2], diag, ele.dsrc, lambdaf) 

                tusweeps = Kernel(be.make_loop(ele.neles, func=_tusweep),
                    ele.upts[0], ele.upts[1], ele.upts[2], diag, ele.dsrc, lambdaf)    

                kernels += [pre_tlusgs, tlsweeps, tusweeps]             

            # Collect kernels and make meta kernels
            ele.lusgs = MetaKernel(kernels)

            # Update kernel
            ele.update = Kernel(be.make_loop(ele.neles, _update),
                ele.upts[0], ele.upts[1]
            )
    
    def step(self):
        resid = self.rhs(0, 1, is_norm=True)
        self.sys.eles.lusgs()
        self.sys.eles.update()

        self.sys.post(0)

        return 0, resid


class ColoredLUSGS(BaseSteadyIntegrator):
    name = 'colored-lu-sgs'
    nreg = 3

    def construct_stages(self):
        from pybaram.integrators.lusgs import  make_lusgs_common, make_lusgs_update, make_colored_lusgs

        be = self.be

        # colored-LU-SGS for each elements
        for ele in self.sys.eles:
            # Get Coloring result
            ncolor, icolor, lev_color = ele.coloring()
             
            # diagonal and lambda array
            diag = np.empty(ele.neles)
            lambdaf = np.empty((ele.nface, ele.neles))

            # Get Python functions of flux and wave speed
            _flux = ele.flux_container()
            _lambdaf = ele.make_wave_speed()
            nv = (0, ele.nfvars)

            # Get viscosity if exists
            mu = ele.mu if hasattr(ele, 'mu') else None
            mut = ele.mut if hasattr(ele, 'mut') else None

            # Compile LU-SGS functions
            _update = make_lusgs_update(ele)
            _pre_lusgs = make_lusgs_common(ele, _lambdaf, factor=1.0)
            _lsweep, _usweep = make_colored_lusgs(
                be, ele, nv, icolor, lev_color, _flux
            )

            # Initiate LU-SGS kernel objects
            pre_lusgs = Kernel(
                be.make_loop(ele.neles, _pre_lusgs), 
                ele.upts[0], ele.dt, diag, lambdaf, mu, mut
            )
            
            lsweeps = [
                Kernel(be.make_loop(n0=n0, ne=ne, func=_lsweep),
                ele.upts[0], ele.upts[1], ele.upts[2], diag, ele.dsrc, lambdaf
                ) 
                for n0, ne in zip(ncolor[:-1], ncolor[1:])
            ]

            usweeps = [
                Kernel(be.make_loop(n0=n0, ne=ne, func=_usweep),
                ele.upts[0], ele.upts[1], ele.upts[2], diag, ele.dsrc, lambdaf
                ) 
                for n0, ne in zip(ncolor[::-1][1:], ncolor[::-1][:-1])
            ]

            kernels = [pre_lusgs, *lsweeps, *usweeps]

            # LU-SGS for turbulent variables
            if hasattr(ele, 'mut'):
                # Get Python function of flux and wave speed for turbulent variables
                _tflux = ele.tflux_container()
                _tlambdaf = ele.make_turb_wave_speed()
                tnv = (ele.nfvars, ele.nvars)

                # Compile LU-SGS functions for turbulent variables
                _pre_tlusgs = make_lusgs_common(ele, _tlambdaf, factor=1.0)
                _tlsweep, _tusweep = make_colored_lusgs(
                    be, ele, tnv, icolor, lev_color, _tflux
                )

                # Initiate LU-SGS kernel objects for turbulent variables
                pre_tlusgs = Kernel(
                    be.make_loop(ele.neles, _pre_tlusgs), 
                    ele.upts[0], ele.dt, diag, lambdaf, mu, mut
                )                

                tlsweeps = [
                    Kernel(be.make_loop(n0=n0, ne=ne, func=_tlsweep),
                    ele.upts[0], ele.upts[1], ele.upts[2], diag, ele.dsrc, lambdaf
                    ) 
                    for n0, ne in zip(ncolor[:-1], ncolor[1:])
                ]

                tusweeps = [
                    Kernel(be.make_loop(n0=n0, ne=ne, func=_tusweep),
                    ele.upts[0], ele.upts[1], ele.upts[2], diag, ele.dsrc, lambdaf
                    ) 
                    for n0, ne in zip(ncolor[::-1][1:], ncolor[::-1][:-1])
                ] 

                kernels += [pre_tlusgs, *tlsweeps, *tusweeps]  

            # Collect kernels and make meta kernels
            ele.lusgs = MetaKernel(kernels)

            # Update kernel
            ele.update = Kernel(be.make_loop(ele.neles, _update),
                ele.upts[0], ele.upts[1]
            )
    
    def step(self):
        resid = self.rhs(0, 1, is_norm=True)
        self.sys.eles.lusgs()
        self.sys.eles.update()

        self.sys.post(0)

        return 0, resid

            



             

