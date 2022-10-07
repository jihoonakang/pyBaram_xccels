# -*- coding: utf-8 -*-
from pybaram.backends import Backend
from pybaram.backends.cpu.loop import make_serial_loop1d, make_parallel_loop1d

import numba as nb
import os


class CPUBackend(Backend):
    """
    Backend for CPU computation
    - Support single thread and multi threads
    - Just-in Time compile via Numba
    """
    name = 'cpu'

    def __init__(self, cfg):
        # Get mutli-thread type
        self.multithread = multithread = cfg.get('backend-cpu', 'multi-thread', default='single')

        # Loop structure for multi-thread type
        if multithread == 'single':
            self.make_loop = make_serial_loop1d
            
            # Enforce to disable OpenMP
            os.environ['OMP_NUM_THREADS'] = '1'
        else:
            self.make_loop = make_parallel_loop1d

            # Threading layer selection
            if multithread in ['default', 'forksafe', 'threadsafe', 'safe', 'omp', 'tbb']:
                nb.config.THREADING_LAYER = multithread
    
    def compile(self, func, outer=False):
        # JIT compile the Python function
        if self.multithread == 'single' or not outer:
            return nb.jit(nopython=True, fastmath=True)(func)
        else:
            # Enable Numba parallelization if the function is not nested
            return nb.jit(nopython=True, fastmath=True, parallel=True)(func)
