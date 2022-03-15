# -*- coding: utf-8 -*-
from pybaram.backends import Backend
from pybaram.backends.cpu.loop import make_serial_loop1d, make_parallel_loop1d

import numba as nb


class CPUBackend(Backend):
    """
    CPU 계산용 Backend
    - single thread 및 Multi-thread 지원
    - Numba를 이용하여 Just-in Time compile
    """
    name = 'cpu'

    def __init__(self, cfg):
        # Mutli-thread type
        multithread = cfg.get('backend-cpu', 'multithread', default='single')

        # Loop 함수 설정
        if multithread == 'single':
            self.make_loop = make_serial_loop1d
        else:
            self.make_loop = make_parallel_loop1d

            # Threading layer selection
            if multithread in ['default', 'forksafe', 'threadsafe', 'safe', 'omp', 'tbb']:
                nb.config.THREADING_LAYER = multithread
    
    def compile(self, func):
        return nb.jit(nopython=True, fastmath=True)(func)

