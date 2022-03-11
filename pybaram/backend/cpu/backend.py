# -*- coding: utf-8 -*-
from pybaram.backend import Backend
from pybaram.backend.cpu.loop import make_serial_loop1d, make_parallel_loop1d

import numba as nb
import numpy as np


class CPUBackend(Backend):
    name = 'cpu'

    def __init__(self, fpdtype='double', is_parallel=False, threadlayer='default'):
        super().__init__(fpdtype)

        # Loop1d 함수 연결
        if not is_parallel:
            self.make_loop = make_serial_loop1d
        else:
            self.make_loop = make_parallel_loop1d

            # Threading layer selection
            if threadlayer in ['default', 'forksafe', 'threadsafe', 'safe']:
                nb.config.THREADING_LAYER=threadlayer
    
    def array(self, arr, dtype='none'):
        if dtype == 'none':
            dtype = self._fpdtype
        # Data allocation
        return np.array(arr, dtype=dtype)

    def ones(self, n):
        return np.ones(n, dtype=self._fpdtype)

    def zeros(self, n):
        return np.zeros(n, self._fpdtype)

    def empty(self, n):
        return np.empty(n, self._fpdtype)
