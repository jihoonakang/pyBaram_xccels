import numpy as np


class Backend:
    name = 'base'
    # Floating point data type
    _fpdtype_map = {'double' : np.float64, 'single' : np.float32}

    def __init__(self, fpdtype='double'):
        self._fpdtype=self._fpdtype_map[fpdtype]