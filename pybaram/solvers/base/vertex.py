# -*- coding: utf-8 -*-
import numpy as np


class BaseVertex:
    def __init__(self, be, cfg, elemap, vtx, ivtx, neivtx):
        # Save arguments
        self.be = be
        self.cfg = cfg

        # Number of vertex
        self.nvtx = len(ivtx) - 1
        
        # Dimensions from element
        ele0 = elemap[next(iter(elemap))]
        self.ndims, self.nvars = ele0.ndims, ele0.nvars
        self.primevars = ele0.primevars

        # Get index of vertex
        self._idx = self._get_index(elemap, vtx)
        self._ivtx = ivtx
        self._neivtx = neivtx

    def _get_index(self, elemap, vtx):
        # Parse index of vertex
        cell_nums = {c: i for i, c in enumerate(elemap)}
        return np.array([[cell_nums[t], e, v] for t, e, v, z in vtx]).T.copy()
