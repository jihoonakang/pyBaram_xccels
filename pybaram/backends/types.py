# -*- coding: utf-8 -*-
from mpi4py import MPI
    

class ArrayBank:
    def __init__(self, mat, idx):
        self.idx = idx
        self.mat = mat

    @property
    def value(self):
        return self.mat[self.idx]


class NullKernel:
    def __call__(self, *args):
        pass


class Kernel:
    def __init__(self, fun, *args, arg_trans_pos=False):
        self._fun = fun
        self._args = args

        # Argument 전치 여부 (call 에서 받는 Argument)
        if arg_trans_pos:
            self._sum_args = lambda x, y : y + x 
        else:
            self._sum_args = lambda x, y : x + y

    def __call__(self, *args):
        # Argument 결합
        args = self._sum_args(self._args, args)

        # Parse args (for ArrayBank...)
        args = [arg.value if hasattr(arg, 'value') else arg for arg in args]

        # Run Kernel
        self._fun(*args)

    def update_args(self, *args):
        self._args = args


class Queue:
    """
    Simple Queue
    - 저장된 Kernel을 모두 계산
    - sync 호출 시 MPI 통신 완료 
    """
    def __init__(self):
        self._reqs = []

    def sync(self):
        MPI.Prequest.Waitall(self._reqs)
        self._reqs = []

    def register(self, *reqs):
        for req in reqs:
            self._reqs.append(req)
