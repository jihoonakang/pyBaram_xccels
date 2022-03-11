from mpi4py import MPI


class ArrayBank:
    def __init__(self, mat, idx):
        self.idx = idx
        self.mat = mat

    @property
    def value(self):
        return self.mat[self.idx]


class NullKernel:
    def __call__(self, *args, **kwargs):
        pass


class Kernel:
    def __init__(self, fun, *args,  **kwargs):
        self._fun = fun
        self._args, self._kwargs = args, kwargs

    def __call__(self, *args, **kwargs):
        args = self._args + args
        kwargs.update(self._kwargs)

        # Parse args
        args = [arg.value if hasattr(arg, 'value') else arg for arg in args]

        # Run Kernel
        self._fun(*args, **kwargs)

    def update_args(self, *args):
        self._args = args


class Queue:
    def __init__(self):
        self._reqs = []

    def sync(self):
        MPI.Prequest.Waitall(self._reqs)
        self._reqs = []

    def register(self, *reqs):
        for req in reqs:
            self._reqs.append(req)
