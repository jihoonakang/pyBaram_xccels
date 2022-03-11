# -*- coding: utf-8 -*-
from pybaram.backend.base import Backend
from pybaram.backend.cpu.backend import CPUBackend
from pybaram.utils.misc import subclass_by_name


def get_backend(name, *args, **kwargs):
    return subclass_by_name(Backend, name)(*args, **kwargs)
