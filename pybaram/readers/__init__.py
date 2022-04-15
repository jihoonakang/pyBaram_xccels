# -*- coding: utf-8 -*-

from pybaram.readers.base import BaseReader
from pybaram.readers.cgns import CGNSReader
from pybaram.readers.gmsh import GMSHReader

from pybaram.utils.misc import subclass_by_name, subclasses


def get_reader(extn, *args, **kwargs):
    reader_map = {ex : cls for cls in subclasses(BaseReader) for ex in cls.extn}

    return reader_map[extn](*args, **kwargs)
