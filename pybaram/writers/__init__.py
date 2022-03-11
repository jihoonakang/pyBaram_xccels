# -*- coding: utf-8 -*-
from pybaram.utils.misc import subclass_by_name
from pybaram.writers.base import BaseWriter
from pybaram.writers.vtk import VTKWriter
from pybaram.writers.tecplot import TecplotWriter


def get_writer(meshf, solnf, outf, **kwargs):
    suffix = outf.split('.')[-1]

    writer = subclass_by_name(BaseWriter, suffix)

    return writer(meshf, solnf, outf, **kwargs)
