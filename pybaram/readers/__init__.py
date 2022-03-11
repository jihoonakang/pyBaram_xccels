# -*- coding: utf-8 -*-

from pybaram.readers.cgns import CGNSReader


def get_reader(*args, **kwargs):
    return CGNSReader(*args, **kwargs)