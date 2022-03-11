# -*- coding: utf-8 -*-
import os


def csv_write(fname, header):
    outf = open(fname, 'a')

    if os.path.getsize(fname) == 0:
        print(','.join(header), file=outf)

    return outf


class BasePlugin:
    name = None
