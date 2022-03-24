# -*- coding: utf-8 -*-
from pybaram.utils.nb import dot
from pybaram.solvers.euler.bcs import (make_bc_far, make_bc_sup_out, make_bc_sup_in,
                                       make_bc_sub_inv, make_bc_sub_outp, make_bc_slip_wall)


def get_bc(self, name, bcargs):
    bc = eval('make_bc_'+name)
    return bc(bcargs)


def make_bc_adia_wall(bcargs):
    nvars, ndims = bcargs['nvars'], bcargs['ndims']

    def bc(ul, ur, nf):
        ur[0] = ul[0]

        for idx in range(ndims):
            ur[idx+1] = -ul[idx+1]

        ur[nvars-1] = ul[nvars-1]

    return bc


def make_bc_isotherm_wall(bcargs):
    nvars, ndims = bcargs['nvars'], bcargs['ndims']
    gamma = bcargs['gamma']
    pmin = bcargs['pmin']

    e = bcargs['cptw'] / gamma

    def bc(ul, ur, nf):
        p = max((gamma - 1)*(ul[nvars-1] - 0.5 *
                             dot(ul, ul, ndims, 1, 1)/ul[0]), pmin)
        ur[0] = p/e/(gamma-1)

        for idx in range(ndims):
            ur[idx+1] = -ul[idx+1]

        ur[nvars-1] = ur[0]*e

    return bc
