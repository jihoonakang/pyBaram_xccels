# -*- coding: utf-8 -*-
from pybaram.utils.nb import dot

import numpy as np


def get_bc(self, name, bcargs):
    bc = eval('make_bc_'+name)
    return bc(bcargs)


def make_bc_sup_out(bcargs):
    nvars = bcargs['nvars']

    def bc(ul, ur, nf):
        for idx in range(nvars):
            ur[idx] = ul[idx]

    return bc


def make_bc_sup_in(bcargs):
    nvars, ndims = bcargs['nvars'], bcargs['ndims']
    gamma = bcargs['gamma']
    rho, p = bcargs['rho'], bcargs['p']

    # Conservative variable at boundary
    ub = np.empty(nvars)
    ub[0] = rho
    for i, k in enumerate('uvw'[:ndims]):
        ub[i+1] = rho*bcargs[k]

    ub[nvars-1] = p/(gamma-1) + 0.5*sum(ub[1:-1]**2)/rho

    def bc(ul, ur, nf):
        for idx in range(nvars):
            ur[idx] = ub[idx]

    return bc


def make_bc_sub_inv(bcargs):
    nvars, ndims = bcargs['nvars'], bcargs['ndims']
    gamma = bcargs['gamma']
    rho, pmin = bcargs['rho'], bcargs['pmin']

    # Conservative variable at boundary
    ub = np.empty(nvars-1)
    ub[0] = rho
    for i, k in enumerate('uvw'[:ndims]):
        ub[i+1] = rho*bcargs[k]

    qb = 0.5*dot(ub, ub, ndims, 1, 1) / ub[0]

    def bc(ul, ur, nf):
        for idx in range(nvars-1):
            ur[idx] = ub[idx]

        pl = max((gamma - 1)*(ul[nvars-1] - 0.5 *
                              dot(ul, ul, ndims, 1, 1)/ul[0]), pmin)
        ur[nvars-1] = pl / (gamma-1) + qb

    return bc


def make_bc_sub_outp(bcargs):
    nvars, ndims = bcargs['nvars'], bcargs['ndims']
    gamma, p = bcargs['gamma'], bcargs['p']

    def bc(ul, ur, nf):
        for idx in range(nvars-1):
            ur[idx] = ul[idx]

        ur[nvars-1] = p / (gamma-1) + 0.5*dot(ur, ur, ndims, 1, 1)/ur[0]

    return bc


def make_bc_slip_wall(bcargs):
    nvars, ndims = bcargs['nvars'], bcargs['ndims']

    def bc(ul, ur, nf):
        vn = dot(ul, nf, ndims, 1)
        ur[0] = ul[0]

        for idx in range(ndims):
            ur[idx+1] = ul[idx+1] - 2*vn*nf[idx]

        ur[nvars-1] = ul[nvars-1]

    return bc


def make_bc_far(bcargs):
    nvars, ndims = bcargs['nvars'], bcargs['ndims']
    gamma, pmin = bcargs['gamma'], bcargs['pmin']
    rho, p = bcargs['rho'], bcargs['p']

    # Speed of sound, entropy at bc
    cb = np.sqrt(gamma*p/rho)
    sb = p / rho**gamma
    cb_gmo = 2*cb/(gamma-1)
    vb = np.array([bcargs[k] for k in 'uvw'[:ndims]])

    def bc(ul, ur, nf):
        # Contravariant velocity
        contrab = dot(vb, nf, ndims)

        # speed of sound, entropy at left
        rhol = ul[0]
        contral = dot(ul, nf, ndims, 1)/rhol
        pl = max((gamma - 1)*(ul[nvars-1] - 0.5 *
                              dot(ul, ul, ndims, 1, 1)/rhol), pmin)
        cl = np.sqrt(gamma*pl/rhol)

        # Riem-
        if abs(contrab) >= cb and contral >= 0:
            rm = contral - 2*cl/(gamma-1)
        else:
            rm = contrab - cb_gmo

        # Riem+
        if abs(contrab) >= cb and contral < 0:
            rp = contrab + cb_gmo
        else:
            rp = contral + 2*cl/(gamma-1)

        # Characteristic
        contra = 0.5*(rp + rm)
        c = 0.25*(gamma - 1)*(rp - rm)

        if contral < 0:
            rho = ((1.0/(gamma*sb))*c**2)**(1/(gamma-1))
        else:
            rho = (rhol**gamma*c**2/(gamma*pl))**(1/(gamma-1))

        p = rho*c**2/gamma

        ur[0] = rho
        if contral >= 0:
            for i in range(ndims):
                ur[i+1] = rho*(ul[i + 1]/rhol + (contra - contral)*nf[i])
        else:
            for i in range(ndims):
                ur[i+1] = rho*(vb[i] + (contra - contrab)*nf[i])

        ur[nvars-1] = p / (gamma-1) + 0.5*dot(ur, ur, ndims, 1, 1)/rho

    return bc
