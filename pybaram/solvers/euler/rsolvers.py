# -*- coding: utf-8 -*-
from pybaram.utils.nb import dot
from pybaram.utils.np import eps

import numpy as np
import re


def get_rsolver(name, be, cplargs):
    """
    docstring
    """
    fname = re.sub('\+', 'p', name)
    flux = eval('make_' + fname)(cplargs)

    return be.compile(flux)


def make_rusanov(cplargs):
    nvars, gamma = cplargs['nfvars'], cplargs['gamma']
    flux = cplargs['flux']
    locals = cplargs['locals']

    def rsolver(ul, ur, nf, fn):
        fl, fr = locals((nvars,)), locals((nvars,))
        
        pl, contravl = flux(ul, nf, fl)
        pr, contravr = flux(ur, nf, fr)

        contrav = 0.5*(contravl + contravr)
        an = np.sqrt(gamma*(pl+pr) / (ul[0] + ur[0])) + np.abs(contrav)

        for jdx in range(nvars):
            fn[jdx] = 0.5*(fl[jdx] + fr[jdx]) - 0.5*an*(ur[jdx] - ul[jdx])

    return rsolver


def make_roem(cplargs):
    ndims, nvars, gamma = cplargs['ndims'], cplargs['nfvars'], cplargs['gamma']
    flux = cplargs['flux']
    locals = cplargs['locals']

    def rsolver(ul, ur, nf, fn):
        fl, fr = locals((nvars,)), locals((nvars,))
        vl, vr = locals((ndims,)), locals((ndims,))
        dv, va = locals((ndims,)), locals((ndims,))
        du, bdq = locals((nvars,)), locals((nvars,))

        pl, contravl = flux(ul, nf, fl)
        pr, contravr = flux(ur, nf, fr)

        # Specific enthalpy, contra velocity for left / right
        for jdx in range(ndims):
            vl[jdx] = ul[jdx+1] / ul[0]
            vr[jdx] = ur[jdx+1] / ur[0]

        hl = (ul[nvars-1] + pl)/ul[0]
        hr = (ur[nvars-1] + pr)/ur[0]

        # Difference between two state
        drho = ur[0] - ul[0]
        dp = pr - pl
        dh = hr - hl
        dcontrav = contravr - contravl
        for jdx in range(ndims):
            dv[jdx] = vr[jdx] - vl[jdx]

        # Compute Roe averaged density and enthalpy
        rrr = np.sqrt(ur[0]/ul[0])
        ratl = 1.0/(1.0 + rrr)
        ratr = rrr*ratl
        ra = rrr*ul[0]
        ha = hl*ratl + hr*ratr

        for jdx in range(ndims):
            va[jdx] = vl[jdx]*ratl + vr[jdx]*ratr

        contrava = dot(va, nf, ndims)

        aa = np.sqrt((gamma - 1)*(ha - 0.5*dot(va, va, ndims)))
        rcp_aa = 1/aa

        # Compute |M|, add a small number to avoid a possible singularity of f
        abs_ma = np.abs(contrava*rcp_aa) + eps

        # Eigen structure
        b1 = max(0.0, max(contrava + aa, contravr + aa))
        b2 = min(0.0, min(contrava - aa, contravl - aa))

        #  Normalized wave speed
        b1b2 = b1*b2
        rcp_b1_b2 = 1.0/(b1 - b2)
        b1 = b1*rcp_b1_b2
        b2 = b2*rcp_b1_b2
        b1b2 = b1b2*rcp_b1_b2

        # 1-D shock discontinuity sensing term and Mach number based function f,g
        if pl < pr:
            SDST = pl / pr
        else:
            SDST = pr / pl

        h = 1.0 - SDST
        f = abs_ma**h
        g = f/(1.0 + abs_ma)

        for jdx in range(nvars-1):
            du[jdx] = ur[jdx] - ul[jdx]
        du[nvars-1] = ur[0]*hr - ul[0]*hl

        # BdQ
        bdq[0] = drho - f*dp*rcp_aa**2
        bdq[nvars - 1] = bdq[0]*ha + ra*dh
        for jdx in range(ndims):
            bdq[jdx+1] = bdq[0]*va[jdx] + ra*(dv[jdx] - nf[jdx]*dcontrav)

        for jdx in range(nvars):
            fn[jdx] = b1*fl[jdx] - b2*fr[jdx] + b1b2*(du[jdx] - g*bdq[jdx])

    return rsolver


def make_hllem(cplargs):
    ndims, nvars, gamma = cplargs['ndims'], cplargs['nfvars'], cplargs['gamma']
    flux = cplargs['flux']

    def prepare():
        fl, fr = np.empty(nvars), np.empty(nvars)
        vl, vr = np.empty(ndims), np.empty(ndims)
        dv, va = np.empty(ndims), np.empty(ndims)
        df = np.empty(nvars)
        return fl, fr, vl, vr, dv, va, df

    def rsolver(ul, ur, nf, fn, fl, fr, vl, vr, dv, va, df):
        pl, contravl = flux(ul, nf, fl)
        pr, contravr = flux(ur, nf, fr)

        # Specific enthalpy, contra velocity for left / right
        for jdx in range(ndims):
            vl[jdx] = ul[jdx+1] / ul[0]
            vr[jdx] = ur[jdx+1] / ur[0]

        hl = (ul[nvars-1] + pl)/ul[0]
        hr = (ur[nvars-1] + pr)/ur[0]

        # Compute Roe averaged density and enthalpy
        rrr = np.sqrt(ur[0]/ul[0])
        ratl = 1.0/(1.0 + rrr)
        ratr = rrr*ratl
        ra = rrr*ul[0]
        ha = hl*ratl + hr*ratr

        for jdx in range(ndims):
            va[jdx] = vl[jdx]*ratl + vr[jdx]*ratr

        qq = dot(va, va, ndims)
        contrava = dot(va, nf, ndims)
        aa = np.sqrt((gamma - 1)*(ha - 0.5*dot(va, va, ndims)))

        # Eigen structure
        b1 = max(0.0, max(contrava + aa, contravr + aa))
        b2 = min(0.0, min(contrava - aa, contravl - aa))

        # Contact term
        um = 0.5*(b1 + b2)
        delta = aa / (aa + abs(um))

        # Difference between two states
        drho = ur[0] - ul[0]
        dp = pr - pl
        dcontra = contravr - contravl

        for jdx in range(ndims):
            dv[jdx] = vr[jdx] - vl[jdx]

        alp1 = drho - dp/aa**2

        df[0] = alp1
        for jdx in range(ndims):
            df[jdx+1] = alp1*va[jdx]
        df[nvars-1] = 0.5*alp1*qq

        alp2 = ra
        for jdx in range(ndims):
            df[jdx+1] += alp2*(dv[jdx] - nf[jdx]*dcontra)
        df[nvars-1] += alp2*(dot(va, dv, ndims) - contrava*dcontra)

        #  Normalized wave speed
        b1b2 = b1*b2
        rcp_b1_b2 = 1.0/(b1 - b2)
        b1 = b1*rcp_b1_b2
        b2 = b2*rcp_b1_b2
        b1b2 = b1b2*rcp_b1_b2

        for jdx in range(nvars):
            fn[jdx] = b1*fl[jdx] - b2*fr[jdx] \
                + b1b2*((ur[jdx] - ul[jdx]) - delta*df[jdx])

    return prepare, rsolver


def make_ausmpwp(cplargs):
    ndims, nvars, gamma = cplargs['ndims'], cplargs['nfvars'], cplargs['gamma']
    to_primevars = cplargs['to_primevars']

    alpha = 3/16

    def prepare():
        vl, vr = np.empty(ndims), np.empty(ndims)
        return vl, vr

    def rsolver(ul, ur, nf, fn, vl, vr):
        pl = to_primevars(ul, vl)
        pr = to_primevars(ur, vr)

        # Specific enthalpy, contra velocity, tangential velocity for left / right
        hl = (ul[nvars-1] + pl)/ul[0]
        hr = (ur[nvars-1] + pr)/ur[0]

        contral = dot(vl, nf, ndims)
        contrar = dot(vr, nf, ndims)
        vl2 = dot(vl, vl, ndims) - contral**2
        vr2 = dot(vr, vr, ndims) - contrar**2

        conmid = 0.5*(contral + contrar)

        # Specific enthalpy along normal
        hn = 0.5*(hl - 0.5*vl2 + hr - 0.5*vr2)
        cs2 = 2.0*(gamma - 1)/(gamma + 1)*hn
        cs = np.sqrt(cs2)

        # Speed of sound at midpoint, mach number
        if conmid > 0:
            cmid = cs2/max(abs(contral), cs)
        else:
            cmid = cs2/max(abs(contrar), cs)

        ml = contral/cmid
        mr = contrar/cmid

        # AUSM-type function
        if abs(ml) < 1.0:
            mlp = 0.25*(ml + 1.0)**2
            plp = mlp*(2.0 - ml) + alpha*ml*(ml**2 - 1.0)**2
        else:
            mlp = 0.5*(ml + abs(ml))
            plp = mlp / ml

        if abs(mr) < 1.0:
            mrm = -0.25*(mr - 1.0)**2
            prm = -mrm*(2.0 + mr) - alpha*mr*(mr**2 - 1.0)**2
        else:
            mrm = 0.5*(mr - abs(mr))
            prm = mrm / mr

        m_mid = 0.5*(mlp + mrm)
        p_mid = plp*pl + prm*pr

        # 1-D shock discontinuity sensing term and pressure based function f, w
        if pl < pr:
            SDST = pl / pr
        else:
            SDST = pr / pl

        SDST2 = SDST**2

        if p_mid > eps:
            fl = (pl/p_mid - 1.0)*SDST2
            fr = (pr/p_mid - 1.0)*SDST2
        else:
            fl = 0
            fr = 0

        wei = 1.0 - SDST*SDST2

        # M+. M-
        if m_mid > 0.0:
            mp = mlp + mrm*((1.0 - wei)*(1.0 + fr) - fl)
            mm = mrm*wei*(1.0 + fr)
        else:
            mp = mlp*wei*(1.0 + fl)
            mm = mrm + mlp*((1.0 - wei)*(1.0 + fl) - fr)

        # Flux
        for jdx in range(nvars - 1):
            fn[jdx] = cmid*(mp*ul[jdx] + mm*ur[jdx])

        for jdx in range(ndims):
            fn[jdx + 1] += nf[jdx] * p_mid

        fn[nvars - 1] = cmid*(mp*ul[0]*hl + mm*ur[0]*hr)

    return prepare, rsolver


def make_ausmpup(cplargs):
    ndims, nvars, gamma = cplargs['ndims'], cplargs['nfvars'], cplargs['gamma']
    to_primevars = cplargs['to_primevars']

    alpha, beta = 3/16, 1/8
    kp, ku = 1, 1

    def prepare():
        vl, vr = np.empty(ndims), np.empty(ndims)
        return vl, vr

    def rsolver(ul, ur, nf, fn, vl, vr):
        pl = to_primevars(ul, vl)
        pr = to_primevars(ur, vr)

        # Specific enthalpy, contra velocity, tangential velocity for left / right
        hl = (ul[nvars-1] + pl)/ul[0]
        hr = (ur[nvars-1] + pr)/ur[0]

        contral = dot(vl, nf, ndims)
        contrar = dot(vr, nf, ndims)

        csl = 2.0*(gamma - 1)/(gamma + 1)*hl
        csr = 2.0*(gamma - 1)/(gamma + 1)*hr

        ccl = csl*csl/max(contral, csl)
        ccr = csr*csr/max(-contrar, csr)

        # Speed of sound at midpoint, mach number
        rhmid = 0.5*(ul[0] + ur[0])
        cmid = min(ccl, ccr)
        ml = contral/cmid
        mr = contrar/cmid

        # AUSM-type function
        if abs(ml) < 1.0:
            mlp = 0.25*(ml + 1.0)**2
            plp = mlp*(2.0 - ml) + alpha*ml*(ml**2 - 1.0)**2
            mlp += beta*(ml**2 - 1.0)**2
        else:
            mlp = 0.5*(ml + abs(ml))
            plp = mlp / ml

        if abs(mr) < 1.0:
            mrm = -0.25*(mr - 1.0)**2
            prm = -mrm*(2.0 + mr) - alpha*mr*(mr**2 - 1.0)**2
            mrm -= beta*(mr**2 - 1.0)**2
        else:
            mrm = 0.5*(mr - abs(mr))
            prm = mrm / mr

        # M+. M-
        mdp = 0.5*(ml**2 + mr**2)
        mdp = -kp*max(1 - mdp, 0.0)*(pr - pl) / (rhmid*cmid**2)
        mmid = mlp + mrm + mdp
        mp = 0.5*(mmid + abs(mmid))
        mm = 0.5*(mmid - abs(mmid))

        pu = -ku*plp*abs(prm)*rhmid*cmid*(contrar - contral)
        p_mid = plp*pl + prm*pr + pu

        # Flux
        for jdx in range(nvars - 1):
            fn[jdx] = cmid*(mp*ul[jdx] + mm*ur[jdx])

        for jdx in range(ndims):
            fn[jdx + 1] += nf[jdx] * p_mid

        fn[nvars - 1] = cmid*(mp*ul[0]*hl + mm*ur[0]*hr)

    return prepare, rsolver
