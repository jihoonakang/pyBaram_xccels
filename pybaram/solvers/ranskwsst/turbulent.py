# -*- coding: utf-8 -*-
import numpy as np


def make_vorticity(be, cplargs):
    # Compute vorticity

    ndims = cplargs['ndims']

    if ndims == 2:
        def vorticity(uc, gc):
            inv_rho = 1/uc[0]
            u = uc[1] / inv_rho
            v = uc[2] / inv_rho

            rho_x = gc[0][0]
            rho_y = gc[1][0]

            u_y = gc[1][1] - u*rho_y
            v_x = gc[0][2] - v*rho_x 

            # Compute mean rotation rate and its magnitude
            w_xy = (u_y - v_x)*inv_rho
            return abs(w_xy)

    else:
        def vorticity(uc, gc):
            inv_rho = 1/uc[0]
            u = uc[1] * inv_rho
            v = uc[2] * inv_rho
            w = uc[3] * inv_rho

            rho_x = gc[0][0]
            rho_y = gc[1][0]
            rho_z = gc[2][0]

            u_y = gc[1][1] - u*rho_y
            u_z = gc[2][1] - u*rho_z

            v_x = gc[0][2] - v*rho_x
            v_z = gc[2][2] - v*rho_z

            w_x = gc[0][3] - w*rho_x
            w_y = gc[1][3] - w*rho_y

            # Compute mean rotation rate and its magnitude
            w_xy = (u_y - v_x)*inv_rho
            w_yz = (v_z - w_y)*inv_rho
            w_zx = (w_x - u_z)*inv_rho
            return np.sqrt(w_xy**2 + w_yz**2 + w_zx**2)

    # Compile
    return be.compile(vorticity)


def make_blendingF1(be, cplargs):
    ndims, nvars = cplargs['ndims'], cplargs['nvars']
    betast = cplargs['betast']
    sigmaw2 = cplargs['sigmaw2']
    
    def f1(uc, gc, mu, d):
        rho = uc[0]
        k, w = uc[nvars-2]/rho, uc[nvars-1]/rho

        term1 = np.sqrt(k) / (betast*w*d)
        term2 = 500*mu/rho / (w*d**2)

        # Compute dk/dx_i dw/dx_i
        kwcross = 0
        for i in range(ndims):
            rho_x = gc[i][0]
            k_x = (gc[i][nvars-2] - k*rho_x)/rho
            w_x = (gc[i][nvars-1] - w*rho_x)/rho
            kwcross += k_x*w_x

        cdkw = max(2*rho*sigmaw2/w*kwcross, 1e-20)        
        term3 = 4*rho*sigmaw2*k/cdkw/d**2

        arg1 = min(max(term1, term2), term3)

        return np.tanh(arg1**4)

    return be.compile(f1)


def make_blendingF2(be, cplargs):
    nvars = cplargs['nvars']
    betast = cplargs['betast']

    def f2(uc, mu, d):
        rho = uc[0]
        k, w = uc[nvars-2]/rho, uc[nvars-1]/rho

        term1 = (np.sqrt(k) / (betast*w*d))
        term2 = 500*mu/rho / (w*d**2)

        arg2 = max(2*term1, term2)

        return np.tanh(arg2**2)

    return be.compile(f2)
