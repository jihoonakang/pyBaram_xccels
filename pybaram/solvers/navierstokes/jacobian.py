def get_viscous_jacobian(name, be, cplargs):
    return eval('make_'+name+'_jacobian')(be, cplargs)


# Thin layer Navier-Stokes Jacobian
def make_tlns_jacobian(be, cplargs):

    # Ref) Blazek, J. (2005).
    # Computational Fluid Dynamics: Principles and Applications (2nd ed.).
    # Elsevier.

    # Constants
    pr, gamma = cplargs['pr'], cplargs['gamma']
    ndims = cplargs['ndims']
    
    def tlns2d(uf, nf, A, gf, idx, mu_arr):
        # Basic variables
        nx = nf[0]
        ny = nf[1]

        inv_rho = 1/uf[0]
        u = uf[1]*inv_rho
        v = uf[2]*inv_rho
        e = uf[3]*inv_rho

        mu = mu_arr[idx]

        # Gradient at cell
        rho_x = gf[0][0]
        rho_y = gf[1][0]

        u_x = inv_rho*(gf[0][1] - u*rho_x)
        u_y = inv_rho*(gf[1][1] - u*rho_y)
        v_x = inv_rho*(gf[0][2] - v*rho_x)
        v_y = inv_rho*(gf[1][2] - v*rho_y)

        e_x = inv_rho*(gf[0][3] - e*rho_x)
        e_y = inv_rho*(gf[1][3] - e*rho_y)

        # Jacobian variables
        a1 = nx**2/3.0 + 1.0
        a2 = nx*ny/3.0
        a3 = ny**2/3.0 + 1.0
        a4 = gamma/pr

        c1 = inv_rho*((u_x - u*inv_rho*rho_x)*nx \
                        + (u_y - u*inv_rho*rho_y)*ny)
        c2 = inv_rho*((v_x - v*inv_rho*rho_x)*nx \
                        + (v_y - v*inv_rho*rho_y)*ny)
        c3 = -inv_rho*inv_rho*(rho_x * nx + rho_y * ny)
        c4 = u*inv_rho*((2.*u_x - u*inv_rho*rho_x)*nx \
                        + (2.*u_y - u*inv_rho*rho_y)*ny)
        c5 = v*inv_rho*((2.*v_x - v*inv_rho*rho_x)*nx \
                        + (2.*v_y - v*inv_rho*rho_y)*ny)
        c6 = inv_rho*((v*u_x + u*v_x - u*v*inv_rho*rho_x)*nx \
                      + (v*u_y + u*v_y - u*v*inv_rho*rho_y)*ny)
        c7 = inv_rho*((e_x - e*inv_rho*rho_x)*nx + (e_y - e*inv_rho*rho_y)*ny)
        
        b_21 = -a1*c1 - a2*c2
        b_31 = -a2*c1 - a3*c2
        
        # Computes Jacobian matrix
        A[1][0] -= mu*b_21
        A[1][1] -= mu*a1*c3
        A[1][2] -= mu*a2*c3
        A[2][0] -= mu*b_31
        A[2][1] -= mu*a2*c3
        A[2][2] -= mu*a3*c3
        A[3][0] -= mu*((a4-a1)*c4 + (a4-a3)*c5 - 2.*a2*c6 - a4*c7)
        A[3][1] -= -mu*(a4*c1 + b_21)
        A[3][2] -= -mu*(a4*c2 + b_31)
        A[3][3] -= mu*a4*c3

    def tlns3d(uf, nf, A, gf, idx, mu_arr):
        # Basic variables
        nx = nf[0]
        ny = nf[1]
        nz = nf[2]

        inv_rho = 1/uf[0]
        u = uf[1]*inv_rho
        v = uf[2]*inv_rho
        w = uf[3]*inv_rho
        e = uf[4]*inv_rho

        mu = mu_arr[idx]

        # Gradient at cell
        rho_x = gf[0][0]
        rho_y = gf[1][0]
        rho_z = gf[2][0]

        u_x = inv_rho*(gf[0][1] - u*rho_x)
        u_y = inv_rho*(gf[1][1] - u*rho_y)
        u_z = inv_rho*(gf[2][1] - u*rho_z)
        v_x = inv_rho*(gf[0][2] - v*rho_x)
        v_y = inv_rho*(gf[1][2] - v*rho_y)
        v_z = inv_rho*(gf[2][2] - v*rho_z)
        w_x = inv_rho*(gf[0][3] - w*rho_x)
        w_y = inv_rho*(gf[1][3] - w*rho_y)
        w_z = inv_rho*(gf[2][3] - w*rho_z)
        
        e_x = inv_rho*(gf[0][4] - e*rho_x)
        e_y = inv_rho*(gf[1][4] - e*rho_y)
        e_z = inv_rho*(gf[2][4] - e*rho_z)

        # Derivatives
        rho_psi = rho_x*nx + rho_y*ny + rho_z*nz
        inv_rho_psi = -rho_psi*inv_rho*inv_rho
        u_psi = u_x*nx + u_y*ny + u_z*nz
        v_psi = v_x*nx + v_y*ny + v_z*nz
        w_psi = w_x*nx + w_y*ny + w_z*nz
        e_psi = e_x*nx + e_y*ny + e_z*nz

        u_rho_psi = inv_rho*u_psi - u*inv_rho*inv_rho*rho_psi
        v_rho_psi = inv_rho*v_psi - v*inv_rho*inv_rho*rho_psi
        w_rho_psi = inv_rho*w_psi - w*inv_rho*inv_rho*rho_psi
        e_rho_psi = inv_rho*e_psi - e*inv_rho*inv_rho*rho_psi

        u2_rho_psi = 2.*u*inv_rho*u_psi - u**2*inv_rho*inv_rho*rho_psi
        v2_rho_psi = 2.*v*inv_rho*v_psi - v**2*inv_rho*inv_rho*rho_psi
        w2_rho_psi = 2.*w*inv_rho*w_psi - w**2*inv_rho*inv_rho*rho_psi
        uv_rho_psi = u*inv_rho*v_psi + v*inv_rho*u_psi - u*v*inv_rho*inv_rho*rho_psi
        uw_rho_psi = u*inv_rho*w_psi + w*inv_rho*u_psi - u*w*inv_rho*inv_rho*rho_psi
        vw_rho_psi = v*inv_rho*w_psi + w*inv_rho*v_psi - v*w*inv_rho*inv_rho*rho_psi
        
        # Temporal variables
        a1 = nx**2/3.0 + 1.0
        a2 = nx*ny/3.0
        a3 = nx*nz/3.0
        a4 = ny**2/3.0 + 1.0
        a5 = ny*nz/3.0
        a6 = nz**2/3.0 + 1.0
        a7 = gamma/pr

        b_21 = - a1*u_rho_psi - a2*v_rho_psi - a3*w_rho_psi
        b_31 = - a2*u_rho_psi - a4*v_rho_psi - a5*w_rho_psi
        b_41 = - a3*u_rho_psi - a5*v_rho_psi - a6*w_rho_psi

        b_51 = (a7-a1)*u2_rho_psi + (a7-a4)*v2_rho_psi + (a7-a6)*w2_rho_psi \
                - a7*e_rho_psi \
                - 2.0*a2*uv_rho_psi - 2.0*a3*uw_rho_psi - 2.0*a5*vw_rho_psi
        
        b_52 = -a7*u_rho_psi - b_21
        b_53 = -a7*v_rho_psi - b_31
        b_54 = -a7*w_rho_psi - b_41

        # Jacobian matrix
        A[1][0] -= mu*b_21
        A[1][1] -= mu*a1*inv_rho_psi
        A[1][2] -= mu*a2*inv_rho_psi
        A[1][3] -= mu*a3*inv_rho_psi
        A[2][0] -= mu*b_31
        A[2][1] -= mu*a2*inv_rho_psi
        A[2][2] -= mu*a4*inv_rho_psi
        A[2][3] -= mu*a5*inv_rho_psi
        A[3][0] -= mu*b_41
        A[3][1] -= mu*a3*inv_rho_psi
        A[3][2] -= mu*a5*inv_rho_psi
        A[3][3] -= mu*a6*inv_rho_psi
        A[4][0] -= mu*b_51
        A[4][1] -= mu*b_52
        A[4][2] -= mu*b_53
        A[4][3] -= mu*b_54
        A[4][4] -= a7*inv_rho_psi
    
    if ndims == 2:
        return be.compile(tlns2d)
    elif ndims == 3:
        return be.compile(tlns3d)


# Exact viscous Jacobian matrix
def make_exact_jacobian(be, cplargs):

    # Ref) T. J. Chung. (2010).
    # Computational Fluid Dynamics (2nd ed.).
    # Cambridge University Press.

    # Constants
    pr, gamma = cplargs['pr'], cplargs['gamma']
    ndims = cplargs['ndims']

    def visjacobian2d(uf, nf, A, gf, idx, mu_arr):
        
        mu = mu_arr[idx]
        mur = 4.0/3.0*mu
        lam = -2.0/3.0*mu

        inv_rho = 1/uf[0]
        u = uf[1]*inv_rho
        v = uf[2]*inv_rho
        e = uf[3]*inv_rho

        nx = nf[0]
        ny = nf[1]

        # Basic difference
        rho_x = gf[0][0]
        rho_y = gf[1][0]

        rhou_x = gf[0][1]
        rhou_y = gf[1][1]

        rhov_x = gf[0][2]
        rhov_y = gf[1][2]

        rhoe_x = gf[0][3]
        rhoe_y = gf[1][3]

        u_x = inv_rho*(rhou_x - u*rho_x)
        u_y = inv_rho*(rhou_y - u*rho_y)
        v_x = inv_rho*(rhov_x - v*rho_x)
        v_y = inv_rho*(rhov_y - v*rho_y)

        tau_xx = mur*u_x + lam*v_y
        tau_yy = mur*v_y + lam*u_x
        tau_xy = mu*(u_y + v_x)

        # Viscous flux Jacobian
        inv_rho2 = inv_rho*inv_rho
        txx_0 = -inv_rho2*(mur*(rhou_x - 2*u*rho_x) + lam*(rhov_y - 2*v*rho_y))
        txx_1 = -inv_rho2*mur*rho_x
        txx_2 = -inv_rho2*lam*rho_y
        txy_0 = -inv_rho2*mu*(rhou_y + rhov_x - 2*u*rho_y - 2*v*rho_x)
        txy_1 = -inv_rho2*mu*rho_y
        txy_2 = -inv_rho2*mu*rho_x
        tyy_0 = -inv_rho2*(mur*(rhov_y - 2*v*rho_y) + lam*(rhou_x - 2*u*rho_x))
        tyy_1 = -inv_rho2*lam*rho_x
        tyy_2 = -inv_rho2*mur*rho_y

        f30 = u*txx_0 + v*txy_0 - inv_rho*(u*tau_xx+v*tau_xy) \
                                + mu*gamma*inv_rho2/pr \
                                * (-rhoe_x + (2*e-3*u*u-3*v*v)*rho_x + 2*u*rhou_x + 2*v*rhov_x)
        f31 = u*txx_1 + inv_rho * tau_xx - mu*gamma*inv_rho2/pr \
                                * (rhou_x - 2*u*rho_x) + v*txy_1
        f32 = v*txy_2 + inv_rho * tau_xy - mu*gamma*inv_rho2/pr \
                                * (rhov_x - 2*v*rho_x) + u*txx_2
        f33 = -mu*gamma*inv_rho2/pr * rho_x

        g30 = u*txy_0 + v*tyy_0 - inv_rho*(u*tau_xy + v*tau_yy) \
                                + mu*gamma*inv_rho2/pr \
                                * (-rhoe_y + (2*e-3*u*u-3*v*v)*rho_y + 2*u*rhou_y + 2*v*rhov_y)
        g31 = u*txy_1 + inv_rho * tau_xy - mu*gamma*inv_rho2/pr \
                                * (rhou_y - 2*u*rho_y) + v*tyy_1
        g32 = u*txy_2 + inv_rho * tau_yy - mu*gamma*inv_rho2/pr \
                                * (rhov_y - 2*v*rho_y) + v*tyy_2
        g33 = -mu*gamma*inv_rho2/pr * rho_y

        A[1][0] -= txx_0*nx + txy_0*ny
        A[1][1] -= txx_1*nx + txy_1*ny
        A[1][2] -= txx_2*nx + txy_2*ny
        A[2][0] -= txy_0*nx + tyy_0*ny
        A[2][1] -= txy_1*nx + tyy_1*ny
        A[2][2] -= txy_2*nx + tyy_2*ny
        A[3][0] -= f30*nx + g30*ny
        A[3][1] -= f31*nx + g31*ny
        A[3][2] -= f32*nx + g32*ny
        A[3][3] -= f33*nx + g33*ny
        
    def visjacobian3d(uf, nf, A, gf, idx, mu_arr):
        # Basic variables
        inv_rho = 1/uf[0]
        u = uf[1]*inv_rho
        v = uf[2]*inv_rho
        w = uf[3]*inv_rho
        e = uf[4]*inv_rho

        nx = nf[0]
        ny = nf[1]
        nz = nf[2]

        # Constants
        inv_rho2 = inv_rho*inv_rho
        mu = mu_arr[idx]
        mur = 4.0/3.0*mu
        lam = -2.0/3.0*mu
        k = inv_rho2*mu*gamma/pr

        # Basic difference
        rho_x = gf[0][0]
        rho_y = gf[1][0]
        rho_z = gf[2][0]

        rhou_x = gf[0][1]
        rhou_y = gf[1][1]
        rhou_z = gf[2][1]

        rhov_x = gf[0][2]
        rhov_y = gf[1][2]
        rhov_z = gf[2][2]

        rhow_x = gf[0][3]
        rhow_y = gf[1][3]
        rhow_z = gf[2][3]

        rhoe_x = gf[0][4]
        rhoe_y = gf[1][4]
        rhoe_z = gf[2][4]

        u_x = inv_rho*(rhou_x - u*rho_x)
        u_y = inv_rho*(rhou_y - u*rho_y)
        u_z = inv_rho*(rhou_z - u*rho_z)
        v_x = inv_rho*(rhov_x - v*rho_x)
        v_y = inv_rho*(rhov_y - v*rho_y)
        v_z = inv_rho*(rhov_z - v*rho_z)
        w_x = inv_rho*(rhow_x - w*rho_x)
        w_y = inv_rho*(rhow_y - w*rho_y)
        w_z = inv_rho*(rhow_z - w*rho_z)

        # Stress tensor
        t_xx = 2*mu*inv_rho*(u_x - 1/3*(u_x + v_y + w_z))
        t_yy = 2*mu*inv_rho*(v_y - 1/3*(u_x + v_y + w_z))
        t_zz = 2*mu*inv_rho*(w_z - 1/3*(u_x + v_y + w_z))

        t_xy = mu*inv_rho*(v_x + u_y)
        t_yz = mu*inv_rho*(w_y + v_z)
        t_xz = mu*inv_rho*(u_z + w_x)

        # Jacobian components
        f21 = -inv_rho2*(mur*rhou_x + lam*(rhov_y+rhow_z) \
                        - mur*(2*u*rho_x - v*rho_y - w*rho_z))
        f22 = -inv_rho2*mur*rho_x
        f23 = -inv_rho2*lam*rho_y
        f24 = -inv_rho2*lam*rho_z
        f31 = -inv_rho2*mu*(rhou_y+rhov_x - 2*u*rho_y - 2*v*rho_x)
        f32 = -inv_rho2*mu*rho_y
        f33 = -inv_rho2*mu*rho_x
        f34 = 0.0
        f41 = -inv_rho2*mu*(rhou_z+rhow_x - 2*u*rho_z - 2*w*rho_x)
        f42 = -inv_rho2*mu*rho_z
        f43 = 0.0
        f44 = -inv_rho2*mu*rho_x
        f51 = -inv_rho*(u*t_xx+v*t_xy+w*t_xz) + u*f21 + v*f31 + w*f41 \
                - k * (rhoe_x - (2*e-3*u*u-3*v*v-3*w*w)*rho_x \
                    - 2*u*rhou_x - 2*v*rhov_x - 2*w*rhow_x)
        f52 = u * f22 + v*f32 + w*f42 + inv_rho*t_xx \
                + k * (2*u*rho_x - rhou_x)
        f53 = u * f23 + v*f33 + w*f43 + inv_rho*t_xy \
                + k * (2*v*rho_x - rhov_x)
        f54 = u * f24 + v*f34 + w*f44 + inv_rho*t_xz \
                + k * (2*w*rho_x - rhow_x)
        f55 = -k * rho_x

        g21 = f31
        g22 = f32
        g23 = f33
        g24 = 0.0
        g31 = -inv_rho2*(lam*(rhou_x+rhow_z) \
                        + mur*(rhov_y + u*rho_x - 2*v*rho_y + w*rho_z))
        g32 = -inv_rho2*lam*rho_x
        g33 = -inv_rho2*mur*rho_y
        g34 = -inv_rho2*lam*rho_z
        g41 = -inv_rho2*mu*(rhov_z+rhow_y-2*v*rho_z-2*w*rho_y)
        g42 = 0.0
        g43 = -inv_rho2*mu*rho_z
        g44 = -inv_rho2*mu*rho_y
        g51 = -inv_rho*(u*t_xy + v*t_yy + w*t_yz) + u*g21 + v*g31 + w*g41 \
                - k * (rhoe_y - (2*e-3*u*u-3*v*v-3*w*w)*rho_y \
                    - 2*u*rhou_y - 2*v*rhov_y - 2*w*rhow_y)
        g52 = u * g22 + v*g32 + w*g42 + inv_rho*t_xy \
                + k * (2*u*rho_y - rhou_y)
        g53 = u * g23 + v*g33 + w*g43 + inv_rho*t_yy \
                + k * (2*v*rho_y - rhov_y)
        g54 = u * g24 + v*g34 + w*g44 + inv_rho*t_yz \
                + k * (2*w*rho_y - rhow_y)
        g55 = -k * rho_y

        h21 = f41
        h22 = f42
        h23 = 0.0
        h24 = f44
        h31 = g41
        h32 = 0.0
        h33 = g43
        h34 = g44
        h41 = -inv_rho2*(lam*(rhou_x + rhov_y) \
                         + mur*(rhow_z + u*rho_x + v*rho_y - 2*w*rho_z))
        h42 = -inv_rho2*lam*rho_x
        h43 = -inv_rho2*lam*rho_y
        h44 = -inv_rho2*mur*rho_z
        h51 = -inv_rho*(u*t_xz + v*t_yz + w*t_zz) + u*g21 + v*g31 + w*g41 \
                - k * (rhoe_z - (2*e-3*u*u-3*v*v-3*w*w)*rho_z \
                    - 2*u*rhou_z - 2*v*rhov_z - 2*w*rhow_z)
        h52 = u * g22 + v*g32 + w*g42 + inv_rho*t_xz \
                + k * (2*u*rho_z - rhou_z)
        h53 = u * g23 + v*g33 + w*g43 + inv_rho*t_yz \
                + k * (2*v*rho_z - rhov_z)
        h54 = u * g24 + v*g34 + w*g44 + inv_rho*t_zz \
                + k * (2*w*rho_z - rhow_z)
        h55 = -k * rho_z

        A[1][0] -= f21*nx + g21*ny + h21*nz
        A[1][1] -= f22*nx + g22*ny + h22*nz
        A[1][2] -= f23*nx + g23*ny + h23*nz
        A[1][3] -= f24*nx + g24*ny + h24*nz
        A[2][0] -= f31*nx + g31*ny + h31*nz
        A[2][1] -= f32*nx + g32*ny + h32*nz
        A[2][2] -= f33*nx + g33*ny + h33*nz
        A[2][3] -= f34*nx + g34*ny + h34*nz
        A[3][0] -= f41*nx + g41*ny + h41*nz
        A[3][1] -= f42*nx + g42*ny + h42*nz
        A[3][2] -= f43*nx + g43*ny + h43*nz
        A[3][3] -= f44*nx + g44*ny + h44*nz
        A[4][0] -= f51*nx + g51*ny + h51*nz
        A[4][1] -= f52*nx + g52*ny + h52*nz
        A[4][2] -= f53*nx + g53*ny + h53*nz
        A[4][3] -= f54*nx + g54*ny + h54*nz
        A[4][4] -= f55*nx + g55*ny + h55*nz

    if ndims == 2:
        return be.compile(visjacobian2d)
    elif ndims == 3:
        return be.compile(visjacobian3d)
    