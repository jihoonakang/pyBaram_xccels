from pybaram.utils.nb import dot
import numpy as np


def make_jacobi_update(nv):
    # Update next time step solution
    def _update(i_begin, i_end, uptsb, dub):
        for idx in range(i_begin, i_end):
            for kdx in range(nv[0], nv[1]):
                uptsb[kdx, idx] += dub[kdx, idx]

    return _update


def make_jacobi_common(be, ele, nv, _jacobian, _dsrc=None, factor=1.0):
    # Number of faces
    nface = ele.nface

    # Number of variables
    dnv = nv[1] - nv[0]

    # Normal vectors at faces and displacement from cell center to neighbor cells
    fnorm_vol, vec_fnorm = ele.mag_fnorm * ele.rcp_vol, ele.vec_fnorm
    dxc = np.linalg.norm(ele.dxc, axis=2)

    # Gradient at cell
    grad = ele.grad

    # Temporal 2D array
    matrix = be.local_matrix()

    def _pre_jacobi(i_begin, i_end, uptsb, dt, diag, mu=None, mut=None):
        # Compute digonal matrix
        for idx in range(i_begin, i_end):
            diag[:, :, idx] = 0.0
            u = uptsb[:, idx]
            gf = grad[:, :, idx]
            ap = matrix(dnv*dnv, (dnv, dnv))

            # Computes diagonal matrix based on neighbor cells
            for jdx in range(nface):                
                dx = dxc[jdx, idx]
                nf = vec_fnorm[jdx, :, idx]

                _jacobian(u, nf, ap, gf, idx, mu, dx, mut)
                for row in range(dnv):
                    for col in range(dnv):
                        diag[row, col, idx] += ap[row][col]*fnorm_vol[jdx, idx]
            
            # Derivative of source term for turbulence model
            if _dsrc is not None:
                _dsrc(ap, idx, u)
                for row in range(dnv):
                    for col in range(dnv):
                        diag[row, col, idx] += ap[row][col]
            
            # Complete implicit operator
            for kdx in range(dnv):
                diag[kdx, kdx, idx] += 1/(dt[idx]*factor)
            
            # Compute inverse
            diag[:, :, idx] = np.linalg.inv(diag[:, :, idx])
    
    return _pre_jacobi


def make_jacobi_sweep(be, ele, nv, _jacobian):
    # Make local array
    array = be.local_array()
    matrix = be.local_matrix()

    # Get element attributes
    nface = ele.nface
    dnv = nv[1] - nv[0]

    # Get index array for neihboring cells
    nei_ele = ele.nei_ele

    # Normal vectors at faces
    fnorm_vol, vec_fnorm = ele.mag_fnorm * ele.rcp_vol, ele.vec_fnorm
    dxc = np.linalg.norm(ele.dxc, axis=2)

    # Gradient at cell
    grad = ele.grad

    def _jacobi_sweep(i_begin, i_end, uptsb, rhsb, dub, rod, mu=None, mut=None):
        # Compute R-(L+U)x
        for idx in range(i_begin, i_end):
            rhs = array(dnv)
            am = matrix(dnv*dnv, (dnv, dnv))

            # Initialize rhs array with RHS
            for k in range(dnv):
                rhs[k] = rhsb[k+nv[0], idx]

            # Computes Jacobian matrix based on neighbor cells
            for jdx in range(nface):
                neib = nei_ele[jdx, idx]
                dx = dxc[jdx, idx]
                
                if neib != idx:
                    nf = vec_fnorm[jdx, :, idx]
                    gf = grad[:, :, neib]
                    
                    _jacobian(uptsb[:, neib], nf, am, gf, neib, mu, dx, mut)
                    for k in range(dnv):
                        rhs[k] -= dot(am[k], dub[:, neib], dnv, 0, nv[0])*fnorm_vol[jdx, idx]

            # Allocates to each rod array
            for k in range(dnv):
                rod[k+nv[0], idx] = rhs[k]
        
    def _jacobi_compute(i_begin, i_end, dub, rod, diag, subres=None):
        # Compute Ax = b
        for idx in range(i_begin, i_end):
            rhs = array(dnv)

            # Reallocate rod element value to rhs array
            for k in range(dnv):
                rhs[k] = rod[k+nv[0], idx]
            
            # Inner-update dub array
            for kdx in range(dnv):
                dub[kdx+nv[0], idx] = dot(diag[kdx, :, idx], rhs, dnv)

            # Save rho error
            if subres is not None:
                subres[idx] = abs(dub[0, idx])

    return _jacobi_sweep, _jacobi_compute
