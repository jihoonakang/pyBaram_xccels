import numpy as np


def make_diff_flux(nvars, fluxf):
    def _diff_flux(u, du, f, df, nf):
        for i in range(nvars):
            du[i] += u[i]

        fluxf(u, nf, f)
        fluxf(du, nf, df)

        for i in range(nvars):
            df[i] -= f[i]

    return _diff_flux


def make_lusgs(be, ele, icolor, _flux, _lambdaf, factor=1.0):
    # dimensions
    nvars, ndims, nface, neles = ele.nvars, ele.ndims, ele.nface, ele.neles
    
    # Vectors
    fnorm_vol, vec_fnorm = ele.mag_fnorm * ele.rcp_vol, ele.vec_fnorm
    dxc = np.linalg.norm(ele.dxc, axis=2)

    # index
    nei_ele = ele.nei_ele

    # Pre-compile functions
    _diff_flux = be.compile(make_diff_flux(nvars, _flux))

    def _pre_lusgs(i_begin, i_end, uptsb, dt, diag, lambdaf):
        u = np.zeros(nvars)
        nf = np.empty(ndims)

        # Construct Matrix
        for idx in range(i_begin, i_end):
            diag[idx] = 1 / (dt[idx]*factor)

            for jdx in range(nface):
                dx = dxc[jdx, idx]
                for kdx in range(ndims):
                    nf[kdx] = vec_fnorm[jdx, kdx, idx]

                # Wave speed at face
                for kdx in range(nvars):
                    u[kdx] = uptsb[kdx, idx]

                lamf = _lambdaf(u, nf, dx)

                neib = nei_ele[jdx, idx]
                if neib > 0:
                    for kdx in range(nvars):
                        u[kdx] = uptsb[kdx, neib]

                    # Find maximum wave speed at face
                    lamf = max(_lambdaf(u, nf, dx), lamf)

                # Diffusive margin of wave speed at face
                lamf *= 1.01

                lambdaf[jdx, idx] = lamf
                diag[idx] += 0.5*lamf*fnorm_vol[jdx, idx]

    def _lower_sweep(i_begin, i_end, marked, uptsb, rhsb, dub, diag, lambdaf):
        u = np.zeros(nvars)
        du = np.zeros(nvars)
        f = np.zeros(nvars)
        dfj = np.zeros(nvars)
        nf = np.empty(ndims)
        df = np.zeros(nvars)

        for _idx in range(i_begin, i_end):
            # Coloring 순서
            idx = icolor[_idx]
            marked[idx] = 1

            for kdx in range(nvars):
                df[kdx] = 0.0

            for jdx in range(nface):
                for kdx in range(ndims):
                    nf[kdx] = vec_fnorm[jdx, kdx, idx]

                neib = nei_ele[jdx, idx]
                if neib > -1:
                    if marked[neib] == 1:
                        for kdx in range(nvars):
                            u[kdx] = uptsb[kdx, neib]
                            du[kdx] = 0

                        for kdx in range(0, nvars):
                            du[kdx] = dub[kdx, neib]

                        _diff_flux(u, du, f, dfj, nf)

                        for kdx in range(0, nvars):
                            df[kdx] += (dfj[kdx] - lambdaf[jdx, idx]
                                        * dub[kdx, neib])*fnorm_vol[jdx, idx]

            for kdx in range(0, nvars):
                dub[kdx, idx] = (rhsb[kdx, idx] -
                                       0.5*df[kdx])/diag[idx]

    def _upper_sweep(i_begin, i_end, marked, uptsb, rhsb, dub, diag, lambdaf):
        u = np.zeros(nvars)
        du = np.zeros(nvars)
        f = np.zeros(nvars)
        dfj = np.zeros(nvars)
        nf = np.empty(ndims)
        df = np.zeros(nvars)

        # Upper sweep (backward)
        for _idx in range(i_end-1, i_begin-1, -1):
            idx = icolor[_idx]
            marked[idx] = -1

            for kdx in range(nvars):
                df[kdx] = 0.0

            for jdx in range(nface):
                for kdx in range(ndims):
                    nf[kdx] = vec_fnorm[jdx, kdx, idx]

                neib = nei_ele[jdx, idx]
                if neib > -1:
                    if marked[neib] == -1:
                        for kdx in range(nvars):
                            u[kdx] = uptsb[kdx, neib]
                            du[kdx] = 0

                        for kdx in range(0, nvars):
                            du[kdx] = rhsb[kdx, neib]

                        _diff_flux(u, du, f, dfj, nf)

                        for kdx in range(0, nvars):
                            df[kdx] += (dfj[kdx] - lambdaf[jdx, idx]
                                        * rhsb[kdx, neib])*fnorm_vol[jdx, idx]

            for kdx in range(0, nvars):
                rhsb[kdx, idx] = dub[kdx, idx] - \
                    0.5*df[kdx]/diag[idx]

    def _update(i_begin, i_end, uptsb, rhsb):
        # Update
        for idx in range(i_begin, i_end):
            for kdx in range(nvars):
                uptsb[kdx, idx] += rhsb[kdx, idx]

    return _pre_lusgs, _lower_sweep, _upper_sweep, _update

