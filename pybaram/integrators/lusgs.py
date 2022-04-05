import numpy as np


def make_diff_flux(nvars, dnv, fluxf):
    def _diff_flux(u, du, f, df, nf):
        for i in range(nvars):
            du[i] += u[i]

        fluxf(u, nf, f)
        fluxf(du, nf, df)

        for i in range(dnv):
            df[i] -= f[i]

    return _diff_flux


def make_lusgs_update(ele):
    # dimensions
    nvars = ele.nvars

    def _update(i_begin, i_end, uptsb, rhsb):
        # Update
        for idx in range(i_begin, i_end):
            for kdx in range(nvars):
                uptsb[kdx, idx] += rhsb[kdx, idx]

    return _update


def make_lusgs_common(ele, _lambdaf, factor=1.0):
    # dimensions
    nvars, nface = ele.nvars, ele.nface

    # Vectors
    fnorm_vol, vec_fnorm = ele.mag_fnorm * ele.rcp_vol, ele.vec_fnorm
    dxc = np.linalg.norm(ele.dxc, axis=2)

    # index
    nei_ele = ele.nei_ele

    def _pre_lusgs(i_begin, i_end, uptsb, dt, diag, lambdaf, mu=None, mut=None):
        # Construct Matrix
        for idx in range(i_begin, i_end):
            diag[idx] = 1 / (dt[idx]*factor)

            for jdx in range(nface):
                dx = dxc[jdx, idx]
                nf = vec_fnorm[jdx, :, idx]

                # Wave speed at face
                u = uptsb[:, idx]

                lamf = _lambdaf(u, nf, dx, idx, mu, mut)

                neib = nei_ele[jdx, idx]
                if neib > 0:
                    u = uptsb[:, neib]

                    # Find maximum wave speed at face
                    lamf = max(_lambdaf(u, nf, dx, neib, mu, mut), lamf)

                # Diffusive margin of wave speed at face
                lamf *= 1.01

                lambdaf[jdx, idx] = lamf
                diag[idx] += 0.5*lamf*fnorm_vol[jdx, idx]

    return _pre_lusgs


def make_serial_lusgs(be, ele, nv, mapping, unmapping, _flux):
    # dimensions
    nvars, nface = ele.nvars, ele.nface
    dnv = nv[1] - nv[0]

    # Vectors
    fnorm_vol, vec_fnorm = ele.mag_fnorm * ele.rcp_vol, ele.vec_fnorm

    # index
    nei_ele = ele.nei_ele

    # Pre-compile functions
    _diff_flux = be.compile(make_diff_flux(nvars, dnv, _flux))

    def _lower_sweep(i_begin, i_end, uptsb, rhsb, dub, diag, dsrc, lambdaf):
        du = np.zeros(nvars)
        f = np.zeros(dnv)
        dfj = np.zeros(dnv)
        df = np.zeros(dnv)

        for _idx in range(i_begin, i_end):
            idx = mapping[_idx]

            for kdx in range(dnv):
                df[kdx] = 0.0

            for jdx in range(nface):
                nf = vec_fnorm[jdx, :, idx]

                neib = nei_ele[jdx, idx]
                if neib > -1 and unmapping[neib] < _idx:
                    u = uptsb[:, neib]
                    du[:] = 0
                    for kdx in range(nv[0], nv[1]):
                        du[kdx] = dub[kdx, neib]

                    _diff_flux(u, du, f, dfj, nf)

                    for kdx in range(dnv):
                        df[kdx] += (dfj[kdx] - lambdaf[jdx, idx]
                                    * dub[kdx+nv[0], neib])*fnorm_vol[jdx, idx]

            for kdx in range(dnv):
                dub[kdx+nv[0], idx] = (rhsb[kdx+nv[0], idx] -
                                       0.5*df[kdx])/(diag[idx] + dsrc[kdx+nv[0], idx])

    def _upper_sweep(i_begin, i_end, uptsb, rhsb, dub, diag, dsrc, lambdaf):
        du = np.zeros(nvars)
        f = np.zeros(dnv)
        dfj = np.zeros(dnv)
        df = np.zeros(nvars)

        # Upper sweep (backward)
        for _idx in range(i_end-1, i_begin-1, -1):
            idx = mapping[_idx]

            for kdx in range(dnv):
                df[kdx] = 0.0

            for jdx in range(nface):
                nf = vec_fnorm[jdx, :, idx]

                neib = nei_ele[jdx, idx]
                if neib > -1 and unmapping[neib] > _idx:
                    u = uptsb[:, neib]
                    du[:] = 0
                    for kdx in range(nv[0], nv[1]):
                        du[kdx] = rhsb[kdx, neib]

                    _diff_flux(u, du, f, dfj, nf)

                    for kdx in range(dnv):
                        df[kdx] += (dfj[kdx] - lambdaf[jdx, idx]
                                    * rhsb[kdx+nv[0], neib])*fnorm_vol[jdx, idx]

            for kdx in range(dnv):
                rhsb[kdx+nv[0], idx] = dub[kdx+nv[0], idx] - \
                    0.5*df[kdx]/(diag[idx] + dsrc[kdx+nv[0], idx])

    return _lower_sweep, _upper_sweep


def make_colored_lusgs(be, ele, icolor, lcolor, _flux):
    # dimensions
    nvars, nface = ele.nvars, ele.nface

    # Vectors
    fnorm_vol, vec_fnorm = ele.mag_fnorm * ele.rcp_vol, ele.vec_fnorm

    # Pre-compile functions
    _diff_flux = be.compile(make_diff_flux(nvars, _flux))

    # index
    nei_ele = ele.nei_ele

    def _lower_sweep(i_begin, i_end, uptsb, rhsb, dub, diag, dsrc, lambdaf):
        du = np.zeros(nvars)
        f = np.zeros(nvars)
        dfj = np.zeros(nvars)
        df = np.zeros(nvars)

        for _idx in range(i_begin, i_end):
            # Coloring 순서
            idx = icolor[_idx]
            curr_level = lcolor[idx]

            for kdx in range(nvars):
                df[kdx] = 0.0

            for jdx in range(nface):
                nf = vec_fnorm[jdx, :, idx]

                neib = nei_ele[jdx, idx]
                if neib > -1:
                    if lcolor[neib] < curr_level:
                    #if neib < idx:
                        u = uptsb[:, neib]
                        for kdx in range(nvars):
                            du[kdx] = dub[kdx, neib]

                        _diff_flux(u, du, f, dfj, nf)

                        for kdx in range(0, nvars):
                            df[kdx] += (dfj[kdx] - lambdaf[jdx, idx]
                                        * dub[kdx, neib])*fnorm_vol[jdx, idx]

            for kdx in range(0, nvars):
                dub[kdx, idx] = (rhsb[kdx, idx] -
                                       0.5*df[kdx])/(diag[idx] + dsrc[kdx, idx])

    def _upper_sweep(i_begin, i_end, uptsb, rhsb, dub, diag, dsrc, lambdaf):
        du = np.zeros(nvars)
        f = np.zeros(nvars)
        dfj = np.zeros(nvars)
        df = np.zeros(nvars)

        # Upper sweep (backward)
        #for _idx in range(i_end-1, i_begin-1, -1):
        for _idx in range(i_begin, i_end):
            idx = icolor[_idx]
            curr_level = lcolor[idx]

            for kdx in range(nvars):
                df[kdx] = 0.0

            for jdx in range(nface):
                nf = vec_fnorm[jdx, :, idx]

                neib = nei_ele[jdx, idx]
                if neib > -1:
                    if lcolor[neib] > curr_level:
                        u = uptsb[:, neib]
                        for kdx in range(nvars):
                            du[kdx] = rhsb[kdx, neib]

                        _diff_flux(u, du, f, dfj, nf)

                        for kdx in range(0, nvars):
                            df[kdx] += (dfj[kdx] - lambdaf[jdx, idx]
                                        * rhsb[kdx, neib])*fnorm_vol[jdx, idx]

            for kdx in range(0, nvars):
                rhsb[kdx, idx] = dub[kdx, idx] - \
                    0.5*df[kdx]/(diag[idx] + dsrc[kdx, idx])

    return _lower_sweep, _upper_sweep
