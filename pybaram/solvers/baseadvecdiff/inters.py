# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvec import BaseAdvecIntInters, BaseAdvecBCInters, BaseAdvecMPIInters
from pybaram.backends.types import Kernel
from pybaram.utils.nb import dot

import numpy as np


class BaseAdvecDiffIntInters(BaseAdvecIntInters):
    def construct_kernels(self, elemap):
        # View of elemenet array
        fpts = [cell.fpts for cell in elemap.values()]
        dfpts = [cell.grad for cell in elemap.values()]
        nele = len(fpts)

        # Gradient at face
        gradf = np.empty((self.ndims, self.nvars, self.nfpts))

        self.compute_delu = Kernel(self._make_delu(), *fpts)

        self.compute_grad_at = Kernel(
            self._make_grad_at(nele), gradf, *fpts, *dfpts
        )

        self.compute_flux = Kernel(self._make_flux(), gradf, *fpts)

    def _make_grad_at(self, nele):
        nvars, ndims = self.nvars, self.ndims
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx

        # Mangitude and direction of the connecting vector
        inv_tf = 1/np.linalg.norm(self._dx_adj, axis=0)
        tf = self._dx_adj * inv_tf
        avec = self._vec_snorm/np.einsum('ij,ij->j', tf, self._vec_snorm)

        def grad_at(i_begin, i_end, gradf, *uf):
            # Parse element views (fpts, grad)
            du = uf[:nele]
            gradu = uf[nele:]
            gf = np.empty(ndims)

            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]

                tfi = tf[:, idx]
                inv_tfi = inv_tf[idx]
                aveci = avec[:, idx]

                # Compute the average of gradient at face
                for jdx in range(nvars):
                    gfl = gradu[lti][:, jdx, lei]
                    gfr = gradu[rti][:, jdx, rei]
                    for kdx in range(ndims):
                        gf[kdx] = 0.5*(gfl[kdx] + gfr[kdx])

                    gft = dot(gf, tfi, ndims)

                    # Compute gradient with jump term
                    for kdx in range(ndims):
                        gf[kdx] -= (gft - du[lti][lfi, jdx, lei]
                                    * inv_tfi)*aveci[kdx]
                        gradf[kdx, jdx, idx] = gf[kdx]

        return self.be.make_loop(self.nfpts, grad_at)


class BaseAdvecDiffMPIInters(BaseAdvecMPIInters):
    def construct_kernels(self, elemap):
        # Buffer array
        lhs = np.empty((self.nvars, self.nfpts))
        rhs = np.empty((self.nvars, self.nfpts))

        # Gradient at face and buffer
        gradf = np.empty((self.ndims, self.nvars, self.nfpts))
        grad_rhs = np.empty((self.ndims, self.nvars, self.nfpts))

        # View of element array
        fpts = [cell.fpts for cell in elemap.values()]
        dfpts = [cell.grad for cell in elemap.values()]
        nele = len(fpts)

        self.compute_delu = Kernel(self._make_delu(), rhs, *fpts)

        self.compute_grad_at = Kernel(
            self._make_grad_at(), gradf, grad_rhs, *fpts
        )

        self.compute_flux = Kernel(self._make_flux(), gradf, rhs, *fpts)

        self.pack = Kernel(self._make_pack(), lhs, *fpts)
        self.send, self.sreq = self._make_send(lhs)
        self.recv, self.rreq = self._make_recv(rhs)

        self.pack_grad = Kernel(self._make_pack_grad(), gradf, *dfpts)
        self.send_grad, self.sgreq = self._make_send(gradf)
        self.recv_grad, self.rgreq = self._make_recv(grad_rhs)

    def _make_grad_at(self):
        nvars, ndims = self.nvars, self.ndims
        lt, le, lf = self._lidx

        # Mangitude and direction of the connecting vector
        inv_tf = 1/np.linalg.norm(self._dx_adj, axis=0)
        tf = self._dx_adj * inv_tf
        avec = self._vec_snorm/np.einsum('ij,ij->j', tf, self._vec_snorm)

        def grad_at(i_begin, i_end, gradf, grad_rhs, *du):
            gf = np.empty(ndims)

            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                tfi = tf[:, idx]
                inv_tfi = inv_tf[idx]
                aveci = avec[:, idx]

                # Compute the average of gradient at face
                for jdx in range(nvars):
                    for kdx in range(ndims):
                        gf[kdx] = 0.5*(gradf[kdx, jdx, idx] +
                                       grad_rhs[kdx, jdx, idx])

                    gft = dot(gf, tfi, ndims)

                    # Compute gradient with jump term
                    for kdx in range(ndims):
                        gf[kdx] -= (gft - du[lti][lfi, jdx, lei]
                                    * inv_tfi)*aveci[kdx]

                        gradf[kdx, jdx, idx] = gf[kdx]

        return self.be.make_loop(self.nfpts, grad_at)

    def _make_pack_grad(self):
        ndims, nvars = self.ndims, self.nvars
        lt, le, _ = self._lidx

        def pack(i_begin, i_end, lhs, *uf):
            for idx in range(i_begin, i_end):
                lti, lei = lt[idx], le[idx]

                for jdx in range(nvars):
                    for kdx in range(ndims):
                        lhs[kdx, jdx, idx] = uf[lti][kdx, jdx, lei]

        return self.be.make_loop(self.nfpts, pack)


class BaseAdvecDiffBCInters(BaseAdvecBCInters):
    def construct_kernels(self, elemap):
        self.construct_bc()

        # View of elemenet array
        fpts = [cell.fpts for cell in elemap.values()]
        dfpts = [cell.grad for cell in elemap.values()]
        nele = len(fpts)

        # Gradient at face
        gradf = np.empty((self.ndims, self.nvars, self.nfpts))

        self.compute_delu = Kernel(self._make_delu(), *fpts)

        self.compute_grad_at = Kernel(
            self._make_grad_at(nele), gradf, *fpts, *dfpts
        )

        self.compute_flux = Kernel(self._make_flux(), gradf, *fpts)

    def _make_grad_at(self, nele):
        nvars, ndims = self.nvars, self.ndims
        lt, le, lf = self._lidx

        # Mangitude and direction of the connecting vector
        inv_tf = 1/np.linalg.norm(self._dx_adj, axis=0)
        tf = self._dx_adj * inv_tf
        avec = self._vec_snorm/np.einsum('ij,ij->j', tf, self._vec_snorm)

        def grad_at(i_begin, i_end, gradf, *uf):
            # Parse element views (fpts, grad)
            du = uf[:nele]
            gradu = uf[nele:]
            gf = np.empty(ndims)

            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                tfi = tf[:, idx]
                inv_tfi = inv_tf[idx]
                aveci = avec[:, idx]

                # Compute the average of gradient at face
                for jdx in range(nvars):
                    for kdx in range(ndims):
                        gf[kdx] = gradu[lti][kdx, jdx, lei]

                    gft = dot(gf, tfi, ndims)

                    # Compute gradient with jump term
                    for kdx in range(ndims):
                        gf[kdx] -= (gft - du[lti][lfi, jdx, lei]
                                    * inv_tfi)*aveci[kdx]
                        gradf[kdx, jdx, idx] = gf[kdx]

        return self.be.make_loop(self.nfpts, grad_at)
