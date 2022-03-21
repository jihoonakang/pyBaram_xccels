# -*- coding: utf-8 -*-
from mpi4py import MPI

import numpy as np

from pybaram.plugins.base import BasePlugin, csv_write
from pybaram.utils.np import npeval


class ForcePlugin(BasePlugin):
    name = 'force'

    def __init__(self, intg, cfg, suffix):
        self.cfg = cfg
        sect = 'soln-plugin-{}-{}'.format(self.name, suffix)

        #  MPI
        self._comm = comm = MPI.COMM_WORLD
        self._rank = rank = comm.rank

        # Get dynamic pressure
        const = cfg.items('constants')
        rho = npeval(cfg.get(sect, 'rho', 1.0), const)
        vel = npeval(cfg.get(sect, 'vel', 1.0), const)
        area = npeval(cfg.get(sect, 'area', 1.0), const)
        self._rcp_dynp = 1.0/(0.5*rho*vel**2*area)

        # Get vector
        self.ndims = ndims = intg.sys.ndims
        self.dname = dname = cfg.get(sect, 'dir-name', 'xyz'[:ndims])
        dvec = np.eye(ndims)
        self.dvec = np.array([npeval(cfg.get(sect, 'dir-{}'.format(d), dvec[i]), const)
                              for i, d in enumerate(dname)])

        # bcmap
        bcmap = {bc.bctype: bc for bc in intg.sys.bint}

        # Get idx, norm
        self._bcinfo = bcinfo = {}

        try:
            bc = bcmap[suffix]
            t, e, _ = bc._lidx
            mag, vec = bc._mag_snorm, bc._vec_snorm

            for i in np.unique(t):
                mask = (t == i)
                eidx = e[mask]
                norm = vec[:, mask]*mag[mask]

                if intg.sys.name in ['euler']:
                    bcinfo[i] = (eidx, norm)
                    self.viscous = False
                else:
                    dxn = np.linalg.norm(bc._dx_adj[:, mask], axis=0)/2
                    bcinfo[i] = (eidx, norm, dxn)
                    self.viscous = True
        except:
            self.viscous = False

        # Get integration mode
        self.mode = intg.mode
        if self.mode == 'steady':
            self.itout = cfg.getint(sect, 'iter-out', 100)
            lead = ['iter']
        else:
            self.dtout = cfg.getfloat(sect, 'dt-out')
            self.tout_next = intg.tcurr
            intg.add_tlist(self.dtout)
            lead = ['t']

        # Out file name and header
        if rank == 0:
            fname = "force_{}.csv".format(suffix)
            header = lead + ['c{}'.format(x) for x in dname]
            self.outf = csv_write(fname, header)

    def __call__(self, intg):
        if self.mode == 'steady':
            if not intg.isconv and intg.iter % self.itout:
                return
            txt = [intg.iter]

        else:
            if abs(intg.tcurr - self.tout_next) > 1e-6:
                return
            txt = [intg.tcurr]

        # eles, solns를 list로 변환
        eles = list(intg.sys.eles)
        solns = list(intg.curr_soln)

        # Force 계산
        force = []
        if not self.viscous:
            for i, (eidx, norm) in self._bcinfo.items():
                soln = solns[i]
                p = eles[i].conv_to_prim(soln[:, eidx], self.cfg)[1]
                force.append(np.sum(p*norm, axis=1))
        else:
            for i, (eidx, norm, dxn) in self._bcinfo.items():
                soln = solns[i]
                p = eles[i].conv_to_prim(soln[:, eidx], self.cfg)[1]
                force.append(np.sum(p*norm, axis=1))

        # 계수 계산
        if force:
            cf = np.dot(self.dvec, np.sum(force, axis=0))*self._rcp_dynp
        else:
            cf = np.zeros(len(self.dname))

        if self._rank != 0:
            self._comm.Reduce(cf, None, op=MPI.SUM, root=0)
        else:
            self._comm.Reduce(MPI.IN_PLACE, cf, op=MPI.SUM, root=0)

        if self._rank == 0:
            # Write
            row = txt + cf.tolist()
            print(','.join(str(r) for r in row), file=self.outf)

            # Flush to disk
            self.outf.flush()
