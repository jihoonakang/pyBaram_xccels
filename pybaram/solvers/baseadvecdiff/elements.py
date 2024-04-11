# -*- coding: utf-8 -*-
from pybaram.solvers.navierstokes.jacobian import get_viscous_jacobian
from pybaram.solvers.baseadvec import BaseAdvecElements


class BaseAdvecDiffElements(BaseAdvecElements):
    
    def make_jacobian(self, sign, vistype):
        cplargs = {
            'ndims': self.ndims,
            'gamma': self._const['gamma'],
            'pr': self._const['pr'],
            'to_prim': self.to_flow_primevars()
        }

        conv_func = super().make_jacobian(sign)
        vis_func = get_viscous_jacobian(vistype, self.be, cplargs)

        def _jacobian(u, nf, A, gf, idx, mu, *args):
            conv_func(u, nf, A)
            vis_func(u, nf, A, gf, idx, mu)

        return self.be.compile(_jacobian)
