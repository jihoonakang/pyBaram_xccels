# -*- coding: utf-8 -*-
from pybaram.plugins import get_plugin
from pybaram.solvers import get_system
from pybaram.utils.misc import ProxyList


class BaseIntegrator:
    def __init__(self, be, cfg, msh, soln, comm):
        self.be = be
        
        # Get equation system
        self.sys = get_system(be, cfg, msh, soln, comm, self.nreg)

        self.mesh = msh
        self._curr_idx = 0

        # Check aux array (turbulence variables or others for post processing)
        try:
            self.curr_aux
            self.is_aux=True
        except:
            self.is_aux=False

        # Construct Plugins
        self.completed_handler = plugins = ProxyList()
        for sect in cfg.sections():
            if sect.startswith('soln-plugin'):
                name = sect.split('-')[2:]

                # Check plugin has suffix
                if len(name) > 1:
                    name, suffix = name
                else:
                    name, suffix = name[0], None

                plugins.append(get_plugin(name, self, cfg, suffix))

    @property
    def curr_soln(self):
        return self.sys.eles.upts[self._curr_idx]

    @property
    def curr_aux(self):
        return self.sys.eles.aux

    @property
    def curr_mu(self):
        mu = self.sys.eles.mu

        if hasattr(self.sys.eles, 'mut'):
            mu = ProxyList([m1 + m2 for m1, m2 in zip(mu, self.sys.eles.mut)])
        
        return mu