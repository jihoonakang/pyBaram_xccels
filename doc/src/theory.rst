*******
Theory
*******

Governing Equations
===================
``pyBaram`` can solve convection-diffusion equations, which is written as follows.

.. math::
   \frac{\partial U}{\partial t} + \nabla \cdot (F_c - F_v) = S.

where, :math:`U` are conservative variable vector.
:math:`F_c, F_v` are the convective and viscous flux, respectively.
:math:`S` is the source vector.

Euler Equations
----------------
The governing equations of inviscid flow are written as follows.

.. math::
   U = \begin{bmatrix}
    \rho \\ \rho u \\ \rho v \\ \rho w \\ \rho e_t
   \end{bmatrix}

where, :math:`\rho` is density, :math:`u,v,w` are components of velocity vector and
:math:`e_t` is total specific internal energy. From equation of state, 
specific internal energy can be written as follows.

.. math::
    e &= \frac{p}{(\gamma -1) \rho} \\
    e_t &= e + \frac{1}{2} (u^2 + v^2 + w^2)

where :math:`p` is pressure and :math:`\gamma` is ratio of specific heats.

Euler equations have only convective flux, which can be written as follows.

.. math::
   F_c = \begin{bmatrix}
    \rho u & \rho v & \rho w \\
    \rho u^2 + p & \rho u v & rho u w \\
    \rho u v & \rho v^2 + p & \rho v w \\
    \rho u w & \rho v w & \rho w^2 + p \\
    \rho u h_t & \rho v h_t & \rho w h_t
   \end{bmatrix}

where :math:`h_t` is total specific enthalpy, which can be defined as follows.

.. math::
   h &= e + \frac{p}{\rho} \\
   h_t &= h + \frac{1}{2} (u^2 + v^2 + w^2)

Navier-Stokes Equations
------------------------
Viscous flux of Navier-Stokes equations are written as follows.

.. math::
    F_v = \begin{bmatrix}
    0 & 0 & 0 \\
    \tau_{xx} & \tau_{xy} & \tau_{xz} \\
    \tau_{yx} & \tau_{yy} & \tau_{yz} \\
    \tau_{zx} & \tau_{zy} & \tau_{zz} \\
    \Theta_x & \Theta_y & \Theta_z
    \end{bmatrix}

where, :math:`\tau` is shear stress, which can be written as follows.

.. math::
   \tau_{xx} &=  2\frac{\mu}{\rho}(u_x - \frac{1}{3}(u_x + v_y + w_z)) \\
   \tau_{xy} &= \frac{\mu}{\rho}(v_x + u_y)

:math:`\mu`` is viscosity and :math:`u_x` is derivative of velocity. :math:`\Theta` can be written as follows.

.. math::
   \Theta_x = u \tau_{xx} + v \tau_{xy} + w \tau_{xz} + \frac{\gamma\mu}{Pr} T_x

where, :math:`T` is temperature and :math:`Pr` is Prandtl number, which is a non-dimensionalized value.

.. math::
    Pr = \frac{\mu C_p}{k}

where, :math:`C_p` is specific heat at constant pressure and :math:`k` is thermal conductivity.

RANS Equations
---------------
For RANS (Reynolds-averaged Navier-Stokes) equations, turbulent viscosity is computed by turbulent model equation.
One equation `Spalart-Allmaras model <https://turbmodels.larc.nasa.gov/spalart.html#sa>`_ and
Two equation `SST model <https://turbmodels.larc.nasa.gov/spalart.html#sst>`_ models are employed in ``pyBaram``.
With turbulent viscosity :math:`\mu_t`, shear stress in viscous flux can be modified as follows.

.. math::
   \tau_{xx} &=  2\frac{\mu+\mu_t}{\rho}(u_x - \frac{1}{3}(u_x + v_y + w_z)) \\
   \tau_{xy} &= \frac{\mu+\mu_t}{\rho}(v_x + u_y)

Turbulent thermal conductivity is computed using turbulent Prandtl number :math:`Pr_t`, thus 
:math:`\Theta` in viscous flux can be modified as follows.

.. math::
   \Theta_x = u \tau_{xx} + v \tau_{xy} + w \tau_{xz} + \gamma \left(\frac{\mu}{Pr} + \frac{\mu_t}{Pr_t} \right) T_x

Finite Volume Method
=====================
Cell-centered finite volume method is employed to discretize in space. 
For each cell, the semi-discrete form of the governing equation can be written as follows.

.. math::
   \frac{\partial \bar{U}_i}{\partial t} = 
   -\frac{1}{\Delta V_i} \sum_{f} (H_c (U_f^+, U_f^-, \vec{n}_f) - H_v (\bar{U}_f, \nabla U_f, \vec{n}_f)) \Delta A_f + \bar{S}_i

where 
:math:`\bar{U}_i` and :math:`\bar{S}_i` represent the cell-averaged state variable vector
and source term vector at the :math:`i-th` cell, respectively. 
:math:`H_c`` and :math:`H_v` denote numerical inviscid and viscous fluxes, respectively. 
:math:`\bar{U}_f`  and :math:`\nabla U_f` correspond to the face-averaged state and 
gradient vectors at the :math:`f-th` face, respectively. 
Furthermore, :math:`n_f` and :math:`\Delta A_f` denote the unit normal vector and area 
of the :math:`f-th` face, respectively. :math:`\Delta V_i` is the volume of the :math:`i-th` cell. 
:math:`U_f^+` and :math:`U_f^-` are the left and right state vectors at the :math:`f-th` face;
they can be obtained by MUSCL-type reconstruction, as below

.. math::
   U_f^+ = \bar{U}_i + \phi_i \nabla U_i \cdot x_{i,f},

where :math:`\nabla U_i` corresponds to the gradient of the state variables at the :math:`i-th` cell
and :math:`x_{i,f}` denotes the position vector from cell center to face. 
Furthermore, :math:`\phi_i` is slope limiter at `i-th` cell for robustly capturing shock discontinuities; 
:math:`U_f^-` can be computed similarly at the adjacent cell

The procedures to compute the right-hand side can be summarized as follows:

Gradient Calculation
---------------------
The gradient of each cell is computed by least-square, green-gauss or 
its hybrid :cite:`shima_hybrid_gradient` and numerical formulation can be written as follows.

.. math::
   \nabla U = M \cdot 
   \begin{bmatrix}
    \Delta U_{f1} \\
    \Delta U_{f2} \\
    ...
   \end{bmatrix}

where :math:`M` is pre-computed operation matrix and :math:`\Delta U_{fi}` is difference of 
conservative vector at `f`-th face of the cell.
``pyBaram`` computes gradient with two steps.

* Compute :math:`\Delta U_{fi}` at each ``Inters`` class in :mod:`pybaram.solvers.baseadvec.inters`
    * `make_delu` method generates loop.
    * `construct_kernels` method of each ``Inters`` generates kernels.

* Compute :math:`\nabla U` at ``BaseAdvecElements``  class in :mod:`pybaram.solvers.baseadvec.elements`.
    * Operation matrix :math:`M` is pre-computed at `_prelsq` method of ``BaseElements`` class
    * `make_grad` method of the class generates loop.
    * `construct_kernels` method of the class generates kernels.

Slope Limiter
-------------
In order to capture shock-wave robustly, the slope of linear reconstruction should be limited.
``pyBaram`` computes MLP-u slope limiter with two steps.

* Search extreme value at vertex on MLP stencil :cite:`Park2010,Park2012`
    * `make_extv` method of each `Vertex` class in :mod:`pybaram.solvers.baseadvec.vertex` generates the loop
    * `construct_kernels` method of the same `Vertex` class initiates kernels

* Compute MLP-u1/u2 limiter :cite:`Park2010,Park2012` :math:`\phi` at each ``BaseAdvecElements`` class in :mod:`pybaram.solvers.baseadvec.elements`
    * `make_mlp_u` method of the class generates loop
    * `construct_kernles` method of the class initiates kernels.


MUSCL-type reconstruction
--------------------------
With gradient and slope limiter on each cell, the :math:`U_f^+` and :math:`U_f^-` is reconstructed linearly.

* Compute MUSCL-type reconstruction :math:`U_f` at each ``BaseAdvecElements`` class in :mod:`pybaram.solvers.baseadvec.elements`
    * `make_recon` method of the class generates loop
    * `construct_kernles` method of the class initiates kernels.

Convective Flux 
----------------
Each ``Inters`` class in :mod:`pybaram.solvers.euler.inters` computes convective flux.

* `make_flux` method generates loop to compute convective flux along the interface.
* At `construct_kernels` method of the ``Inters`` class in :mod:`pybaram.solvers.baseadvec` generates kernels.
* :math:`\Delta A_f, \vec{n}_f` are pre-computed and stored as `_mag_snorm` and `_vec_snorm` at ``BaseInters`` class in :mod:`pybaram.solvers.base.inters`. 
* Various approximate Riemann solver :math:`H_c` are implemented in :mod:`pybaram.solvers.euler.rsolvers`. 

    * RoeM :cite:`Kim2003`
    * AUSMPW+ :cite:`Kim2001`
    * AUSM+up :cite:`Liou2006`
    * HLLEM :cite:`Einfeldt1991`
    * Rusanov :cite:`rusanov1962calculation`
*  `fpts` in each element stores :math:`U_L, U_R` before execution and saves :math:`H_c \Delta A_f` after execution.

Viscous Flux
-------------
Each ``Inters`` class in :mod:`pybaram.solvers.navierstokes` computes viscous flux.

* `make_flux` method generates loop to compute viscous flux, as well as convective flux, along the interface.
* Averaged state and gradient vectors at face are computed.
* Viscous flux :math:`H_v` is implemented in :mod:`pybaram.solvers.navierstokes.visflux`

Negative Divergence of Fluxes
-----------------------------
After computing flux at faces, divergence of flux can be computed with finite volume method.

* Compute :math:`-\frac{1}{\Delta V_i} \sum_{f} H \Delta A_f` at ``BaseAdvecElements`` class in :mod:`pybaram.solvers.baseadvec.elements`.
    * `_make_div_upts` method of the class generates loop.
    * `construct_kernels` method of the class generates kernels.

Turbulence Models
=================
One or Two equations of RANS turbulence models are also computed with similar procedure.
Source terms are added after divergence of flux.

* :mod:`pybaram.solvers.rans` module generates overall kernels to compute RANS equations
* :mod:`pybaram.solvers.ranssa` module generates kernels for Spalart-Allmaras RANS model :cite:`Spalart1994` 
* :mod:`pybaram.solvers.ranskwsst` module generates kernels for SST RANS model :cite:`Menter1994` 

Time Integrations
==================
After computing right hand side (negative gradient of flux), the solution can be updated by integration over time.
Currently, explicit Runge-Kutta schemes :cite:`Martinelli1988,Gottlieb1998` and 
implicit LU-SGS schemes :cite:`Yoon1988` are implemented.
The classes for these integrators are provided in :mod:`pybaram.integrators` module.

References
==========
.. bibliography:: references.bib