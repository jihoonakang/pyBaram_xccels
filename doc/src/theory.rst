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
Euler equation, the governing equation of inviscid flow, is written as follows.

.. math::
   U = \begin{bmatrix}
    \rho \\ \rho u \\ \rho v \\ \rho w \\ \rho e_t
   \end{bmatrix}

where, :math:`\rho` is density, :math:`u,v,w` are component of velocity vector and
:math:`e_t` is total specific internal energy. From equations of state, 
specific internal energy can be written as follows.

.. math::
    e &= \frac{p}{(\gamma -1) \rho} \\
    e_t &= e + \frac{1}{2} (u^2 + v^2 + w^2)

where :math:`p` is pressure and :math:`\gamma` is ratio of specific heats.

Euler equation has only convective flux, which can be written as follows.

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
To solve viscous flow, viscous flux is added to Euler equations.

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

where, :math:`T` is temperature and :math:`Pr` is Prandt number, which is a non-dimensionalized value.

.. math::
    Pr = \frac{\mu C_p}{k}

where, :math:`C_p` is specific heat at constant pressure and :math:`k` is thermal conductivity.

RANS Equations
---------------
For RANS (Reynolds-averaged Navier-Stokes) equations, turbulent viscosity is computed by turbulent model equation.
One equation `Spalart-Allmaras model <https://turbmodels.larc.nasa.gov/spalart.html#sa>`_ and
Two equation `SST model <https://turbmodels.larc.nasa.gov/spalart.html#sst>`_ models are employed in ``pyBaram``.
With tubulent viscosity :math:`\mu`, shear stress in viscous flux can be modified as follows.

.. math::
   \tau_{xx} &=  2\frac{\mu+\mu_t}{\rho}(u_x - \frac{1}{3}(u_x + v_y + w_z)) \\
   \tau_{xy} &= \frac{\mu+\mu_t}{\rho}(v_x + u_y)

Turbulent thermal conductivity is computed using turbulent Prandtl number :math:`Pr_t` and 
:math:`\Theta` in viscous flux can be modified as follows.

.. math::
   \Theta_x = u \tau_{xx} + v \tau_{xy} + w \tau_{xz} + \frac{\gamma\mu}{Pr + Pr_t} T_x

Finite Volume Method
=====================
To discretize in space, cell-centered finite volume method is employed. 
For each cell, the semi-discrete form of the gonverning equation can be written as follows.

.. math::
   \frac{\partial \bar{U}_i}{\partial t} = 
   -\frac{1}{\Delta V_i} \sum_{f} (H_c (U_f^+, U_f^-, \vec{n}_f) - H_v (\bar{U}_f, \nabla U_f, \vec{n}_f)) \Delta A_f + \bar{S}_i

where, :math:`\Delta V_i` is a volume of `i-th` cell and 
:math:`\Delta A_f, \vec{n}_f` are area and unit normal vector at the face :math:`f`, respectively.
:math:`H_c` is numerical convective flux, which can be solved by approximate Riemann solver.
:math:`U_f^+, U_f^-` are the left and right conservative variable vectors and they are computed using MUSCL-type reconstruction.
:math:`H_v` is the viscous flux and :math:`\bar{U}_m, \nabla U_m` are averaged conservative variable and its derivatives.

The right hand side term can be computed with following procedures.

Gradient Calculation
---------------------
The gradient of each cell is computed by least-square, green-gauss or its hybrid :cite:`shima_hybrid_gradient` and numerical formulation can be written as follows.

.. math::
   \nabla U = M \cdot 
   \begin{bmatrix}
    \Delta U_{f1} \\
    \Delta U_{f2} \\
    ...
   \end{bmatrix}

where :math:`M` is pre-computed operation matrix and :math:`\Delta U_{fi}` is difference of convervative vector at `i`-th face of the cell.
``pyBaram`` compute gradient in two step.

* Compute :math:`\Delta U_{fi}` at each ``Inters`` class in :mod:`pybaram.solvers.baseadvec.inters`
    * `make_delu` method generates loop.
    * `construct_kernels` method of each ``Inters`` generates kernels.

* Compute :math:`\nabla U` at ``BaseAdvecElements``  class in :mod:`pybaram.solvers.baseadvec.elements`.
    * Operation matrix :math:`M` is pre-compuated at `_prelsq` method of ``BaseElements`` class
    * `make_grad` method of the class generates loop.
    * `construct_kernels` method of the class generates kernels.

MUSCL-type reconstruction
--------------------------
Left and right states are computed via MUSCL-type reconstruction

.. math::
   U_f^+ = \bar{U}_i + \phi \nabla U_i \cdot x_{i,f}

where, :math:`x_{i,f}` is the position vector from cell center to the face center.
:math:`\phi`` is the slope limiter.

Three kernels in ``pyBaram`` implement the reconstruction.

* Search extreme value at vertex for MLP-u1/u2 limiter :cite:`Park2010,Park2012`
    * `make_extv` method of each `Vertex` class in :mod:`pybaram.solvers.baseadvec.vertex` generates the loop
    * `construct_kernels` method of the same `Vertex` class initiates kernels

* Compute MLP limiter :math:`\phi` at each ``BaseAdvecElements`` class in :mod:`pybaram.solvers.baseadvec.elements`
    * `make_mlp_u` method of the class generates loop
    * `construct_kernles` method of the class initiates kernels.

* Compute MUSCL-type reconstruction :math:`U_f` at each ``BaseAdvecElements`` class in :mod:`pybaram.solvers.baseadvec.elements`
    * `make_recon` method of the class generates loop
    * `construct_kernles` method of the class initiates kernels.

Convective Flux 
----------------
Each ``inters`` class in :mod:`pybaram.solvers.euler.inters` computes convective flux.

* `make_flux` method generates loop to compute convective flux along the interface.
* At `construct_kernels` method of the ``Inters`` class in :mod:`pybaram.solvers.baseadvec` generates kernels.
* :math:`\Delta A_f, \vec{n}_f` are pre-computed and stored as `_mag_snorm` and `_vec_snorm` at ``BaseInters`` class in :mod:`pybaram.solvers.base.inters`. 
* Various approximate Riemann solver :math:`H_c` are implemented in :mod:`pybaram.solvers.euler.rsolvers`. 

    * RoeM :cite:`Kim2003`
    * AUSMPW+ :cite:`Kim2001`
    * AUSM+up :cite:`Liou2006`
    * HLLEM :cite:`Einfeldt1991`
    * Rusanov :cite:`rusanov1962calculation`
*  `fpts` in each elements stores :math:`U_L, U_R` before execuation and saves :math:`H_c \Delta A_f` after execution.

Viscous Flux
-------------
Each ``inters`` class in :mod:`pybaram.solvers.navierstokes` compute viscous flux.

* `make_flux` method generates loop to compute viscous flux, as well as convective flux, along the interface.
* Averaged state and gradient vectors at face is computed.
* Viscous flux :math:`H_v` is implemented in :mod:`pybaram.solvers.navierstokes.visflux`

Turbulent models
----------------
One or Two equations of RANS turbulent models are also computed with similar procedure.
As well as the divergence of flux, Source terms are also computed.

* :mod:`pybaram.solvers.rans` module generates overall kernels to compute RANS equations
* :mod:`pybaram.solvers.ranssa` module generates kernels for Spalart-Allmaras RANS model :cite:`Spalart1994` 
* :mod:`pybaram.solvers.ranskwsst` module generates kernels for SST RANS model :cite:`Menter1994` 

Time Integrations
==================
After computing right hand side (negative gradient of flux), the solution can be updated by integration over time.
Currently Explicit Runge-Kutta schemes :cite:`Martinelli1988,Gottlieb1998` and Implicit LU-SGS schemes :cite:`Yoon1988` are implemented.
These schemes are implmeneted in :mod:`pybaram.integrators` module.

References
==========
.. bibliography:: references.bib