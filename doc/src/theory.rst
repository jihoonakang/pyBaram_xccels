*******
Theory
*******

Governing Equations
===================
``pyBaram`` can solve convection-diffusion equations, which can be written as follows.

.. math::
   \frac{\partial U}{\partial t} + \nabla \cdot (F_c - F_v) = 0.

where, :math:`U` are conservative variable vector.
:math:`F_c, F_v` are the convective and viscous flux, respectively.

Euler Equations
----------------
Euler equation, the governing equation of inviscid flow, can be written as follows.

.. math::
   U = \begin{bmatrix}
    \rho \\ \rho u \\ \rho v \\ \rho w \\ \rho e_t
   \end{bmatrix}

where, :math:`\rho` is density, :math:`\vec{V} = (u,v,w)` are component of velocity vector and
:math:`e_t` is total internal energy. From constitutive relation, internal energy can be written as follows.

.. math::
    e &= \frac{p}{(\gamma -1) \rho} \\
    e_t &= e + \frac{1}{2} (u^2 + v^2 + w^2)

where :math:`p` is pressure and :math:`\gamma` is ratio of specific heats.

Euler equation has only convective flux, which can be written as follows.

.. math::
   F_c = \begin{bmatrix}
    \rho \vec{V} \cdot \vec{n} \\
    \rho u \vec{V} \cdot \vec{n} + n_x p\\
    \rho v \vec{V} \cdot \vec{n} + n_y p\\
    \rho w \vec{V} \cdot \vec{n} + n_z p\\
    \rho h_t V_{contra}
   \end{bmatrix}

where :math:`\vec{n}=(n_x, n_y, n_z)` is normal vector and :math:`h_t` is total enthalpy, which can be defined as follows.

.. math::
   h &= e + \frac{p}{\rho} \\
   h_t &= h + \frac{1}{2} (u^2 + v^2 + w^2)

Navier-Stokes Equations
------------------------
To solve viscous flow, viscous flux is added to Euler equations.

.. math::
    F_v = \begin{bmatrix}
    0 \\
    n_x \tau_{xx} + n_y \tau_{xy} + n_z \tau_{xz} \\
    n_x \tau_{yx} + n_y \tau_{yy} + n_z \tau_{yz} \\
    n_x \tau_{zx} + n_y \tau_{zy} + n_z \tau_{zz} \\
    n_x \Theta_x + n_y \Theta_y + n_z \Theta_z
    \end{bmatrix}

where, :math:`\tau` is shear stress, which can be written as follows.

.. math::
   \tau_{xx} &=  2\frac{\mu}{\rho}(u_x - \frac{1}{3}(u_x + v_y + w_z)) \\
   \tau_{xy} &= \frac{\mu}{\rho}(v_x + u_y)

:math:`\mu`` is viscosity and :math:`u_x` is derivative of velocity. :math:`Theta` can be written as follows.

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
:math:`Theta` in viscous flux can be modified as follows.

.. math::
   \Theta_x = u \tau_{xx} + v \tau_{xy} + w \tau_{xz} + \frac{\gamma\mu}{Pr + Pr_t} T_x

Finite Volume Method
=====================
To discretize in space, cell-centered finite volume method is employed. 
For each cell, the semi-discrete form of the gonverning equation can be written as follows.

.. math::
   \frac{\partial U}{\partial t} = 
   -\frac{1}{Vol} \sum_{f} (H_c (U_L, U_R, \vec{n}_f) - H_v (U_m, \nabla U_m, \vec{n}_f)) S_f.

where, :math:`vol` is a volume of cell and 
:math:`S_f, \vec{n}_f` are area and unit normal vector at the face :math:`f`, respectively.
:math:`H_c` is numerical convective flux, which can be solved by approximate Riemann solver.
:math:`U_L, U_R` are the left and right conservative variable vectors and they are computed using MUSCL-type reconstruction.
:math:`H_v` is the viscous flux and :math:`U_m, \nabla U_m` are averaged conservative variable and its derivatives.

Gradient Calculation
---------------------
The gradient of each cell is computed by least-square, green-gauss or its hybrid and numerical formulation can be written as follows.

.. math::
   \nabla U = M \cdot 
   \begin{bmatrix}
    \Delta U_{f1} \\
    \Delta U_{f2} \\
    ...
   \end{bmatrix}

where :math:`M` is pre-computed operation matrix and :math:`\Delta U_{fi}` is difference of convervative vector at `i`-th face of the cell.
``pyBaram`` compute gradient in two step.

* Compute :math:`\Delta U_{fi}` at each ``Inters`` class in `pybaram.solvers.baseadvec.inters`
    * `make_delu` method generates loop.
    * `construct_kernels` method of each ``Inters`` generates kernels.

* Compute :math:`\nabla U` at ``BaseAdvecElements``  class in `pybaram.solvers.baseadvec.elements`.
    * Operation matrix :math:`M` is pre-compuated at `_prelsq` method of ``BaseElements`` class
    * `make_grad` method of the class generates loop.
    * `construct_kernels` method of the class generates kernels.

MUSCL-type reconstruction
--------------------------
Left and right states are computed via MUSCL-type reconstruction

.. math::
   U_f = U + \phi \nabla U \cdot \Delta x_f

where, :math:`x_f` is the position vector from cell center to the face center.
:math:`\phi`` is the slope limiter.

Three kernels in ``pyBaram`` implement the reconstruction.

* Search extreme value at vertex for MLP-u1/u2 limiter
    * `make_extv` method of each `Vertex` class in `pybaram.solvers.baseadvec.vertex` generates the loop
    * `construct_kernels` method of the same `Vertex ` class initiates kernels

* Compute MLP limiter :math:`\phi` at each ``BaseAdvecElements`` class in `pybaram.solvers.baseadvec.elements`
    * `make_mlp_u` method of the class generates loop
    * `construct_kernles` method of the class initiates kernels.

* Compute MUSCL-type reconstruction :math:`U_f` at each ``BaseAdvecElements`` class in `pybaram.solvers.baseadvec.elements`
    * `make_recon` method of the class generates loop
    * `construct_kernles` method of the class initiates kernels.

Convective Flux 
----------------
Each ``inters`` class in `pybaram.solvers.euler.inters` compute convective flux.

* `make_flux` method generates loop to compute convective flux along the interface.
* At `construct_kernels` method of the ``Inters`` class in `pybaram.solvers.baseadvec` generates kernels.
* :math:`S_f, \vec{n}_f` are pre-computed and stored as `_mag_snorm` and `_vec_snorm` at ``BaseInters`` class in `pybaram.solvers.base.inters`.
* Various approximate Riemann solver :math:`H_c` are implemented in `pybaram.solvers.euler.rsolvers`.
*  `fpts` in each elements stores :math:`U_L, U_R` before execuation and saves :math:`H_c S_f` after execution.
