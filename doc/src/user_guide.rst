**********
User Guide
**********

How to Run
==========
pyBaram provides a console script, which uses ``argparse`` module.
When you run ``pybaram``, following help output is given::

    user@Computer ~/pyBaram$ pybaram

    usage: pybaram [-h] [--verbose] {import,partition,run,restart,export} ...

    positional arguments:
    {import,partition,run,restart,export}
                            sub-command help
        import              import --help
        partition           partition --help
        run                 run --help
        restart             run --help
        export              export --help

    optional arguments:
    -h, --help            show this help message and exit
    --verbose, -v

1. ``pybaram import`` --- Convert the mesh generator output to pyBaram mesh file (``.pbrm``).    
   pyBaram can convert `CGNS <https://cgns.github.io/>`_ mesh (``.cgns``) file or `Gmsh <http:http://geuz.org/gmsh/>`_ mesh file (``.msh``)
    
   Example::

        user@Computer ~/pyBaram$ pybaram import mesh.cgns mesh.pbrm

2. ``pybaram partition`` --- Partition mesh file for MPI parallel computation.
   Currently, unsplitted mesh file can be partitioned.
    
   Example::

        user@Computer ~/pyBaram$ pybaram partition 2 mesh.pbrm mesh_p2.pbrm

3. ``pybaram run`` --- Conduct flow simulation with a given mesh and configuration files (``.ini``).

   Example::
        
        user@Computer ~/pyBaram$ pybaram run mesh.pbrm conf.ini

   If you would like conduct MPI parallel computation, please use ``mpirun -n <cores>`` to launch ``pybaram`` script. 
   Note that the mesh file should be partitioned by the same number of cores.

   Example::
        
        user@Computer ~/pyBaram$ mpirun -np 2 pybaram run mesh_p2.pbrm conf.ini

4. ``pybaram restart`` --- Restart flow simulation with a given mesh and solution files. 
   If you would like to restart with different methods, please append the configuration file.

   Example::
        
        user@Computer ~/pyBaram$ pybaram restart mesh.pbrm sol-100.pbrs

5. ``pybaram export`` --- Convert solution files to `VTK <https://vtk.org/>`_ unstructured file (``.vtu``) 
   or `Tecplot <https://www.tecplot.com/>`_ data file (``.plt``).

   Example::
        
        user@Computer ~/pyBaram$ pybaram export mesh.pbrm sol-100.pbrs out.vtu


Mesh File
---------
``pyBaram`` can handle unstructured mixed elements; however, there are some cautions. Currently, only a single unstructured zone can be solved. It is important that volumes and faces are appropriately labeled. The volume label for a single zone should be set as fluid, and faces assigned for boundary conditions must have distinct labels.


Configuration File
==================
The parameters for ``pyBaram`` simulation are specified in the configuration file. This file is written in the INI file format, and it is parsed using the ``configparser`` module. The following sections provide details on the sections and parameters.

Backends
---------
The backend section configures how to run ``pybaram``. 
Currently, ``pybaram`` can be running on CPU and there is only 'backend-cpu' section.

[backend-cpu]
*************
Parameterize cpu backend with

1. multi-thread --- for selecting the multi-threading layer. This parameter passes to ``Numba``.

    ``single`` | ``parallel`` | ``omp`` | ``tbb`` 

    where

        * ``single`` --- use only one thread for the program. It is default value. 
          If you are running with only MPI parallel computation, please use it. 
          Some numerical schemes only support single thread option.
        
        * ``parallel`` --- use the default multi-threading layer of ``Numba``. 
          Depending on the libraries, ``omp`` or ``tbb`` is used.

        * ``omp`` --- use `OpenMP <https://www.openmp.org/>`_ multi-threading layer. 

        * ``tbb`` --- use `Intel Threading building Blocks <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html>`_ multi-threading layer.

Example::

    [backend-cpu]
    multi-thread = parallel

Constants
---------
In the constants section, essential and user-defined constants are configured. 
Some constants can be expressed as a function of other constants.
Followings are the essential constant depending on the equation to solve.

1. gamma --- ratio of the specific heats. For conventional air, :math:`\gamma=1.4`.
   All compressible equations need it.

    `float`

2. mu --- dynamic viscosity. It should be defined for constant viscosity.

    `float`

3. Pr --- Prandtl number. It should be defined for viscous simulation.
   For conventional air, :math:`Pr=0.72`.

    `float`

4. Prt --- Turbulent Prandtl number. It should be defined for turbulent simulation.
   For conventional air, :math:`Prt=0.9`.

    `float`

Example::

    [constants]
    gamma = 1.4
    Pr = 0.72
    Prt = 0.9
    Re = 6.5e6
    mach = 0.729
    rhof = 1.0
    uf = %(mach)s
    pf = 1/%(gamma)s
    mu = %(mach)s/%(Re)s
    nutf = 4*%(mu)s/%(rhof)s

Solvers
-------
In following sections, numerical schemes are configured.

[solver]
********
Type of equations and spatial discretization schemes are configured as follows.

1. system --- type of equations. 

    ``euler`` | ``navier-stokes`` | ``rans-sa`` | ``rans-kwsst``

    * ``rans-`` `model` --- Reynolds-averaged Navier-Stokes equation with turbulence model. 

        * ``rans-sa`` --- one equation Spalart-Allmaras model

        * ``rans-kwsst`` --- two-equation :math:`k\omega`-SST model

2. order --- spatial order of accuracy.

    ``1`` | ``2``

3. gradient ---- method to calculate gradient. The default value is ``hybrid``.

    ``hybrid`` | ``least-square`` | ``weighted-least-square`` | ``green-gauss``

4. limiter --- slope limiter for shock-capturing. It is configured only if the order is 2. 
   Default value is ``none``.

    ``none`` | ``mlp-u1`` | ``mlp-u2``

5. u2k --- tuning parameter for MLP-u2 limiter. Normally it is :math:`O(1)`.

    `float`

6. riemann-solver --- scheme to compute inviscid flux at interface.

    ``rusanov`` | ``roe`` | ``roem`` | ``rotated-roem`` | ``hllem`` | ``ausmpw+`` | ``ausm+up``

7. viscosity --- method to compute viscosity.
   Default value is ``constant``.

   ``constant`` | ``sutherland``

Example::

    [solver]
    system = rans-kwsst
    order = 2
    limiter = mlp-u2
    u2k = 5.0
    riemann-solver = ausmpw+
    viscosity = sutherland

[solver-viscosity-sutherland]
*****************************
The parameters associated with Sutherland's law can be configured as follows:

1. muref --- Reference viscosity of the problem. See the `note <https://turbmodels.larc.nasa.gov/Papers/sutherland_notes_cfl3d_fun3d.pdf>`_

    `float`

2. Tref --- Reference temperature of the problem. This is a dimensional unit.

    `float`

3. CpTf --- Free-stream enthalpy.
    
    `float`

4. Ts --- Sutherland temperature. This is a dimensional unit. Default value is 110.4 K.

    `float`

5. c1 --- Sutherland constant. This is a dimensional unit. Default value is :math:`1.458\times 10^{-6}`.

    `float`

If muref is not provided, the reference viscosity is computed using the following formula:

.. math::
    \mu_{\infty} = \frac{C_1 T_{\infty}^{3/2}}{T_{\infty} + T_s}

Example::

    [solver-viscosity-sutherland]
    muref = mu
    Tref = 300
    CpTf = 1 / (gamma -1)*pf/rhof
    Ts = 110.4
    c1 = 1.458e-6

[solver-time-integrator]
************************
Time integration (or relaxation) schemes and related parameters are configured.

1. mode --- steady or unsteady computation. Currently, dual-time stepping approach is not supported.

    ``steady`` | ``unsteady``

2. controller --- method to calculate time step size for unsteady simulation. 

    ``cfl`` | ``dt``

3. cfl --- Courant - Friedrichs - Lewy Number. 
   For unsteady simulation, it is required only for ``cfl`` controller.
   For steady simulation, it is a mandatory to solve.

    `float`

4. dt --- time step size for unsteady simulation with ``dt`` controller

    `float`

5. stepper --- method to advance time step.
   For unsteady simulation, there are following options

    ``eulerexplicit`` | ``tvd-rk3``

   For steady simulation, following options can be selected.

    ``eulerexplicit`` | ``tvd-rk3`` | ``rk5`` | ``lu-sgs`` | ``colored-lu-sgs`` | ``jacobi`` | ``blu-sgs`` | ``colored-blu-sgs``

    * ``lu-sgs`` --- This scheme works only if disabling multi-threading layer (``single``).

6. time --- initial and the last time for unsteady simulation

    `float` , `float`

7. max-iter --- the maximum iteration number for steady simulation

    `int`

8. tolerance --- stopping criteria for the magnitude of residual for steady simulation.

    `float`

9. res-var --- the residual variable to apply tolerance stopping criteria. 
   The variable should be selected among the conservative variables. 
   Default variable is `rho`.

    `string`

10. coloring --- the coloring strategy for colored LU-SGS scheme provided in `networkx.greedy_color` algorithm.
    Default variable is `largest_first`.

     `string`

11. turb-cfl-factor --- The factor of the ``cfl`` number for turbulent equations with respect to that of flow equations. 
    It adjusts the pseudo time for turbulence equations to alleviate numerical difficulties. The default value is 1.0.

12. sub-iter --- The maximum iteration number for Jacobi sub-iteration process. The default value is 10.

    `int`

13. sub-tol --- The stopping criteria for the Jacobi sub-iteration. The default value is 0.005.

    `float`

14. visflux-jacobian --- The computing type of viscous jacobian matrix for several implicit methods.
    There are three options, ``tlns``, ``approximate``, and ``none``.

    * ``tlns`` --- Based on Thin Layer Navier-Stokes equation (TLNS). Default.

    * ``approximate`` --- Based on Spectral radius. This type computes diagonal elements only.

    * ``none`` --- No viscous flux Jacobian imported. This type can cause convergence delay.

    * Applicable methods --- ``jacobi``, ``blu-sgs``, ``colored-blu-sgs``

    `string`

Example for unsteady simulation::

    [solver-time-integrator]
    controller = cfl
    stepper = tvd-rk3
    time = 0, 0.25
    cfl = 0.9

Example for steady simulation::

    [solver-time-integrator]
    mode = steady
    cfl = 5.0
    stepper = colored-lu-sgs
    max-iter = 10000
    tolerance = 1e-12
    res-var = rhou


[solver-cfl-ramp]
*****************
If this section is configured, CFL number can be ramped up linearly. 
Initially CFL number starts from the assigned ``cfl`` in ``[solver-time-integrator]``.

1. ``iter0`` --- iteration until maintaining the initial CFL.

    `int`

2. ``max-iter`` --- final iteration to finish CFL ramping.

    `int`

3. ``max-cfl`` --- final CFL 

    `float`

Example::

    [solver-cfl-ramp]
    iter0 = 500
    max-iter = 2500
    max-cfl = 10.0

Initial and Boundary Conditions
--------------------------------
Following sections configure initial and boundary conditions. 
The position variables (`x`, `y`, `z`) and 
few numerical functions (:math:`\sin, \cos, \tanh, \exp, \sqrt {}`)
and constant (:math:`\pi`) can be used.

Non-dimensionlization
*********************
``pyBaram`` does not non-dimensionalize the governing equations, so it is recommended to assign scaled variables for initial and boundary conditions. 

The following approach is recommended:

.. math::    
    \rho^* = \frac{\rho}{\rho_{\infty}}, u^* = \frac{u}{a_{\infty}}, p^*=\frac{p}{\rho_{\infty} a_{\infty}^2}, h^*=\frac{h}{a_{\infty}^2}

Here, :math:`\rho`, :math:`u`, and :math:`p` denote the density, velocity, and pressure, respectively. :math:`a` denotes the speed of sound, and :math:`h` denotes the specific enthalpy.

For the free-stream, the non-dimensionalized values can be written as follows:

.. math::
    \rho^*_{\infty}=1, u^*_{\infty}=M_{\infty}, p^*_{\infty}=\frac{1}{\gamma}, h^*_{\infty}=\frac{1}{\gamma-1}.

The normalized free-stream viscosity :math:`\mu_{\infty}^*` is chosen to satisfy Reynolds number :math:`Re_L` based on the characteristic length :math:`L`. 

.. math::
    Re_L = \frac{\rho_{\infty} u_{\infty} L}{\mu_{\infty}} = \frac{\rho^*_{\infty} u^*_{\infty} L^*}{\mu^*_{\infty}} \\
    \mu_{\infty}^* = \frac{M_{\infty}}{Re_L} L^* 

where :math:`M_{\infty}` denotes free-stream Mach number. :math:`L^*` is the non-dimnensionalized characteristic length (i.e., the chord length in the mesh.)

The the viscosity can be calulated via Sutherland law as:

.. math::
    \mu^* = \mu_{\infty}^* (T^*)^{3/2} \frac{1+T_s / T_{\infty}}{T^* + T_s / T_{\infty}}.

Here the unit of :math:`T_s, T_{\infty}` is in Kelvin.

[soln-ics]
**********
The intial condition is configured. All primitive variables should be configured. 

Examples::

    [soln-ics]
    rho = rhof
    u = uf*cos(aoa/180*pi)
    v = uf*sin(aoa/180*pi)
    p = pf

In this, examples, ``rhof``, ``uf``, ``pf`` and ``aoa`` are assigned at ``[constants]`` section.

[soln-bcs-`name`]
*****************
The boundary conditions for the label `name` is configured. 
The label should be same as the mesh file (``.pbrm``).

1. type --- type of boundary condition.
   To solve Euler system, following types can be used.

     ``slip-wall`` | ``sup-out`` | ``sup-in`` | ``sub-outp`` | ``far`` 

   To solve Navier-Stokes or RANS system, following types can be used.

     ``slip-wall`` | ``adia-wall`` | ``isotherm-wall`` | ``sup-out`` | ``sup-in`` | ``sub-outp`` | ``sub-inv`` |  ``far`` 

The details of type and required variables are summarized as follows.

* ``slip-wall`` --- slip wall boundary condition. 

* ``adia-wall`` --- adiabatic wall boundary condition. 

* ``isotherm-wall`` --- isothermal wall boundary condition.

    * ``CpTw`` --- wall enthalpy 

* ``sup-out`` --- supersonic outlet boundary condition

* ``sup-in`` --- supersonic inlet boundary condition

    * `all primitive variables`

* ``sub-outp`` --- subsonic outlet boundary condition with back pressure

    * ``p`` --- back pressure

* ``sub-inv`` --- subsonic inlet boundary condition with velocity

    * ``rho`` --- density

    * ``u, v, w`` --- velocity components.

    * `turbulent variables`

* ``sub-inptt`` --- subsonic inlet boundary condition with total conditions

    * ``p0`` --- total pressure

    * ``CpT0`` --- total enthalpy

    * ``dir`` --- velocity direction components.

    * `turbulent variables`

* ``far`` --- far boundary condition

    * `all primitive variables`

Examples::

    [soln-bcs-far]
    type = far
    rho = rhof
    u = uf*cos(aoa/180*pi)
    v = uf*sin(aoa/180*pi)
    p = pf

    [soln-bcs-airfoil]
    type = adia-wall

Plugins
--------
Plugins in ``pyBaram`` serve as post-processing modules after iterations. If a plugin is not configured, no post-processing will occur. The following plugins can be configured:

[soln-plugin-stats]
*******************
The `stats` plugin writes a fundamental log file. For unsteady simulations, it includes time and time step information for each iteration. In steady simulations, it records the residuals of all conservative variables.

1. ``flushsteps`` --- flush to file for every `flushstep`. Default value is 500.

Examples::
    
    [soln-plugin-stats]
    flushstep = 300


[soln-plugin-writer]
********************
This plugin writes the solution file.

1. ``name`` --- file name. In the name, {n} replaces iteration number and {t} replaces time.

2. ``iter-out`` --- write solution file for every `iter-out`.

Examples::
    
    [soln-plugin-writer]
    name = out-{n}
    iter-out = 5000


[soln-plugin-force-`name`]
**************************
This plugin computes aerodynamic force and moment coefficients along surface labelled `name`.

1. ``iter-out`` --- compute forces for every `iter-out` for steady simulation

    `int`

2. ``dt-out`` --- compute forces for every `dt-out` for unsteady simulation

    `float`

3. ``rho`` --- reference density to compute dynamic pressure

    `float`

4. ``vel`` --- reference velocity to compute dynamic pressure

    `float`

5. ``p`` --- reference pressure which is subtracted from the absolute pressure. 
   The relative pressure is integrated along the surface. The default value is zero.

    `float`

6. ``area`` --- reference area to compute aerodynamic coefficients

    `float`

7. ``length`` --- reference length to compute aerodynamic coefficients

    `float`

8. ``force-dir-name`` --- each character (subscript) denote force direction and its direction will be configured.

    `characters`

9. ``force-dir-`` `character` --- component of force direction vector of each subscript `character`. 
   The dimension of this vector should same as the dimension of space.

    `float`, `float`, ( `float` )

10. ``moment-center`` --- reference position to compute aerodynamic moment.

    `float`, `float`, ( `float` )

11. ``moment-dir-name`` --- each character (subscript) denote moment direction and its direction will be configured.

    `characters`

12. ``moment-dir-`` `character` --- component of moment direction vector of each subscript character. For two-dimensional computation, it is a scalar to indicate whether it is clockwise (-1) or counterclockwise (1). For three-dimensional computation, this vector should have the same dimension as the space.

    `float`, `float`, `float`

Examples::

    [soln-plugin-force-airfoil]
    iter-out = 50
    rho = rhof
    vel = uf
    p = pf
    area = 1.0
    length = 1.0
    force-dir-name = ld
    force-dir-l = -sin(aoa/180*pi), cos(aoa/180*pi)
    force-dir-d = cos(aoa/180*pi), sin(aoa/180*pi)
    moment-center = 0.25, 0
    moment-dir-name = z
    moment-dir-z = -1


[soln-plugin-surface-`name`]
****************************
This plugin integrates variables along the surface labeled as `name`. It provides both integrated and averaged values.

1. ``iter-out`` --- compute forces for every `iter-out` for steady simulation

    `int`

2. ``dt-out`` --- compute forces for every `dt-out` for unsteady simulation

    `float`

3. ``items`` --- items to integrate. Each item is separated by comma

    `strings`

4. `item` --- expression of `item`. As well as reserved variables for initial and boundary conditions,
   `nx`, `ny`, `nz`, which denote the component normal vector, can be used to express item.

Examples::

    [soln-plugin-surface-pout]
    iter-out = 500
    items = p0, mdot
    p0 = p*(1+ (gamma-1)/2*(u**2 + v**2)/(gamma*p/rho))**(gamma/(gamma-1))
    mdot = rho*(u*nx+v*ny)

In this example, total pressure (:math:`p_0`) and mass flow rate (:math:`\dot{m}`) is computed.

API
===
pyBaram provides an API for handling I/O and conducting simulations. Currently, only CLI (command line interface) functions are implemented. The basic usage is described as follows:

.. automodule:: pybaram.api.io
    :members:

.. automodule:: pybaram.api.simulation
    :members:
