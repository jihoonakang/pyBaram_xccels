***************
Developer Guide
***************

Overview of Code Structure
==========================

Start
-----
``pyBaram`` can be executed with the command `pybaram` which is linked to ``__main__.py``. 
For `run` or `restart` modes, the command calls `process_common` in :mod:`pybaram.api.simulation`.
Here, integrator object is initiated and `run` method is called to conduct simulation.

Integrators
-----------
Integrator object conducts time integration of the discretized equations.
When ``integrator`` initiated, it invokes `system` class in the :mod:`pybaram.solvers` module to compute
the right-hand side term of FVM. In addition, plugins are invoked 
by this integrator object for post-processing.
``pybaram`` can conduct both steady and unsteady simulations and they are implemented
in :mod:`pybaram.integrators.steady` and :mod:`pybaram.integrators.unsteady`, respectively. 
Here, `construct_stage` method generates the kernels for time integration.
For unsteady simulation, explicit Runge-Kutta schemes can be applied, which is implemented as below.

.. admonition:: TVD-RK3
   :class: dropdown

    .. autoclass:: pybaram.integrators.unsteady.TVDRK3
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

|

For steady simulation, explicit Runge-Kutta schemes or 
implicit LU-SGS schemes can be used, which is implemented as below.

.. admonition:: 5-stage Runge-Kutta
   :class: dropdown

    .. autoclass:: pybaram.integrators.steady.FiveStageRK
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. admonition:: LU-SGS
   :class: dropdown

    .. autoclass:: pybaram.integrators.steady.LUSGS
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. admonition:: Colored LU-SGS
   :class: dropdown

    .. autoclass:: pybaram.integrators.steady.ColoredLUSGS
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

|

The hierarchy of ``integrator`` class can be shown as below.

.. inheritance-diagram:: pybaram.integrators.unsteady.TVDRK3
                         pybaram.integrators.steady.FiveStageRK
                         pybaram.integrators.steady.LUSGS
                         pybaram.integrators.steady.ColoredLUSGS
    :parts: 1 


Solvers
-------
In :mod:`pybaram.solvers` module, the governing equations and their spatial discretizations
are implemented. For each submodule for governing equations, 
there are ``system``, ``elements``, ``inters`` and ``vertex`` objects.

System
*******
``system`` object, which is invoked from ``integrator``, initiates 
``elements``, ``inters`` and ``vertex`` objects by reading mesh and restarted solution, if exited.
These objects have `construct_kernels` method to generate kernels to compute right-hand side.
Here, ``rhside`` method schedules these kernels. 
For efficiency, non-blocking communications and computations are overlapped.
The class hierarchy of ``system`` can be depicted as below.

.. inheritance-diagram:: pybaram.solvers.ranskwsst.system
                         pybaram.solvers.ranssa.system
                         pybaram.solvers.navierstokes.system
                         pybaram.solvers.euler.system
    :top-classes: pybarm.solver.base.elements.BaseSystem
    :parts: 1 

|

* ``BaseSystem`` : initiates objects and generates kernels from these object

* ``BaseAdvecSystem`` : `rhside` method for advection problems, such as Euler systems.

    .. admonition:: rhside for advection
      :class: dropdown

        .. automethod:: pybaram.solvers.baseadvec.system.BaseAdvecSystem.rhside

* ``BaseAdvecSystem`` : `rhside` method for advection-diffusion problems, such as Navier-Stokes system.

    .. admonition:: rhside for advection-diffusion
      :class: dropdown

        .. automethod:: pybaram.solvers.baseadvecdiff.system.BaseAdvecDiffSystem.rhside

* ``RANSSystem`` : initiates objects and generates kernels from these object for RANS simulation


Elements
********
``elemenets`` object stores solution and other arrays. It also generates kernels, looping over elements.
The class hierarchy can be depicted as below.

.. inheritance-diagram:: pybaram.solvers.navierstokes.elements
                         pybaram.solvers.euler.elements
    :top-classes: pybarm.solver.base.elements.BaseElements
    :parts: 1 

* ``BaseElements`` : defines geometry and related properties

* ``BaseAdvecElements`` : common kernels for finite volume method, allocation of arrays

* ``EulerElements`` : specific kernels for Euler equations

* ``NavierStokesElements`` : specific kernels for Navier-Stokes equations

* ``FluidElements`` : physics of compressible inviscid flow

* ``ViscousFluidElements`` : physics of viscous flow

|

For RANS simultion, class hierarchy can be depicted as below.

.. inheritance-diagram:: pybaram.solvers.ranskwsst.elements
                         pybaram.solvers.ranssa.elements
    :top-classes: pybarm.solver.base.elements.BaseElements
    :parts: 1

* ``RANSElements`` : common kernels for RANS computation

* ``RANSSAElements`` : specific kernels for Spalart-Allmaras turbulence model

* ``RANSKWSSTElements`` : specific kernels for SST turbulence model

* ``RANSSAFluidElements`` : physics of Spalart-Allmaras turbulence model

* ``RANSKWSSTFluidElements`` : physics of SST turbulence model


Inters
*******
``inters`` objects generate kernels looping over interfaces.
There are three interfaces; Internal, boundary and MPI interfaces.
The abstract classes of them can be depicted as below.

.. inheritance-diagram:: pybaram.solvers.base.BaseIntInters
                         pybaram.solvers.base.BaseBCInters
                         pybaram.solvers.base.BaseMPIInters
    :top-classes: pybarm.solver.base.BaseInters
    :parts: 1

* ``BaseInters`` : computes geometrical properties and defines view to refer array in ``elements``

* ``BaseIntInters`` : abstract class for internal interface

* ``BaseBCInters`` : abstract class for physical boundary interface

* ``BaseMPIInters`` : abstract class for MPI boundary interface

|

The class hierarchy of internal interfaces can be depicted as below.

.. inheritance-diagram:: pybaram.solvers.ranskwsst.inters.RANSKWSSTIntInters
                         pybaram.solvers.ranssa.inters.RANSSAIntInters
                         pybaram.solvers.navierstokes.inters.NavierStokesIntInters
                         pybaram.solvers.euler.inters.EulerIntInters
    :top-classes: pybarm.solver.base.elements.BaseIntInters
    :parts: 1 

* ``BaseAdvecIntInters`` : common kernel to compute :math:`\Delta U_{fi}`

* ``BaseAdvecDiffIntInters`` : common kernel to compute :math:`\nabla U_f`

* ``EulerIntInters`` : kerenl to compute inviscid flux

* ``NavierStokesIntInters`` : kerenl to compute viscous flux

* ``RANSIntInters`` : kernel to compute RANS flux

* ``RANSSAInters`` : kernel to compute turbulent flux for Spalart-Allmaras turbulence model

* ``RANSKWSSTInters`` : kernel to compute turbulent flux for SST turbulence model

The class hierarchy of physical boundary interfaces can be depicted as below.

.. inheritance-diagram:: pybaram.solvers.ranskwsst.inters.RANSKWSSTBCInters
                         pybaram.solvers.ranssa.inters.RANSSABCInters
                         pybaram.solvers.navierstokes.inters.NavierStokesBCInters
                         pybaram.solvers.euler.inters.EulerBCInters
    :top-classes: pybarm.solver.base.elements.BaseBCInters
    :parts: 1 

The overall structure and role of these classes are the same as internal interfaces.
The ``construct_bc`` method in ``BaseAdvecInters`` compiles boundary condition function and
specific formulations are implemented in this class. 
For example, the hierarchy of boundary conditions for Euler equations can be depicted as below.

.. inheritance-diagram:: pybaram.solvers.euler.inters.EulerSupOutBCInters
                         pybaram.solvers.euler.inters.EulerSlipWallBCInters
                         pybaram.solvers.euler.inters.EulerSupInBCInters
                         pybaram.solvers.euler.inters.EulerFarInBCInters
                         pybaram.solvers.euler.inters.EulerSubOutPBCInters
    :top-classes: pybaram.solvers.euler.inters.EulerBCInters
    :parts: 1 

The class hierarchy of MPI interfaces can be depicted as below.

.. inheritance-diagram:: pybaram.solvers.ranskwsst.inters.RANSKWSSTMPIInters
                         pybaram.solvers.ranssa.inters.RANSSAMPIInters
                         pybaram.solvers.navierstokes.inters.NavierStokesMPIInters
                         pybaram.solvers.euler.inters.EulerMPIInters
    :top-classes: pybarm.solver.base.elements.BaseMPIInters
    :parts: 1 

The overall structure and role of these class are the same as internal interfaces.
MPI communication kernels are defined in ``BaseAdvecMPIInters``.

Vertex
*******
``vertex`` object generates kernel looping over vertex. 
The class hierarchy can be depicted as below.

.. inheritance-diagram:: pybaram.solvers.baseadvec.vertex
    :parts: 1 

* ``BaseVertex`` : view to refer array in ``elements``

* ``BaseVertex`` : kernel to find extreme values at vertex

Plugins
-------
The ``plugin`` modules handle the post-processing after each iteration or a fixed number of iterations.
The class hierarchy can be depicted as below.

.. inheritance-diagram:: pybaram.plugins.stats
                         pybaram.plugins.writer
                         pybaram.plugins.force
                         pybaram.plugins.surfint
    :top-classes: pybarm.plugins.base.BasePlugin
    :parts: 1 

* ``StatsPlugin`` : collect statistics (time step or residual)

* ``WriterPlugin`` : write output file

* ``ForcePlugin`` : compute aerodynamic force coefficients

* ``SurfIntPlugin`` : compute integrated and averaged properties over boundary surface.

Backends
--------
:mod:`pybaram.backends` module accelerates the pure Python loop and manages the execution of kernels.
Currently, only ``CPUBackend`` is implemented for serial and parallel computation using CPU.
This module provides two features; generating kernel and data type for executions.

Compile Kernel
**************
In ``integrators`` and ``solvers`` modules, pure python functions are defined.
They are compiled as kernel using loop generators in :mod:`pybaram.backends.cpu.loops`
Numba JIT compiler is called and pure Python functions is compiled and 
serial or parallel loop is constructed.

Data Types for Execution
************************
Currently, four data types are defined in :mod:`pybaram.backends.types`.

.. automodule:: pybaram.backends.types
    :members:

Variables
----------
The name of the variable `pyBaram` is somewhat condensed. 
Below table summarizes mathematical symbol and meaning of the major arrays.

.. list-table:: Notation of Variables in `pyBaram`
   :widths: 15 15 45 25
   :header-rows: 1

   * - Name
     - Symbol
     - Meaning
     - N/A
   * - upts
     - :math:`\bar{U}_i``
     - array of cell-averaged state variable vector
     -     
   * - fpts
     - :math:`U_f^\pm`
     - array of state vectors at faces
     -    
   * - grad
     - :math:`\nabla U_i` 
     - array of gradient of the state variables
     -    
   * - lim
     - :math:`\phi_i` 
     - array of slope limiter
     -    
   * - dt
     - :math:`\Delta t` 
     - array of time step size
     -    
   * - vpts
     - 
     - array of minimum and maximum at each vertex
     -    
   * - vol
     - :math:`\Delta V_i` 
     - array of volume of cell
     -    
   * - mag_snorm
     - :math:`\Delta A_f`
     - array of area of face
     -    
   * - vec_snorm
     - :math:`n_f`
     - array of unit normal vector of face
     -    


Code Snippets Analysis
======================
Here, the ways to generate kernels and construct MPI communications are explained with 
two sample code snippets.

Inviscid Flux Kernel
--------------------
There are two methods, ``make_serial_loop1d`` or ``make_parallel_loop1d``, in :mod:`pybaram.backends.cpu.loop` 
and they generate an accelerated kernel from a Python function. 
A function written in pure Python is compiled by a just-in-time compilation of Numba. 
When ``make_parallel_loop1d`` is used, each thread parallelly executes the loop of this compiled function.
Otherwise, the loop of the compiled function is executed sequentially.

.. automodule:: pybaram.backends.cpu.loop
    :members:
    :undoc-members:

As an example, ``comm_flux`` function is considered. 
``EulerIntInters`` class in :mod:`pybaram.solvers.euler.inters` has ``_make_flux`` method, 
which generates the kernel to compute numerical flux.
The function ``comm_flux`` uses a plain for loop, 
which is more similar to the loop structure of C/C++ or Fortran than a Pythonic-style one. 
Therefore, one can readily adopt a well-developed function from legacy solver into `pyBaram`. 
The allocation of local arrays was hoisted because of limited functionalities 
for developing local static variables in Numba. 
Furthermore, the ``_make_flux`` method passes this Python function 
to the ``make_serial_loop1d`` or ``make_parallel_loop1d`` method of the backend object and 
finally returns serialized or parallelized kernel, respectively.

.. autoclass:: pybaram.solvers.euler.inters.EulerIntInters

  .. method:: _make_flux

The generated kernel is constructed by `construct_kernels` method of ``BaseAdvecIntInters`` 
in :mod:`pybaram.solvers.baseadvec`. When this kernel is called, the reconstructed values at the face
:math:`{U}_f^{\pm}` is used as static argument. 
Thus, ``Kernel`` data type binds this compiled kernel and the static arguments. 
When ``Kernel`` object is called, dynamic arguments can be also provided.
All arguments are parsed, then the compiled kernel is executed.

.. autoclass:: pybaram.solvers.euler.inters.BaseAdvecIntInters

  .. method:: construct_kernels

Non-blocking Send/Receive 
-------------------------
``pyBaram`` exploits ``mpi4py`` package for MPI communication. 
Non-blocking communications are used, and they are overlapped with computing kernels.
In ``MPIInters`` class, these methods are implemented.

In `construct_kernels` method, non-blocking send and receive kernels and its request are 
constructed by `_make_send` and `_make_recv` methods. The buffers are passed these methods
and `_sendrecv` method is invoked. 
In this method, `start` function is returned. 
When this function is called with ``Queue`` instance in ``rhside``,
the MPI request of this communication is registered in ``Queue`` instance and 
start this non-blocking communication.
This communication is finalized when `sync` method in ``Queue`` instance is called.

.. autoclass:: pybaram.solvers.baseadvec.inters.BaseAdvecMPIInters

    .. method:: construct_kernels

    .. method:: _sendrecv
    
    .. method:: _make_send

    .. method:: _make_recv

.. autoclass:: pybaram.backends.types.Queue
    :noindex:

    .. method:: register

    .. method:: sync

