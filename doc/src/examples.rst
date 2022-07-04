**********
Examples
**********

Three-dimensional explosion problem
===================================
This example is the extension of shock-tube problem to three-dimesional sphere. 
The details of flow configuration can be found in `this paper <https://doi.org/10.1016/j.compfluid.2012.04.015>`_.
Procedures to obtain unsteady solution can be presented as follows.

1. Convert mesh::

    user@Compuer ~/pyBaram$ pybaram import explosion.cgns explosion.pbrm

2. Partitioning mesh::

    user@Compuer ~/pyBaram$ pybaram partition 4 explosion.pbrm explosion_p4.pbrm

3. Running parallel simulation::

    user@Compuer ~/pyBaram$ mpirun -n 4 pybaram run explosion_p4.pbrm explosion.ini

4. Convert VTK output file for visualization::

    user@Compuer ~/pyBaram$ pybaram export explosion_p4.pbrm out-0.25.pbrs out.vtu

5. Visualizing the solution with Paraview, you can obtain following result.

.. figure:: ./figs/explosion/Density_contour.png
   :width: 200px
   :figwidth: 200px
   :alt: explosion
   :align: center

   Density contour of explosion problem


Transonic flow over RAE2822 airfoil
===================================
One of the famous benchmark to solve transonic flow over airfoil.
The detail flow conditions can be obtained from `NPARC validation page <https://www.grc.nasa.gov/www/wind/valid/raetaf/raetaf.html>`_.
The mesh file is obtained from `SU2 tutorial page <https://su2code.github.io/tutorials/Turbulent_2D_Constrained_RAE2822/>`_.
Procedures to obtain steady-state solution are presented as follows.

1. Convert mesh::

    user@Compuer ~/pyBaram$ pybaram import rae2822.cgns rae2822.pbrm

2. Running simulations::

    user@Compuer ~/pyBaram$ pybaram run rae2822.pbrm rae2822.ini

3. Convert VTK output file for visualization::

    user@Compuer ~/pyBaram$ pybaram export rae2822.pbrm out-10000.pbrs out.vtu

4. Visualizing the solution with Paraview, you can obtain following result.

.. figure:: ./figs/rae2822/Mach_contour.png
   :width: 450px
   :figwidth: 450px
   :alt: rae2822
   :align: center

   Mach contour of flow over RAE2822 airfoil
