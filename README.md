pyBaram
========

Overview
---------
pyBaram is an open-source Python-based software to solve compresible flow using finite volume method on unstructured grids. Baram means the 'Wind' in Korean. It is designed to solve compressible inviscid flow, laminar flow and turbulent flow using RANS (Reynolds Averaged Navier-Stokes) models. All codes are written in Python and hybrid parallel simulations are employed using the high Performane Python packages, Numba [[1]](#1) and mpi4py [[2]](#2).

## References
<a id="1">[1]</a> 
Lam, S. K., Pitrou, A., & Seibert, S. (2015). Numba: A llvm-based python jit compiler. In Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC (pp. 1–6).

<a id="2">[2]</a>
Dalcin, L., Paz, R., & Storti, M. (2005) MPI for Python, Journal of Parallel and Distributed Computing, 65(9), 1108-1115.

Examples
---------
Examples of using pyBaram are available in the examples directory. Currently available examples are:

- 3D Inviscid spherical explosion problem

- 2D transonic turbulent flow over RAE2822 airfoil

Authors
--------
Jin Seok Park <jinseok.park@inha.ac.kr>

License
---------
pyBaram is released under the new BSD License.
