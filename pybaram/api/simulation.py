# -*- coding: utf-8 -*-
from mpi4py import MPI

from pybaram.backends import get_backend
from pybaram.integrators import get_integrator
from pybaram.progressbar import Progressbar


def run(mesh, cfg, be='none', comm='none'):
    """
    Fresh run from mesh and configuration files.

    Parameters
    ----------
    mesh : pyBaram NativeReader object
        pyBaram mesh
    cfg : pyBaram INIFile object
        configuration file
    be : pyBaram backend object
        Backend to compute
    comm : string
        Initiated MPI communicator
    """
    # Run common
    _common(mesh, None, cfg, be, comm)


def restart(mesh, soln, cfg, be='none', comm='none'):
    """
    Restarted run from mesh and configuration files.

    Parameters
    ----------
    mesh : pyBaram NativeReader object
        pyBaram mesh
    soln : pyBaram NativeReader object
        previous pyBaram solution file
    cfg : pyBaram INIFile object
        configuration file
    be : pyBaram backend object
        Backend to compute
    comm : string
        Initiated MPI communicator
    """
    # Check mesh and solution file
    if mesh['mesh_uuid'] != soln['mesh_uuid']:
        raise RuntimeError('Solution is not computed by the mesh')

    # Run common
    _common(mesh, soln, cfg, be, comm)


def _common(msh, soln, cfg, backend, comm):
    if comm == 'none':
        # MPI comm
        comm = MPI.COMM_WORLD

    # Get backend
    if backend == 'none':
        backend = get_backend('cpu', cfg)

    # Get integrator
    integrator = get_integrator(backend, cfg, msh, soln, comm)

    # Add progress bar
    if comm.rank == 0:
        if integrator.mode == 'unsteady':
            pb = Progressbar(integrator.tlist[0], integrator.tlist[-1])

            def callb(intg): return pb(intg.tcurr)
        else:
            pb = Progressbar(0, integrator.itermax, fmt="{:03d}")

            def callb(intg): return pb(intg.iter+1)

        integrator.completed_handler.append(callb)

    integrator.run()
