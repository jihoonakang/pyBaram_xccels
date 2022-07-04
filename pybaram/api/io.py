# -*- coding: utf-8 -*-
from pybaram.readers import get_reader
from pybaram.partitions import get_partition
from pybaram.readers.native import NativeReader
from pybaram.writers import get_writer


import h5py
import os


def import_mesh(inmesh, outmesh, scale=1.0):
    """
    Import genreated mesh to pyBaram.

    Parameters
    ----------
    inmesh : string
        Original mesh from generator (CGNS, Gmsh)
    outmesh : string
        Converted pyBaram mesh (.pbrm)
    scale : float
        Geometric scale factor 
    """
    # Split ext
    extn = os.path.splitext(inmesh)[1]

    # Get reader
    reader = get_reader(extn, inmesh, scale)

    # Get mesh in the pbm format
    mesh = reader.to_pbm()

    # Save to disk
    with h5py.File(outmesh, 'w') as f:
        for k, v in mesh.items():
            f[k] = v


def partition_mesh(inmesh, outmesh, npart):
    """
    Paritioning pyBarm mesh

    Parameters
    ----------
    inmesh : string
        path and name of unspliited pyBaram mesh
    outmesh : string
        path and name of patitioned mesh
    npart : int
        number of partition
    """

    # mesh
    msh = NativeReader(inmesh)

    npart = int(npart)

    get_partition(msh, outmesh, npart)


def export_soln(mesh, soln, out):
    """
    Export solution to visualization file

    Parameters
    ----------
    mesh : string
        pyBaram mesh file
    soln : string
        pyBaram solution file
    out : string
        exported file for visualization
    """
    # Get writer
    writer = get_writer(mesh, soln, out)

    writer.write()