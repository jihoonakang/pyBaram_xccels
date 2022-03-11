from argparse import ArgumentParser, FileType
from mpi4py import MPI

from pybaram.inifile import INIFile
from pybaram.backend import get_backend
from pybaram.readers import get_reader
from pybaram.partitions import get_partition
from pybaram.integrators import get_integrator
from pybaram.progressbar import Progressbar
from pybaram.readers.native import NativeReader
from pybaram.writers import get_writer

import h5py


def process_import(args):
    # Get reader
    reader = get_reader(args.inmesh, args.scale)

    # Get mesh in the pbm format
    mesh = reader.to_pbm()

    # Save to disk
    with h5py.File(args.outmesh, 'w') as f:
        for k, v in mesh.items():
            f[k] = v


def process_part(args):
    # mesh
    msh = NativeReader(args.mesh)

    npart = int(args.npart)

    get_partition(msh, args.out, npart)


def process_common(msh, soln, cfg):
    # MPI comm
    comm = MPI.COMM_WORLD

    # Get integrator
    integrator = get_integrator(cfg, msh, soln, comm)

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


def process_export(args):
    # Get writer
    writer = get_writer(args.mesh, args.soln, args.out)

    writer.write()


def process_run(args):
    # Read mesh
    msh = NativeReader(args.mesh)

    # Configuration
    cfg = INIFile(args.ini)

    # Run common
    process_common(msh, None, cfg)


def process_restart(args):
    # Read mesh and soln
    mesh = NativeReader(args.mesh)
    soln = NativeReader(args.soln)

    # Check mesh and solution file
    if mesh['mesh_uuid'] != soln['mesh_uuid']:
        raise RuntimeError('Solution is not computed by the mesh')

    # Config file
    if args.ini:
        cfg = INIFile(args.ini)
    else:
        cfg = INIFile()
        cfg.fromstr(soln['config'])

    # Run common
    process_common(mesh, soln, cfg)


def main():
    ap = ArgumentParser(prog='pybaram')
    sp = ap.add_subparsers(dest='cmd', help='sub-command help')

    # Common options
    ap.add_argument('--verbose', '-v', action='count')

    # Import command
    ap_import = sp.add_parser('import', help='import --help')
    ap_import.add_argument('inmesh', type=FileType('r'),
                           help='input mesh file')
    ap_import.add_argument('outmesh', help='output mesh file')
    ap_import.add_argument('-s', '--scale', type=float, default=1,
                           help='scale mesh')
    ap_import.set_defaults(process=process_import)

    # Partition command
    ap_part = sp.add_parser('partition', help='partition --help')
    ap_part.add_argument('npart', help='number of partition')
    ap_part.add_argument('mesh', help='mesh file')
    ap_part.add_argument('out', help='partitioned mesh file')
    ap_part.set_defaults(process=process_part)

    # Run command
    ap_run = sp.add_parser('run', help='run --help')
    ap_run.add_argument('mesh', type=str, help='mesh file')
    ap_run.add_argument('ini', type=str, help='config file')
    ap_run.set_defaults(process=process_run)

    # Run restart
    ap_restart = sp.add_parser('restart', help='run --help')
    ap_restart.add_argument('mesh', type=str, help='mesh file')
    ap_restart.add_argument('soln', type=str, help='solution file')
    ap_restart.add_argument('ini', nargs='?', type=str, help='config file')
    ap_restart.set_defaults(process=process_restart)

    # Export command
    ap_export = sp.add_parser('export', help='export --help')
    ap_export.add_argument('mesh', help='mesh file')
    ap_export.add_argument('soln', help='solution file')
    ap_export.add_argument('out', help='output file')
    ap_export.set_defaults(process=process_export)

    # Parse the arguments
    args = ap.parse_args()

    # Invoke the process method
    if hasattr(args, 'process'):
        args.process(args)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()