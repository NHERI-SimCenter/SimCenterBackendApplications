import os  # noqa: INP001, D100
import sys
import time
import argparse
import taichi as ti
from celeris.domain import BoundaryConditions, Domain, Topodata
from celeris.runner import Evolve
from celeris.solver import Solver

ti.init(arch=ti.gpu, advanced_optimization=True, kernel_profiler=False)
precision = ti.f32  # ti.f16 for half-precision or ti.f32 for single precision


def main():  # noqa: C901, D103
    # 1) Set the topography data

    directoryPath = './examples/CrescentCity'  # noqa: N806
    configFilename = 'config.json'  # noqa: N806
    bathymetryFilename = 'bathy.txt'  # noqa: N806
    waveFilename = 'waves.txt'  # noqa: N806

    # i = 0
    # print('Running Celeris with arguments:')  # noqa: T201
    # for arg in sys.argv:
    #     print('Argument ', str(i), ': ', arg)  # noqa: T201
    #     i += 1  # noqa: SIM113

    # arguments = sys.argv[1:]
    # if len(arguments) == 0:
    #     print('Running Celeris without any arguments.')  # noqa: T201
    # else:
    #     print('Running Celeris with custom arguments.')  # noqa: T201
    #     if len(arguments) > 0:
    #         if arguments[0] == '--directory' or arguments[0] == '-d':
    #             if (len(arguments) > 1) and (arguments[1] != ''):
    #                 directoryPath = arguments[1]  # noqa: N806
    #                 print('Running Celeris with directory:', directoryPath)  # noqa: T201
    #     if len(arguments) > 2:  # noqa: PLR2004
    #         if (
    #             arguments[2] == '--file'
    #             or arguments[2] == '-f'
    #             or arguments[2] == '--config'
    #         ):
    #             if (len(arguments) > 3) and (arguments[3] != ''):  # noqa: PLR2004
    #                 print('Running Celeris with config file:', arguments[3])  # noqa: T201
    #                 configFilename = arguments[3]  # noqa: N806
    #             else:
    #                 print(  # noqa: T201
    #                     'Running Celeris without custom config file: ',
    #                     configFilename,
    #                     ' contained in directory: ',
    #                     directoryPath,
    #                 )
    #     if len(arguments) > 4:  # noqa: PLR2004
    #         if (
    #             arguments[4] == '--bathymetry'
    #             or arguments[4] == '-b'
    #             or arguments[4] == '--bathy'
    #         ):
    #             if (len(arguments) > 5) and (arguments[5] != ''):  # noqa: PLR2004
    #                 bathymetryFilename = arguments[5]  # noqa: N806
    #                 print(  # noqa: T201
    #                     'Running Celeris with bathymetry: ',
    #                     waveFilename,
    #                     ' contained in directory: ',
    #                     directoryPath,
    #                 )
    #     if len(arguments) > 6:  # noqa: PLR2004
    #         if (
    #             arguments[6] == '--waves'
    #             or arguments[6] == '--wave'
    #             or arguments[6] == '-w'
    #         ):
    #             if (len(arguments) > 7) and (arguments[7] != ''):  # noqa: PLR2004
    #                 waveFilename = arguments[7]  # noqa: N806
    #                 print(  # noqa: T201
    #                     'Running Celeris with waves: ',
    #                     waveFilename,
    #                     ' contained in directory: ',
    #                     directoryPath,
    #                 )

    parser = argparse.ArgumentParser(
        description='Run Celeris simulation with specified inputs.'
    )

    # Define command-line arguments
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default='./examples/CrescentCity',
        help='Path to the working directory (default: ./examples/CrescentCity)'
    )
    parser.add_argument(
        '-f', '--file', '--config',
        dest='config',
        type=str,
        default='config.json',
        help='Configuration file name (default: config.json)'
    )
    parser.add_argument(
        '-b', '--bathymetry', '--bathy',
        dest='bathymetry',
        type=str,
        default='bathy.txt',
        help='Bathymetry file name (default: bathy.txt)'
    )
    parser.add_argument(
        '-w', '--waves', '--wave',
        dest='waves',
        type=str,
        default='waves.txt',
        help='Wave file name (default: waves.txt)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Print received arguments
    print('Running Celeris with the following settings:')
    print('  Directory:', args.directory)
    print('  Config file:', args.config)
    print('  Bathymetry file:', args.bathymetry)
    print('  Wave file:', args.waves)

    baty = Topodata(
        datatype='celeris',
        path=args.directory,
        filename=args.bathymetry,
    )

    # 2) Set Boundary conditions
    bc = BoundaryConditions(
        celeris=True,
        path=args.directory,
        configfile=args.config,
        wavefile=args.waves,
        precision=precision,
    )

    # 3) Build Numerical Domain
    d = Domain(
        topodata=baty,
        path=args.directory,
        configfile=args.config,
        precision=precision,
    )

    # 4) Solve using SWE (0) BOUSS (1)
    outputDirectoryPath = os.path.join(args.directory, 'output')  # noqa: PTH118, N806
    solver = Solver(domain=d, boundary_conditions=bc, outdir=outputDirectoryPath)

    # 5) Execution
    # print("Cold start")
    # run = Evolve(solver = solver, maxsteps= 100)
    # run = None
    # time.sleep(1)
    # print("Warm start")
    run = Evolve(
        solver=solver, maxsteps=10000, saveimg=True, outdir=outputDirectoryPath
    )

    # run.Evolve_Headless() # Faster , no visualization

    # Showing -> 'h'    # cmap any on matplotlib Library e.g. 'BuGn', 'Blues'
    run.Evolve_Display(
        variable='h', cmapWater='Blues', cmapLand='plasma_r', showSediment=True
    )
    # run.Evolve_Display(
    #     variable='eta',
    #     vmin=-0.5,
    #     vmax=0.60,
    #     cmapWater='seismic',
    #     cmapLand='viridis',
    #     showSediment=True,
    # )
    # run.Evolve_Display(variable='vor',vmin=-0.5,vmax=0.5,cmapWater='viridis', cmapLand='plasma')


if __name__ == '__main__':
    # ti.reset()
    ti.init(arch=ti.gpu, advanced_optimization=True, kernel_profiler=False)
    precision = ti.f32  # ti.f16 for half-precision or ti.f32 for single precision

    main()
