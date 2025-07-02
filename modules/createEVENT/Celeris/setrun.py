import os  # noqa: INP001, D100
import sys
import time
import argparse
import taichi as ti
from celeris.domain import BoundaryConditions, Domain, Topodata
from celeris.runner import Evolve
from celeris.solver import Solver
from celeris.utils import checjson

ti.init(arch=ti.gpu, advanced_optimization=True, kernel_profiler=False)
precision = ti.f32  # ti.f16 for half-precision or ti.f32 for single precision


def main():  # noqa: C901, D103
    # 1) Set the topography data

    directoryPath = './examples/CrescentCity'  # noqa: N806
    configFilename = 'config.json'  # noqa: N806
    bathymetryFilename = 'bathy.txt'  # noqa: N806
    waveFilename = 'waves.txt'  # noqa: N806

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
        filename=args.waves,
        precision=precision,
    )

    # 3) Build Numerical Domain
    d = Domain(
        topodata=baty,
        path=args.directory,
        configfile=args.config,
        precision=precision,
    )
    default_maxsteps = 10000
    if checjson('duration', bc.configfile) == 1:
        duration = float(bc.configfile['duration'])
        print('Duration:', duration)
        if checjson('dt', bc.configfile) == 1:
            if checjson('Courant_num', bc.configfile) == 1:
                Courant = float(bc.configfile['Courant_num'])
                dt = d.dt(Courant=Courant)
                print('dt:', dt)
            else:
                dt = float(bc.configfile['dt'])
                print('dt:', dt)
            maxsteps = int(duration / dt)
        else:
            if checjson('Courant_num', bc.configfile) == 1:
                Courant = float(bc.configfile['Courant_num'])
                dt = d.dt(Courant=Courant)
                print('dt:', dt)
                maxsteps = int(duration / dt)
            else:
                maxsteps = default_maxsteps
    else:
        maxsteps = default_maxsteps
    print('Maxsteps:', maxsteps)

    # 4) Solve using SWE (0) BOUSS (1)
    outputDirectoryPath = os.path.join(args.directory, 'output')  # noqa: PTH118, N806
    solver = Solver(domain=d, boundary_conditions=bc, maxsteps=maxsteps, outdir=outputDirectoryPath)

    # 5) Execution
    # print("Cold start")
    # run = Evolve(solver = solver, maxsteps= 100)
    # run = None
    # time.sleep(1)
    # print("Warm start")
        
        
    run = Evolve(
        solver=solver, maxsteps=maxsteps, saveimg=True, outdir=outputDirectoryPath
    )

    # run.Evolve_Headless() # Faster , no visualization

    # Showing -> 'h'    # cmap any on matplotlib Library e.g. 'BuGn', 'Blues'
    run.Evolve_Display(
        variable='h', cmapWater='Blues', cmapLand='plasma_r', showSediment=True
    )
    # run.Evolve_Display(
    #     variable='eta',
    #     vmin=-10.0,
    #     vmax=10.0,
    #     cmapWater='seismic',
    #     cmapLand='viridis',
    #     showSediment=False,
    # )
    # run.Evolve_Display(variable='vor',vmin=-0.5,vmax=0.5,cmapWater='viridis', cmapLand='plasma')


if __name__ == '__main__':
    # ti.reset()
    ti.init(arch=ti.gpu, advanced_optimization=True, kernel_profiler=False)
    precision = ti.f32  # ti.f16 for half-precision or ti.f32 for single precision

    main()
