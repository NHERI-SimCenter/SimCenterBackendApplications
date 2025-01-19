import sys  # noqa: INP001, D100
import time

import taichi as ti
from celeris.domain import BoundaryConditions, Domain, Topodata
from celeris.runner import Evolve
from celeris.solver import Solver

ti.init(arch=ti.gpu, advanced_optimization=True, kernel_profiler=False)
precision = ti.f32  # ti.f16 for half-precision or ti.f32 for single precision


def main():  # noqa: C901, D103
    # 1) Set the topography data

    configFilename = 'config.json'  # noqa: N806
    directoryPath = './examples/CrescentCity'  # noqa: N806
    bathymetryFilename = 'bathy.txt'  # noqa: N806

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        print('Running Celeris without any arguments')  # noqa: T201
    elif len(arguments) > 1:
        if arguments[0] == '--file' or arguments[0] == '-f':
            if (len(arguments) > 1) and (arguments[1] != ''):
                print('Running Celeris with config file:', arguments[1])  # noqa: T201
                configFilename = arguments[1]  # noqa: N806, F841
            else:
                print('Running Celeris without custom config file')  # noqa: T201
        if len(arguments) > 2:  # noqa: PLR2004
            if arguments[2] == '--directory' or arguments[2] == '-d':
                if (len(arguments) > 3) and (arguments[3] != ''):  # noqa: PLR2004
                    print('Running Celeris with directory:', arguments[3])  # noqa: T201
                    directoryPath = arguments[3]  # noqa: N806
        if len(arguments) > 4:  # noqa: PLR2004
            if arguments[4] == '--bathymetry' or arguments[4] == '-b':
                if (len(arguments) > 5) and (arguments[5] != ''):  # noqa: PLR2004
                    print(  # noqa: T201
                        'Running Celeris with bathymetry: ',
                        arguments[5],
                        ' contained in directory: ',
                        arguments[3],
                    )
                    bathymetryFilename = arguments[5]  # noqa: N806

    baty = Topodata(
        datatype='celeris',
        path=directoryPath,
        filename=bathymetryFilename,
    )

    # 2) Set Boundary conditions
    bc = BoundaryConditions(
        celeris=True,
        path=directoryPath,
        precision=precision,
    )

    # 3) Build Numerical Domain
    d = Domain(topodata=baty, precision=precision)

    # 4) Solve using SWE (0) BOUSS (1)
    solver = Solver(domain=d, boundary_conditions=bc)

    # 5) Execution
    # print("Cold start")
    # run = Evolve(solver = solver, maxsteps= 100)
    # run = None
    # time.sleep(1)
    # print("Warm start")
    run = Evolve(solver=solver, maxsteps=12000, saveimg=False)

    # run.Evolve_Headless() # Faster , no visualization

    # Showing -> 'h'    # cmap any on matplotlib Library e.g. 'BuGn', 'Blues'
    run.Evolve_Display(variable='h', cmapWater='Blues', showSediment=True)
    # run.Evolve_Display(variable='eta',vmin=-5,vmax=5,cmapWater='Blues', showSediment=True)
    # run.Evolve_Display(variable='vor',vmin=-0.5,vmax=0.5,cmapWater='viridis')


if __name__ == '__main__':
    # ti.reset()
    ti.init(arch=ti.gpu, advanced_optimization=True, kernel_profiler=False)
    precision = ti.f32  # ti.f16 for half-precision or ti.f32 for single precision

    main()
