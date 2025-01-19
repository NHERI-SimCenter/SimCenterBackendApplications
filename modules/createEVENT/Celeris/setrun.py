import time  # noqa: INP001, D100

import taichi as ti
from celeris.domain import BoundaryConditions, Domain, Topodata
from celeris.runner import Evolve
from celeris.solver import Solver

ti.init(arch=ti.gpu, advanced_optimization=True, kernel_profiler=False)
precision = ti.f32  # ti.f16 for half-precision or ti.f32 for single precision

# 1) Set the topography data
baty = Topodata(
    datatype='celeris',
    path='./applications/createEVENT/Celeris/examples/CrescentCity',
    filename='bathy.txt',
)

# 2) Set Boundary conditions
bc = BoundaryConditions(
    celeris=True,
    path='./applications/createEVENT/Celeris/examples/CrescentCity',
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
run = Evolve(solver=solver, maxsteps=100000, saveimg=False)

# run.Evolve_Headless() # Faster , no visualization

# Showing -> 'h'    # cmap any on matplotlib Library e.g. 'BuGn', 'Blues'
run.Evolve_Display(variable='h', cmapWater='Blues', showSediment=True)
# run.Evolve_Display(variable='eta',vmin=-5,vmax=5,cmapWater='Blues', showSediment=True)
# run.Evolve_Display(variable='vor',vmin=-0.5,vmax=0.5,cmapWater='viridis')
