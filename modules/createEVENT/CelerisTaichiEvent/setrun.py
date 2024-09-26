import os  # noqa: INP001, D100
import subprocess
import sys

import pip

subprocess.run([sys.executable, '-m', 'pip', 'install', 'taichi'], check=False)  # noqa: S603


import time  # noqa: E402

from celeris.domain import BoundaryConditions, Domain, Topodata  # noqa: E402
from celeris.runner import Evolve  # noqa: E402
from celeris.solver import Solver  # noqa: E402

# 1) Set the topography data. Test[0,600,0,300]
# baty = Topodata('test_rect.xyz','xyz')
baty = Topodata('test_curve.xyz', 'xyz')
# baty = Topodata('Miami_Beach_FL/Miami_Beach_bathy/NorthMiamiBeach.xyz','celeris')

# 2) Set Boundary conditions
bc = BoundaryConditions(West=3, filename='irrWaves.txt')
# bc = BoundaryConditions(West=3,filename='Miami_Beach_FL/waves.txt')

# bc = BoundaryConditions(West=4,init_eta=8)

# 3) Build Numerical Domain (x1,x2,y1,y2,Nx,Ny)
d = Domain(1.0, 500.0, 1.0, 1000.0, 500, 1000)
d.topodata = baty.z()  # Assign baty data to domain
# Set numeric parameters
d.Courant = 0.10

# 4) Solve using SWE
# start_time = time.time()
# Show window render states of simulation or not show but save //BOUS  // SWE
solver = Solver(
    model='SWE',
    domain=d,
    boundary_conditions=bc,
    timeScheme=2,
    useSedTransModel=False,
)
run = Evolve(solver=solver, maxsteps=10000)
# run.Evolve_Headless()
run.Evolve_Display()

# No rendeer at all, send 1% states data to cpu()
# solver = SWE(domain=d, boundary_conditions=bc,timeScheme = 2,show_window=False, maxsteps=10000,outdir='./scratch')
# solver.EvolveHeadless()

# solver = SWE(domain=d, boundary_conditions=bc,timeScheme = 2,show_window=True, maxsteps=1000,model='BOUS')
# solver.Evolve()

# print("Grid resolution: ({:2.2f},{:2.2f})m".format(d.dx(),d.dy()))
# print("Time resolution: {:2.2f}s".format(d.dt()))
# print("Wall Clock Time {:2.2f}s seconds ---".format(time.time() - start_time))

# Plot Final Results
# flood = solver.txAuxiliary2.to_numpy()
# flood = flood[:,:,0]
# flood[flood<0.1] = np.nan
# xx,yy,d = solver.domain.topofield()
# flood[d>0]=np.nan
# levels = np.linspace(-2,11,21)
# plt.figure(figsize=(10, 3))
# plt.title('Max. Flow depth')

# sed = solver.txState_Sed.to_numpy()

# map_depth = plt.contourf(xx,yy,d,levels,cmap='gist_earth_r')
# map_flow = plt.pcolor(xx,yy,flood,cmap='jet')

# bar1=plt.colorbar(map_depth)
# bar2=plt.colorbar(map_flow)
# plt.show()
# print(sed[:,:,0]/10)
# print(sed.max())
# print(solver.maxdepth)
