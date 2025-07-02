import taichi as ti
from celeris.domain import Topodata, BoundaryConditions,Domain
from celeris.solver import Solver
from celeris.runner import Evolve
import argparse
import time
import numpy as np

ti.init(arch = ti.gpu)

baty = Topodata(filename='Topo1D.txt',path='./examples/1D',datatype='xz')

bc = BoundaryConditions(West=2,celeris=False,path='./examples/1D',filename='irrWaves1D.txt')

d = Domain(topodata=baty,x1=0.0,x2=480.0,Nx=480)

solver = Solver(model='Bouss',domain=d,boundary_conditions=bc,timeScheme=2,pred_or_corrector=True,useBreakingModel=False,useSedTransModel=False)

run = Evolve(solver = solver, maxsteps= 10000)
run.Evolve_1D_Display()
#run.Evolve_Headless()
