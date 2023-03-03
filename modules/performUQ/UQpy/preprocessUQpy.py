# written: UQ team @ SimCenter

# import functions for Python 2.X support
from __future__ import division, print_function
import sys
if sys.version.startswith('2'): 
    range=xrange
    string_types = basestring
else:
    string_types = str

import shutil
import json
import os
import stat
import sys
import platform
from subprocess import Popen, PIPE
from pathlib import Path
import subprocess
import argparse

def main(args):

    parser = argparse.ArgumentParser()

    parser.add_argument('--inputFile')
    parser.add_argument('--workflow_driver')    
    parser.add_argument('--runType')
    parser.add_argument('--osType')
    
    args,unknowns = parser.parse_known_args()

    inputFile = args.inputFile   # JSON FILE information from GUI
    workflow_driver = args.workflow_driver
    runType = args.runType
    osType = args.osType
    
    # Step 1: Read the JSON file
    with open(inputFile, "r") as f:
        data = json.load(f)
    
    uq_data = data["UQ"]
    rv_data = data["randomVariables"]
    fem_data = data["Applications"]["FEM"]
    edp_data = data["EDP"]
    
    cwd = os.getcwd()
    templateDir = cwd
    with open(os.path.join(templateDir, "UQpyAnalysis.py"), "w") as f:
        pass
    
    # f.write("from UQpy import PythonModel\n")
    # f.write("from UQpy.distributions import *\n")
    # f.write("from UQpy.reliability import SubsetSimulation\n")
    # f.write("from UQpy.run_model.RunModel import RunModel\n")
    # f.write("from UQpy.sampling import ModifiedMetropolisHastings, Stretch, MetropolisHastings\n")
    # f.write("distributionRV = []\n")
    # f.write("num_of_rves = len(RVdata)\n")
    # f.write("for i in range(num_rves):\n")
    #     f.write("nameRV.append(RVdata[i]["name"])\n")
    #     f.write("if RVdata[i]["distribution"] == 'Normal':\n")
    #         f.write("distributionRV.append(Normal(loc={0}, scale={1}))".format(rv_data[i]["scaleparam"], rv_data[i]["shapeparam"]))
    #         #distributionRV.append(Normal(loc=RVdata[i]["scaleparam"], scale=RVdata[i]["shapeparam"])) # check variance 

         

    # # Create RunModel object
    # # m = PythonModel(model_script='local_Rosenbrock_pfn.py', model_object_name="RunPythonModel")
    # # model = RunModel(model=m)
    
    # # Create Distribution object
    # dist = Rosenbrock(p=100.)
    # dist_prop1 = Normal(loc=0, scale=1)
    # dist_prop2 = Normal(loc=0, scale=10)

    # x = stats.norm.rvs(loc=0, scale=1, size=(100, 2), random_state=83276)

    # # Create MCMC object

    # mcmc_init1 = ModifiedMetropolisHastings(dimension=2, log_pdf_target=dist.log_pdf, seed=x.tolist(),
    #                                                            burn_length=1000, proposal=[dist_prop1, dist_prop2],
    #                                                            random_state=8765)
    # mcmc_init1.run(10000)

    # sampling=Stretch(log_pdf_target=dist.log_pdf, dimension=2, n_chains=1000, random_state=38546)
    
    # # Run Subset simulation
    # x_ss_MMH = SubsetSimulation(sampling=sampling, runmodel_object=model, conditional_probability=0.1,
    #                             nsamples_per_subset=10000, samples_init=mcmc_init1.samples)
                                
                                