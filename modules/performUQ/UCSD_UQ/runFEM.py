"""
authors: Dr. Frank McKenna*, Aakash Bangalore Satish*, Mukesh Kumar Ramancha, Maitreya Manoj Kurumbhati,
and Prof. J.P. Conte
affiliation: SimCenter*; University of California, San Diego

"""

import os
import subprocess
import shutil
import numpy as np


def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)


def runFEM(particleNumber, parameterSampleValues, variables, workdirMain, log_likelihood, calibrationData, numExperiments,
           covarianceMatrixList, edpNamesList, edpLengthsList, scaleFactors, shiftFactors, workflowDriver):
    """ 
    this function runs FE model (model.tcl) for each parameter value (par)
    model.tcl should take parameter input
    model.tcl should output 'output$PN.txt' -> column vector of size 'Ny'
    """

    workdirName = ("workdir." + str(particleNumber + 1))
    analysisPath = os.path.join(workdirMain, workdirName)

    # if os.path.isdir(analysisPath):
    #     shutil.rmtree(analysisPath)
    
    os.mkdir(analysisPath)

    # copy templatefiles
    templateDir = os.path.join(workdirMain, "templatedir")
    copytree(templateDir, analysisPath)

    # change to analysis directory
    os.chdir(analysisPath)

    # write input file and covariance multiplier values list
    covarianceMultiplierList = []
    parameterNames = variables["names"]
    with open("params.in", "w") as f:
        f.write('{}\n'.format(len(parameterSampleValues) - len(edpNamesList)))
        for i in range(len(parameterSampleValues)):
            name = str(parameterNames[i])
            value = str(parameterSampleValues[i])
            if name.split('.')[-1] != 'CovMultiplier':
                f.write('{} {}\n'.format(name, value))
            else:
                covarianceMultiplierList.append(parameterSampleValues[i])

    #subprocess.run(workflowDriver, stderr=subprocess.PIPE, shell=True)

    returnCode = subprocess.call(
       workflowDriver,
       shell=True,
       stdout=subprocess.DEVNULL,
       stderr=subprocess.STDOUT,
    )    # subprocess.check_call(workflow_run_command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)



    # Read in the model prediction
    if os.path.exists('results.out'):
        with open('results.out', 'r') as f:
            prediction = np.atleast_2d(np.genfromtxt(f)).reshape((1, -1))

        os.chdir("../")
        return log_likelihood(calibrationData, prediction, numExperiments, covarianceMatrixList, edpNamesList,
                            edpLengthsList, covarianceMultiplierList, scaleFactors, shiftFactors)
    else:
        os.chdir("../")
        return -np.inf