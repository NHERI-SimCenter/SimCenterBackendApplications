"""
authors: Mukesh Kumar Ramancha, Maitreya Manoj Kurumbhati, Prof. J.P. Conte, Aakash Bangalore Satish*
affiliation: University of California, San Diego, *SimCenter, University of California, Berkeley

"""

# ======================================================================================================================
import os
import sys
import json
import platform
from pathlib import Path
import subprocess

# ======================================================================================================================
def main(inputArgs):

    # Initialize analysis
    mainscriptPath = os.path.abspath(inputArgs[0])
    workdirMain = os.path.abspath(inputArgs[1])
    workdirTemplate = os.path.abspath(inputArgs[2])
    runType = inputArgs[3]  # either "runningLocal" or "runningRemote"
    workflowDriver = inputArgs[4]
    inputFile = inputArgs[5]
    
    mainScriptPath = os.path.dirname(os.path.realpath(__file__))
    cwd = os.getcwd()
    templateDir = cwd
    tmpSimCenterDir = str(Path(cwd).parents[0])

    try:
        os.remove('dakotaTab.out')
        os.remove('dakotaTabPrior.out')
    except OSError:
        pass

    with open(inputFile, "r") as f:
        inputs = json.load(f)
    
    uq_inputs = inputs["UQ"]
    if uq_inputs["uqType"] == "Metropolis Within Gibbs Sampler":
        mainScript = os.path.join(mainScriptPath, "mainscript_hierarchical_bayesian.py")
    else:
        mainScript = os.path.join(mainScriptPath, "mainscript_tmcmc.py")
    
    if platform.system() == "Windows":
        pythonCommand = "python"
    else:
        pythonCommand = "python3"

    command = '"{}" "{}" "{}" "{}" {} {} {}'.format(
        pythonCommand,
        mainScript,
        tmpSimCenterDir,
        templateDir,
        runType,
        workflowDriver,
        inputFile,
    )
    print(command)
    try:
        result = subprocess.check_output(
            command, stderr=subprocess.STDOUT, shell=True
        )
        returnCode = 0
    except subprocess.CalledProcessError as e:
        result = e.output
        returnCode = e.returncode



# ======================================================================================================================

if __name__ == "__main__":
    inputArgs = sys.argv
    main(inputArgs)

# ====================================================================================================================== 