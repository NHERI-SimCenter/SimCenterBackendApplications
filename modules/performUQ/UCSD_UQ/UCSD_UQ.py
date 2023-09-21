import argparse
import json
import os
import platform
import stat
import subprocess
from pathlib import Path
import sys


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--workflowInput")
    parser.add_argument("--workflowOutput")
    parser.add_argument("--driverFile")
    parser.add_argument("--runType")

    args, unknowns = parser.parse_known_args()

    workflowInput = args.workflowInput
    workflowOutput = args.workflowOutput
    driverFile = args.driverFile
    runType = args.runType

    templateDir = os.getcwd()
    tmpSimCenterDir = str(Path(templateDir).parents[0])
    
    mainScriptPath = os.path.dirname(os.path.realpath(__file__))

    if platform.system() == "Windows":
        pythonCommand = "python"
    else:
        pythonCommand = "python3"
    
    # Change permission of workflow driver
    workflowDriverFile = os.path.join(templateDir, driverFile)
    if runType in ["runningLocal"]:
        os.chmod(workflowDriverFile, stat.S_IXUSR | stat.S_IRUSR | stat.S_IXOTH)
    st = os.stat(workflowDriverFile)
    os.chmod(workflowDriverFile, st.st_mode | stat.S_IEXEC)
    driverFile = "./" + driverFile
    print("WORKFLOW: " + driverFile)

    if runType in ["runningLocal"]:
        dakotaJsonFile = os.path.join(os.path.abspath(templateDir), workflowInput)
        with open(dakotaJsonFile, "r") as f:
            jsonInputs = json.load(f)

        # Get the path to the mainscript.py of TMCMC
        mainScript = os.path.join(mainScriptPath, "mainscript.py")
        command = f'"{pythonCommand}" "{mainScript}" "{tmpSimCenterDir}" "{templateDir}" {runType} {driverFile} {workflowInput}'
        print(command)
        try:
            result = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
            returnCode = 0
        except subprocess.CalledProcessError as e:
            result = e.output
            print('RUNNING UCSD_UQ ERROR: ', result)
            returnCode = e.returncode


if __name__ == '__main__':
    main(sys.argv[1:])     
