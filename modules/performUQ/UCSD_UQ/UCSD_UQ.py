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

    if runType in ["runningLocal"]:

        if platform.system() == "Windows":
            pythonCommand = "python"
            driverFile = driverFile + ".bat"
        else:
            pythonCommand = "python3"

        mainScriptDir = os.path.dirname(os.path.realpath(__file__))
        mainScript = os.path.join(mainScriptDir, "mainscript.py")
        templateDir = os.getcwd()
        tmpSimCenterDir = str(Path(templateDir).parents[0])

        # Change permission of driver file
        os.chmod(driverFile, stat.S_IXUSR | stat.S_IRUSR | stat.S_IXOTH)
        st = os.stat(driverFile)
        os.chmod(driverFile, st.st_mode | stat.S_IEXEC)
        driverFile = "./" + driverFile
        print("WORKFLOW: " + driverFile)

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
