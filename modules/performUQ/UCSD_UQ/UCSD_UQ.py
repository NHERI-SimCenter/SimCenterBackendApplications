import argparse
import json
import os
import platform
import stat
import subprocess
from pathlib import Path

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--workflowInput")
    parser.add_argument("--workflowOutput")
    parser.add_argument("--driverFile")
    parser.add_argument("--runType")

    args, unknowns = parser.parse_known_args()

    inputFile = args.workflowInput
    runType = args.runType
    workflowDriver = args.driverFile
    outputFile = args.workflowOutput

    cwd = os.getcwd()
    workdir_main = str(Path(cwd).parents[0])

    mainScriptPath = os.path.dirname(os.path.realpath(__file__))
    templateDir = cwd
    tmpSimCenterDir = str(Path(cwd).parents[0])

    # Change permission of workflow driver
    if platform.system() != "Windows":
        workflowDriverFile = os.path.join(templateDir, workflowDriver)
        if runType in ["runningLocal"]:
            os.chmod(workflowDriverFile, stat.S_IXUSR | stat.S_IRUSR | stat.S_IXOTH)
        st = os.stat(workflowDriverFile)
        os.chmod(workflowDriverFile, st.st_mode | stat.S_IEXEC)
        pythonCommand = "python3"
        workflowDriver = "./" + workflowDriver
    else:
        pythonCommand = "python"

    print("WORKFLOW: " + workflowDriver)

    if runType in ["runningLocal"]:
        dakotaJsonFile = os.path.join(os.path.abspath(templateDir), inputFile)
        with open(dakotaJsonFile, "r") as f:
            jsonInputs = json.load(f)

        # Get the path to the mainscript.py of TMCMC
        mainScript = os.path.join(mainScriptPath, "mainscript.py")
        command = f'"{pythonCommand}" "{mainScript}" "{tmpSimCenterDir}" "{templateDir}" {runType} {workflowDriver} {inputFile}'
        print(command)
        try:
            result = subprocess.check_output(
                command, stderr=subprocess.STDOUT, shell=True
            )
            returnCode = 0
        except subprocess.CalledProcessError as e:
            result = e.output
            returnCode = e.returncode
