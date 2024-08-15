import argparse  # noqa: INP001, D100
import json
import os
import platform
import stat
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workflowInput')
    parser.add_argument('--workflowOutput')
    parser.add_argument('--driverFile')
    parser.add_argument('--runType')

    args, unknowns = parser.parse_known_args()

    inputFile = args.workflowInput  # noqa: N816
    runType = args.runType  # noqa: N816
    workflowDriver = args.driverFile  # noqa: N816
    outputFile = args.workflowOutput  # noqa: N816

    cwd = os.getcwd()  # noqa: PTH109
    workdir_main = str(Path(cwd).parents[0])  # noqa: F821

    # mainScriptPath = inputArgs[0]
    # tmpSimCenterDir = inputArgs[1]
    # templateDir = inputArgs[2]
    # runType = inputArgs[3]  # either "runningLocal" or "runningRemote"

    mainScriptPath = os.path.dirname(os.path.realpath(__file__))  # noqa: PTH120, N816
    templateDir = cwd  # noqa: N816
    tmpSimCenterDir = str(Path(cwd).parents[0])  # noqa: N816, F821

    # Change permission of workflow driver
    if platform.system() != 'Windows':
        workflowDriverFile = os.path.join(templateDir, workflowDriver)  # noqa: PTH118, N816
        if runType == 'runningLocal':
            os.chmod(  # noqa: PTH101
                workflowDriverFile,
                stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR | stat.S_IXOTH,
            )
        st = os.stat(workflowDriverFile)  # noqa: PTH116
        os.chmod(workflowDriverFile, st.st_mode | stat.S_IEXEC)  # noqa: PTH101
        pythonCommand = 'python3'  # noqa: N816

    else:
        pythonCommand = 'python'  # noqa: N816
        workflowDriver = workflowDriver + '.bat'  # noqa: N816

    if runType == 'runningLocal':
        # Get path to python from dakota.json file
        dakotaJsonFile = os.path.join(os.path.abspath(templateDir), inputFile)  # noqa: PTH100, PTH118, N816
        with open(dakotaJsonFile) as f:  # noqa: PTH123
            jsonInputs = json.load(f)  # noqa: N816

        if 'python' in jsonInputs.keys():  # noqa: SIM118
            pythonCommand = jsonInputs['python']  # noqa: N816

        # Get the path to the mainscript.py of TMCMC
        #        mainScriptDir = os.path.split(mainScriptPath)[0]
        mainScript = os.path.join(mainScriptPath, 'mainscript.py')  # noqa: PTH118, N816
        command = f'{pythonCommand} {mainScript} {tmpSimCenterDir} {templateDir} {runType} {workflowDriver} {inputFile}'
        try:
            result = subprocess.check_output(  # noqa: S602
                command, stderr=subprocess.STDOUT, shell=True
            )
            returnCode = 0  # noqa: N816
        except subprocess.CalledProcessError as e:
            result = e.output
            returnCode = e.returncode  # noqa: N816
