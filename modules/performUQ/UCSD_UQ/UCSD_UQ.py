import argparse
import os
import platform
import shlex
import stat
import subprocess
import sys
from pathlib import Path


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

        command = (
            f'"{pythonCommand}" "{mainScript}" "{tmpSimCenterDir}"'
            f' "{templateDir}" {runType} {driverFile} {workflowInput}'
        )

        command_list = shlex.split(command)

        err_file = Path(tmpSimCenterDir) / "UCSD_UQ.err"
        err_file.touch()

        try:
            result = subprocess.run(command_list, capture_output=True, text=True)
            result.check_returncode()
        except subprocess.CalledProcessError:
            with open(err_file, "a") as f:
                f.write(f"ERROR: {result.stderr}")


if __name__ == "__main__":
    main(sys.argv[1:])
