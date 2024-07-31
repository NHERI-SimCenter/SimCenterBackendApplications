import argparse  # noqa: INP001, D100
import os
import platform
import shlex
import stat
import subprocess
import sys
from pathlib import Path


def main(args):  # noqa: ANN001, ANN201, D103
    parser = argparse.ArgumentParser()

    parser.add_argument('--workflowInput')
    parser.add_argument('--workflowOutput')
    parser.add_argument('--driverFile')
    parser.add_argument('--runType')

    args, unknowns = parser.parse_known_args()

    workflowInput = args.workflowInput  # noqa: N806
    workflowOutput = args.workflowOutput  # noqa: N806, F841
    driverFile = args.driverFile  # noqa: N806
    runType = args.runType  # noqa: N806

    if runType in ['runningLocal']:
        if platform.system() == 'Windows':
            pythonCommand = 'python'  # noqa: N806
            driverFile = driverFile + '.bat'  # noqa: N806
        else:
            pythonCommand = 'python3'  # noqa: N806

        mainScriptDir = os.path.dirname(os.path.realpath(__file__))  # noqa: PTH120, N806
        mainScript = os.path.join(mainScriptDir, 'mainscript.py')  # noqa: PTH118, N806
        templateDir = os.getcwd()  # noqa: PTH109, N806
        tmpSimCenterDir = str(Path(templateDir).parents[0])  # noqa: N806

        # Change permission of driver file
        os.chmod(driverFile, stat.S_IXUSR | stat.S_IRUSR | stat.S_IXOTH)  # noqa: PTH101
        st = os.stat(driverFile)  # noqa: PTH116
        os.chmod(driverFile, st.st_mode | stat.S_IEXEC)  # noqa: PTH101
        driverFile = './' + driverFile  # noqa: N806
        print('WORKFLOW: ' + driverFile)  # noqa: T201

        command = (
            f'"{pythonCommand}" "{mainScript}" "{tmpSimCenterDir}"'
            f' "{templateDir}" {runType} {driverFile} {workflowInput}'
        )
        print(command)  # noqa: T201

        command_list = shlex.split(command)

        result = subprocess.run(  # noqa: S603, UP022
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )

        err_file = Path(tmpSimCenterDir) / 'UCSD_UQ.err'
        err_file.touch()

        try:
            result.check_returncode()
        except subprocess.CalledProcessError:
            with open(err_file, 'a') as f:  # noqa: PTH123
                f.write(f'ERROR: {result.stderr}\n\n')
                f.write(f'The command was: {result.args}\n\n')
                f.write(f'The return code was: {result.returncode}\n\n')
                f.write(f'The output of the command was: {result.stdout}\n\n')


if __name__ == '__main__':
    main(sys.argv[1:])
