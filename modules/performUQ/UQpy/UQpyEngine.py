# written: UQ team @ SimCenter  # noqa: CPY001, D100, INP001

# import functions for Python 2.X support
import sys

if sys.version.startswith('2'):
    range = xrange  # noqa: A001, F821
    string_types = basestring  # noqa: F821
else:
    string_types = str

import os
import platform
import stat
import subprocess  # noqa: S404
import sys

import click


@click.command()
@click.option(
    '--workflowInput',
    required=True,
    help='Path to JSON file containing the details of FEM and UQ tools.',
)
@click.option(
    '--workflowOutput',
    required=True,
    help='Path to JSON file containing the details for post-processing.',
)
@click.option(
    '--driverFile',
    required=True,
    help='ASCII file containing the details on how to run the FEM application.',
)
@click.option(
    '--runType', required=True, type=click.Choice(['runningLocal', 'runningRemote'])
)
def main(workflowinput, workflowoutput, driverfile, runtype):  # noqa: ANN001, ANN201, ARG001, D103
    python = sys.executable

    # get os type
    osType = platform.system()  # noqa: N806
    if runtype == 'runningLocal':
        if (
            sys.platform == 'darwin'
            or sys.platform == 'linux'
            or sys.platform == 'linux2'
        ):
            osType = 'Linux'  # noqa: N806
        else:
            driverfile = driverfile + '.bat'  # noqa: PLR6104
            osType = 'Windows'  # noqa: N806
    elif runtype == 'runningRemote':
        osType = 'Linux'  # noqa: N806

    thisScriptDir = os.path.dirname(os.path.realpath(__file__))  # noqa: PTH120, N806

    os.chmod(  # noqa: PTH101
        f'{thisScriptDir}/preprocessUQpy.py',
        stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR | stat.S_IXOTH,
    )

    # 1. Create the UQy analysis python script
    preprocessorCommand = f"'{python}' '{thisScriptDir}/preprocessUQpy.py' --workflowInput {workflowinput} --driverFile {driverfile} --runType {runtype} --osType {osType}"  # noqa: N806

    subprocess.run(preprocessorCommand, shell=True, check=False)  # noqa: S602

    if runtype == 'runningLocal':
        os.chmod(  # noqa: PTH101
            driverfile, stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR | stat.S_IXOTH
        )

    # 2. Run the python script
    UQpycommand = python + ' UQpyAnalysis.py' + ' 1> uqpy.log 2>&1 '  # noqa: N806

    # Change permission of workflow driver
    st = os.stat(driverfile)  # noqa: PTH116
    os.chmod(driverfile, st.st_mode | stat.S_IEXEC)  # noqa: PTH101

    if runtype == 'runningLocal':
        print('running UQpy: ', UQpycommand)  # noqa: T201
        subprocess.run(  # noqa: S602
            UQpycommand, stderr=subprocess.STDOUT, shell=True, check=False
        )


if __name__ == '__main__':
    main()
