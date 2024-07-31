# written: UQ team @ SimCenter

# import functions for Python 2.X support
import sys

if sys.version.startswith('2'):
    range = xrange
    string_types = basestring
else:
    string_types = str

import os
import platform
import stat
import subprocess
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
def main(workflowinput, workflowoutput, driverfile, runtype):
    python = sys.executable

    # get os type
    osType = platform.system()
    if runtype in [
        'runningLocal',
    ]:
        if (
            sys.platform == 'darwin'
            or sys.platform == 'linux'
            or sys.platform == 'linux2'
        ):
            osType = 'Linux'
        else:
            driverfile = driverfile + '.bat'
            osType = 'Windows'
    elif runtype in [
        'runningRemote',
    ]:
        osType = 'Linux'

    thisScriptDir = os.path.dirname(os.path.realpath(__file__))

    os.chmod(
        f'{thisScriptDir}/preprocessUQpy.py',
        stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR | stat.S_IXOTH,
    )

    # 1. Create the UQy analysis python script
    preprocessorCommand = f"'{python}' '{thisScriptDir}/preprocessUQpy.py' --workflowInput {workflowinput} --driverFile {driverfile} --runType {runtype} --osType {osType}"

    subprocess.run(preprocessorCommand, shell=True, check=False)

    if runtype in ['runningLocal']:
        os.chmod(
            driverfile, stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR | stat.S_IXOTH
        )

    # 2. Run the python script
    UQpycommand = python + ' UQpyAnalysis.py' + ' 1> uqpy.log 2>&1 '

    # Change permission of workflow driver
    st = os.stat(driverfile)
    os.chmod(driverfile, st.st_mode | stat.S_IEXEC)

    if runtype in ['runningLocal']:
        print('running UQpy: ', UQpycommand)
        subprocess.run(
            UQpycommand, stderr=subprocess.STDOUT, shell=True, check=False
        )


if __name__ == '__main__':
    main()
