# import functions for Python 2.X support  # noqa: INP001, D100
import os
import sys

if sys.version.startswith('2'):
    range = xrange  # noqa: A001, F821
    string_types = basestring  # noqa: F821
else:
    string_types = str

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))  # noqa: PTH120

import argparse
import platform
import shutil
import stat
import subprocess

from preprocessJSON import preProcessDakota


def main(args):  # noqa: ANN001, ANN201, D103
    parser = argparse.ArgumentParser()

    parser.add_argument('--workflowInput')
    parser.add_argument('--workflowOutput')
    parser.add_argument('--driverFile')
    parser.add_argument('--runType')
    parser.add_argument('--filesWithRV', nargs='*')
    parser.add_argument('--filesWithEDP', nargs='*')
    parser.add_argument('--workdir')

    args, unknowns = parser.parse_known_args()

    inputFile = args.workflowInput  # noqa: N806, F841
    runType = args.runType  # noqa: N806, F841
    workflow_driver = args.driverFile  # noqa: F841
    outputFile = args.workflowOutput  # noqa: N806, F841
    rvFiles = args.filesWithRV  # noqa: N806, F841
    edpFiles = args.filesWithEDP  # noqa: N806, F841

    myScriptDir = os.path.dirname(os.path.realpath(__file__))  # noqa: PTH120, N806

    # desktop applications
    if (
        uqData['samples'] is None  # noqa: F821
    ):  # this happens with new applications, workflow to change
        print('RUNNING PREPROCESSOR\n')  # noqa: T201
        osType = platform.system()  # noqa: N806
        preprocessorCommand = f'"{myScriptDir}/preprocessDakota" {bimName} {samName} {evtName} {edpName} {simName} {driverFile} {runDakota} {osType}'  # noqa: N806, F821
        subprocess.Popen(preprocessorCommand, shell=True).wait()  # noqa: S602
        print('DONE RUNNING PREPROCESSOR\n')  # noqa: T201

    else:
        scriptDir = os.path.dirname(os.path.realpath(__file__))  # noqa: PTH120, N806, F841
        numRVs = preProcessDakota(  # noqa: N806, F841
            bimName,  # noqa: F821
            evtName,  # noqa: F821
            samName,  # noqa: F821
            edpName,  # noqa: F821
            simName,  # noqa: F821
            driverFile,  # noqa: F821
            runDakota,  # noqa: F821
            uqData,  # noqa: F821
        )

        shutil.move(bimName, 'bim.j')  # noqa: F821
        shutil.move(evtName, 'evt.j')  # noqa: F821
        if os.path.isfile(samName):  # noqa: PTH113, F821
            shutil.move(samName, 'sam.j')  # noqa: F821
        shutil.move(edpName, 'edp.j')  # noqa: F821

    # Setting Workflow Driver Name
    workflowDriverName = 'workflow_driver'  # noqa: N806
    if (platform.system() == 'Windows') and (runDakota == 'run'):  # noqa: F821
        workflowDriverName = 'workflow_driver.bat'  # noqa: N806

    # Change permision of workflow driver
    st = os.stat(workflowDriverName)  # noqa: PTH116
    os.chmod(workflowDriverName, st.st_mode | stat.S_IEXEC)  # noqa: PTH101

    # copy the dakota input file to the main working dir for the structure
    shutil.move('dakota.in', '../')

    # change dir to the main working dir for the structure
    os.chdir('../')

    if runDakota == 'run':  # noqa: F821
        dakotaCommand = (  # noqa: N806
            'dakota -input dakota.in -output dakota.out -error dakota.err'
        )
        print('running Dakota: ', dakotaCommand)  # noqa: T201
        try:
            result = subprocess.check_output(  # noqa: S602
                dakotaCommand, stderr=subprocess.STDOUT, shell=True
            )
            returncode = 0
        except subprocess.CalledProcessError as e:
            result = e.output  # noqa: F841
            returncode = e.returncode  # noqa: F841


if __name__ == '__main__':
    main(sys.argv[1:])
