# import functions for Python 2.X support
import os
import sys

if sys.version.startswith('2'):
    range = xrange
    string_types = basestring
else:
    string_types = str

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import argparse
import platform
import shutil
import stat
import subprocess

from preprocessJSON import preProcessDakota


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--workflowInput')
    parser.add_argument('--workflowOutput')
    parser.add_argument('--driverFile')
    parser.add_argument('--runType')
    parser.add_argument('--filesWithRV', nargs='*')
    parser.add_argument('--filesWithEDP', nargs='*')
    parser.add_argument('--workdir')

    args, unknowns = parser.parse_known_args()

    inputFile = args.workflowInput
    runType = args.runType
    workflow_driver = args.driverFile
    outputFile = args.workflowOutput
    rvFiles = args.filesWithRV
    edpFiles = args.filesWithEDP

    myScriptDir = os.path.dirname(os.path.realpath(__file__))

    # desktop applications
    if (
        uqData['samples'] is None
    ):  # this happens with new applications, workflow to change
        print('RUNNING PREPROCESSOR\n')
        osType = platform.system()
        preprocessorCommand = f'"{myScriptDir}/preprocessDakota" {bimName} {samName} {evtName} {edpName} {simName} {driverFile} {runDakota} {osType}'
        subprocess.Popen(preprocessorCommand, shell=True).wait()
        print('DONE RUNNING PREPROCESSOR\n')

    else:
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        numRVs = preProcessDakota(
            bimName,
            evtName,
            samName,
            edpName,
            simName,
            driverFile,
            runDakota,
            uqData,
        )

        shutil.move(bimName, 'bim.j')
        shutil.move(evtName, 'evt.j')
        if os.path.isfile(samName):
            shutil.move(samName, 'sam.j')
        shutil.move(edpName, 'edp.j')

    # Setting Workflow Driver Name
    workflowDriverName = 'workflow_driver'
    if (platform.system() == 'Windows') and (runDakota == 'run'):
        workflowDriverName = 'workflow_driver.bat'

    # Change permision of workflow driver
    st = os.stat(workflowDriverName)
    os.chmod(workflowDriverName, st.st_mode | stat.S_IEXEC)

    # copy the dakota input file to the main working dir for the structure
    shutil.move('dakota.in', '../')

    # change dir to the main working dir for the structure
    os.chdir('../')

    if runDakota == 'run':
        dakotaCommand = (
            'dakota -input dakota.in -output dakota.out -error dakota.err'
        )
        print('running Dakota: ', dakotaCommand)
        try:
            result = subprocess.check_output(
                dakotaCommand, stderr=subprocess.STDOUT, shell=True
            )
            returncode = 0
        except subprocess.CalledProcessError as e:
            result = e.output
            returncode = e.returncode


if __name__ == '__main__':
    main(sys.argv[1:])
