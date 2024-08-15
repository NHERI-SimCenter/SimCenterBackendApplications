# written: UQ team @ SimCenter  # noqa: INP001, D100

# import functions for Python 2.X support
# from __future__ import division, print_function
# import sys
# if sys.version.startswith('2'):
#     range=xrange
#     string_types = basestring
# else:
#     string_types = str

import argparse
import json
import os
import platform
import shutil
import stat
import subprocess
import sys


def main(args):  # noqa: C901, D103
    parser = argparse.ArgumentParser()

    parser.add_argument('--workflowInput')
    parser.add_argument('--workflowOutput')
    parser.add_argument('--driverFile')
    parser.add_argument('--runType')

    args, unknowns = parser.parse_known_args()

    inputFile = args.workflowInput  # noqa: N806
    runType = args.runType  # noqa: N806
    workflow_driver = args.driverFile
    outputFile = args.workflowOutput  # noqa: N806, F841

    #
    # open input file and check for any rvFiles
    #  - need to know in case need to modify driver file
    #

    with open(inputFile, encoding='utf-8') as f:  # noqa: PTH123
        data = json.load(f)

    workflow_driver1 = 'blank'

    # run on local computer
    osType = platform.system()  # noqa: N806
    if runType == 'runningLocal':
        if (
            sys.platform == 'darwin'
            or sys.platform == 'linux'
            or sys.platform == 'linux2'
        ):
            Dakota = 'dakota'  # noqa: N806
            workflow_driver1 = 'workflow_driver1'
            osType = 'Linux'  # noqa: N806
        else:
            Dakota = 'dakota'  # noqa: N806
            workflow_driver = workflow_driver + '.bat'
            workflow_driver1 = 'workflow_driver1.bat'
            osType = 'Windows'  # noqa: N806

    elif runType == 'runningRemote':
        Dakota = 'dakota'  # noqa: N806
        workflow_driver1 = 'workflow_driver1'
        osType = 'Linux'  # noqa: N806

    cwd = os.getcwd()  # noqa: PTH109
    print('CWD: ' + cwd)  # noqa: T201

    thisScriptDir = os.path.dirname(os.path.realpath(__file__))  # noqa: PTH120, N806

    preprocessorCommand = f'"{thisScriptDir}/preprocessDakota" "{inputFile}" "{workflow_driver}" "{workflow_driver1}" "{runType}" "{osType}" '  # noqa: N806

    subprocess.Popen(preprocessorCommand, shell=True).wait()  # noqa: S602

    if runType == 'runningLocal':
        os.chmod(  # noqa: PTH101
            workflow_driver,
            stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR | stat.S_IXOTH,
        )
        os.chmod(  # noqa: PTH101
            workflow_driver1,
            stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR | stat.S_IXOTH,
        )

    command = Dakota + ' -input dakota.in -output dakota.out -error dakota.err'  # noqa: F841

    # Change permission of workflow driver
    st = os.stat(workflow_driver)  # noqa: PTH116
    os.chmod(workflow_driver, st.st_mode | stat.S_IEXEC)  # noqa: PTH101
    os.chmod(workflow_driver1, st.st_mode | stat.S_IEXEC)  # noqa: PTH101

    # copy the dakota input file to the main working dir for the structure
    shutil.copy('dakota.in', '../')

    # If calibration data files exist, copy to the main working directory
    if os.path.isfile('calibrationDataFilesToMove.cal'):  # noqa: PTH113
        calDataFileList = open('calibrationDataFilesToMove.cal')  # noqa: SIM115, PTH123, N806
        datFileList = calDataFileList.readlines()  # noqa: N806
        for line in datFileList:
            datFile = line.strip()  # noqa: N806
            if datFile.split('.')[-1] == 'tmpFile':
                shutil.copy(datFile, f'../{datFile[:-8]}')
            else:
                shutil.copy(datFile, '../')

        # os.remove("calibrationDataFilesToMove.cal")

    # change dir to the main working dir for the structure
    os.chdir('../')

    cwd = os.getcwd()  # noqa: PTH109

    if runType == 'runningLocal':
        #    p = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
        #    for line in p.stdout:
        #        print(str(line))

        dakotaCommand = (  # noqa: N806
            'dakota -input dakota.in -output dakota.out -error dakota.err'
        )

        if 'parType' in data:
            print(data['parType'])  # noqa: T201
            if data['parType'] == 'parRUN':
                dakotaCommand = data['mpiExec'] + ' -n 1 ' + dakotaCommand  # noqa: N806

        print('running Dakota: ', dakotaCommand)  # noqa: T201

        try:
            result = subprocess.check_output(  # noqa: S602
                dakotaCommand, stderr=subprocess.STDOUT, shell=True
            )
            returncode = 0
        except subprocess.CalledProcessError as e:
            result = e.output
            print('RUNNING DAKOTA ERROR: ', result)  # noqa: T201
            returncode = e.returncode  # noqa: F841

        dakotaErrFile = os.path.join(os.getcwd(), 'dakota.err')  # noqa: PTH109, PTH118, N806
        dakotaOutFile = os.path.join(os.getcwd(), 'dakota.out')  # noqa: PTH109, PTH118, N806
        checkErrFile = os.path.getsize(dakotaErrFile)  # noqa: PTH202, N806
        checkOutFile = os.path.exists(dakotaOutFile)  # noqa: PTH110, N806

        if checkOutFile == False and checkErrFile == 0:  # noqa: E712
            with open(dakotaErrFile, 'a') as file:  # noqa: PTH123
                file.write(result.decode('utf-8'))
        else:
            pass


if __name__ == '__main__':
    main(sys.argv[1:])
