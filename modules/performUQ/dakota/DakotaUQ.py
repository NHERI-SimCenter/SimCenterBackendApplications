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
    print('cwd: ' + cwd)  # noqa: T201

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

        print('Running Dakota: ', dakotaCommand)  # noqa: T201

        run_success = True

        try:
            result = subprocess.check_output(  # noqa: S602
                dakotaCommand, stderr=subprocess.STDOUT, shell=True
            )
            returncode = 0
        except subprocess.CalledProcessError as e:
            result = e.output
            # print('RUNNING DAKOTA ERROR: ', result)  # noqa: RUF100, T201
            returncode = e.returncode  # noqa: F841
            run_success = False

        dakotaErrFile = os.path.join(cwd, 'dakota.err')  # noqa: N806, PTH109, PTH118, RUF100, W291
        dakotaOutFile = os.path.join(cwd, 'dakota.out')  # noqa: N806, PTH109, PTH118, RUF100
        dakotaTabFile = os.path.join(cwd, 'dakotaTab.out')  # noqa: N806, PTH109, PTH118, RUF100
        checkErrFile = os.path.exists(dakotaErrFile)  # noqa: PTH110, N806
        checkOutFile = os.path.exists(dakotaOutFile)  # noqa: PTH110, N806
        checkTabFile = os.path.exists(dakotaTabFile)  # noqa: F841, N806, PTH110

        checkErrSize = -1  # noqa: N806
        if checkErrFile > 0:
            checkErrSize = os.path.getsize(dakotaErrFile)  # noqa: F841, N806, PTH202
        if checkOutFile == False and checkErrFile == 0:  # noqa: E712
            with open(dakotaErrFile, 'a') as file:  # noqa: PTH123
                file.write(result.decode('utf-8'))
        else:
            pass

        if not run_success:
            # noqa: RUF100, W293
            display_err = '\nERROR. Dakota did not run. dakota.err not created.'
            #  # noqa: PLR2044, RUF100

            # First see if dakota.err is created
            with open(dakotaErrFile, 'r') as file:  # noqa: PTH123, UP015
                dakota_err = file.read()

            display_err = '\nERROR. Workflow did not run: ' + dakota_err

            # Second, see if workflow.err is found
            if 'workdir.' in dakota_err:
                display_err = '\nERROR. Workflow did not run: ' + dakota_err

                start_index = dakota_err.find('workdir.') + len('workdir.')
                end_index = dakota_err.find('\\', start_index)
                workdir_no = dakota_err[start_index:end_index]

                workflow_err_path = os.path.join(  # noqa: PTH118
                    os.getcwd(), f'workdir.{workdir_no}/workflow.err'  # noqa: PTH109
                )
                if os.path.isfile(workflow_err_path):  # noqa: N806, PTH110, PTH113, RUF100
                    with open(workflow_err_path, 'r') as file:  # noqa: PTH123, UP015
                        workflow_err = file.read()

                    if not workflow_err == '':  # noqa: SIM201
                        display_err = str(
                            '\nERROR running the workflow: \n'
                            + workflow_err
                            + '\n Check out more in '
                            + str(os.path.dirname(workflow_err_path)).replace(  # noqa: PTH120
                                '\\', '/'
                            )
                        )

            print(display_err)  # noqa: T201
            exit(  # noqa: PLR1722
                0
            )  # sy - this could be -1 like any other tools. But if it is 0, quoFEM,EE,WE,Hydro will highlight the error messages in "red" by using the parser in UI. To use this parser in UI, we need to make UI believe that the analysis is successful. Something that needs improvement


if __name__ == '__main__':
    main(sys.argv[1:])
