# written: UQ team @ SimCenter  # noqa: CPY001, D100, INP001

# import functions for Python 2.X support
import sys

if sys.version.startswith('2'):
    range = xrange  # noqa: A001, F821
    string_types = basestring  # noqa: F821
else:
    string_types = str

import argparse
import json
import os
import stat
import subprocess  # noqa: S404
import sys
from pathlib import Path


def main(args):  # noqa: ANN001, ANN201, D103
    parser = argparse.ArgumentParser()

    parser.add_argument('--workflowInput')
    parser.add_argument('--workflowOutput')
    parser.add_argument('--driverFile')
    parser.add_argument('--runType')

    args, unknowns = parser.parse_known_args()  # noqa: F841

    inputFile = args.workflowInput  # noqa: N806
    runType = args.runType  # noqa: N806
    workflowDriver = args.driverFile  # noqa: N806
    outputFile = args.workflowOutput  # noqa: N806, F841

    with open(inputFile, encoding='utf-8') as f:  # noqa: PTH123
        data = json.load(f)

    if runType == 'runningLocal':
        if (
            sys.platform == 'darwin'
            or sys.platform == 'linux'
            or sys.platform == 'linux2'
        ):
            # MAC
            surrogate = 'surrogateBuild.py'
            plom = 'runPLoM.py'  # KZ: main script of PLoM
            # natafExe = os.path.join('nataf_gsa','nataf_gsa')
            natafExe = 'nataf_gsa'  # noqa: N806
            osType = 'Linux'  # noqa: N806
            workflowDriver1 = 'workflowDriver1'  # noqa: N806
            python = 'python3'

        else:
            surrogate = 'surrogateBuild.py'
            plom = 'runPLoM.py'  # KZ: main script of PLoM
            # natafExe = os.path.join('nataf_gsa','nataf_gsa.exe')
            natafExe = 'nataf_gsa.exe'  # noqa: N806
            workflowDriver = workflowDriver + '.bat'  # noqa: N806, PLR6104
            workflowDriver1 = 'workflowDriver1.bat'  # noqa: N806, F841
            osType = 'Windows'  # noqa: N806
            python = 'python'

        cwd = os.getcwd()  # noqa: PTH109
        workdir_main = str(Path(cwd).parents[0])
        print('CWD: ' + cwd)  # noqa: T201
        print('work_dir: ' + workdir_main)  # noqa: T201

        # open the input json file
        with open(inputFile, encoding='utf-8') as data_file:  # noqa: PTH123
            data = json.load(data_file)

        uq_data = data['UQ']

        myScriptDir = os.path.dirname(os.path.realpath(__file__))  # noqa: PTH120, N806

        if os.path.exists(workflowDriver):  # noqa: PTH110
            os.chmod(workflowDriver, stat.S_IXUSR | stat.S_IRUSR | stat.S_IXOTH)  # noqa: PTH101

            st = os.stat(workflowDriver)  # noqa: PTH116
            os.chmod(workflowDriver, st.st_mode | stat.S_IEXEC)  # noqa: PTH101
        else:
            print(workflowDriver + ' not found.')  # noqa: T201

        # change dir to the main working dir for the structure
        os.chdir('../')

        #    p = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
        #    for line in p.stdout:
        #        print(str(line))

        #    dakotaCommand = "dakota -input dakota.in -output dakota.out -error dakota.err"

        """
        LATER, CHANGE THE LOCATION
        """

        if uq_data['uqType'] == 'Train GP Surrogate Model':
            simCenterUQCommand = f'"{python}" "{myScriptDir}/{surrogate}" "{workdir_main}" {inputFile} {workflowDriver} {osType} {runType} 1> logFileSimUQ.txt 2>&1'  # noqa: N806
        elif (
            uq_data['uqType'] == 'Sensitivity Analysis'
            or uq_data['uqType'] == 'Forward Propagation'
        ):
            simCenterUQCommand = f'"{myScriptDir}/{natafExe}" "{workdir_main}" {inputFile} {workflowDriver} {osType} {runType} 1> logFileSimUQ.txt 2>&1'  # noqa: N806
        # KZ: training calling runPLoM.py to launch the model training
        elif uq_data['uqType'] == 'PLoM Model':
            simCenterUQCommand = '"{}" "{}" "{}" {} {} {} {}'.format(  # noqa: N806
                python,
                os.path.join(myScriptDir, plom).replace('\\', '/'),  # noqa: PTH118
                workdir_main.replace('\\', '/'),
                inputFile,
                workflowDriver,
                osType,
                runType,
            )

        # if uq_data['uqType'] == 'Train GP Surrogate Model':
        #     simCenterUQCommand = '"{}" "{}/{}" "{}" {} {} {} {} 1> logFileSimUQ.txt 2> dakota.err2'.format(python,myScriptDir,surrogate,workdir_main,inputFile, workflowDriver, osType, runType)
        # elif uq_data['uqType'] == 'Sensitivity Analysis':
        #     simCenterUQCommand = '"{}/{}" "{}" {} {} {} {} 1> logFileSimUQ.txt 2> dakota.err2'.format(myScriptDir,natafExe,workdir_main,inputFile, workflowDriver, osType, runType)
        # elif uq_data['uqType'] == 'Forward Propagation':
        #     simCenterUQCommand = '"{}/{}" "{}" {} {} {} {} 1> logFileSimUQ.txt 2> dakota.err2'.format(myScriptDir,natafExe,workdir_main,inputFile, workflowDriver, osType, runType)
        # # KZ: training calling runPLoM.py to launch the model training
        # elif uq_data['uqType'] == 'PLoM Model':
        #     simCenterUQCommand = '"{}" "{}" "{}" {} {} {} {} 1> logFileSimUQ.txt 2> dakota.err2'.format(python, os.path.join(myScriptDir,plom).replace('\\','/'),workdir_main.replace('\\','/'),inputFile,workflowDriver,osType,runType)

        # if uq_data['uqType'] == 'Train GP Surrogate Model':
        #     simCenterUQCommand = '"{}" "{}/{}" "{}" {} {} {} {} 1> logFileSimUQ.txt 2>&1'.format(python,myScriptDir,surrogate,workdir_main,inputFile, workflowDriver, osType, runType)
        # elif uq_data['uqType'] == 'Sensitivity Analysis':
        #     simCenterUQCommand = '"{}/{}" "{}" {} {} {} {} 1> logFileSimUQ.txt 2>&1'.format(myScriptDir,natafExe,workdir_main,inputFile, workflowDriver, osType, runType)
        # elif uq_data['uqType'] == 'Forward Propagation':
        #     simCenterUQCommand = '"{}/{}" "{}" {} {} {} {} 1> logFileSimUQ.txt 2>&1'.format(myScriptDir,natafExe,workdir_main,inputFile, workflowDriver, osType, runType)
        # # KZ: training calling runPLoM.py to launch the model training
        # elif uq_data['uqType'] == 'PLoM Model':
        #     simCenterUQCommand = '"{}" "{}" "{}" {} {} {} {}'.format(python, os.path.join(myScriptDir,plom).replace('\\','/'),workdir_main.replace('\\','/'),inputFile,workflowDriver,osType,runType)

        print('running SimCenterUQ: ', simCenterUQCommand)  # noqa: T201

        # subprocess.Popen(simCenterUQCommand, shell=True).wait()

        try:
            result = subprocess.check_output(  # noqa: S602
                simCenterUQCommand, stderr=subprocess.STDOUT, shell=True
            )
            returncode = 0
            print('DONE SUCESS')  # noqa: T201
        except subprocess.CalledProcessError as e:
            result = e.output  # noqa: F841
            returncode = e.returncode  # noqa: F841
            print('DONE FAIL')  # noqa: T201


if __name__ == '__main__':
    main(sys.argv[1:])
