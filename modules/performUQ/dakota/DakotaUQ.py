# written: UQ team @ SimCenter

# import functions for Python 2.X support
# from __future__ import division, print_function
# import sys
# if sys.version.startswith('2'): 
#     range=xrange
#     string_types = basestring
# else:
#     string_types = str

import shutil
import json
import os
import stat
import sys
import platform
from subprocess import Popen, PIPE
import subprocess
import glob
import argparse

def main(args):

    parser = argparse.ArgumentParser()

    parser.add_argument('--workflowInput')
    parser.add_argument('--workflowOutput')    
    parser.add_argument('--driverFile')
    parser.add_argument('--runType')

    args,unknowns = parser.parse_known_args()

    inputFile = args.workflowInput
    runType = args.runType
    workflow_driver = args.driverFile
    outputFile = args.workflowOutput

    #
    # open input file and check for any rvFiles
    #  - need to know in case need to modify driver file
    #
    
    with open(inputFile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    workflow_driver1 = 'blank'

    # run on local computer
    osType = platform.system()
    if runType in ['runningLocal',]:

        if (sys.platform == 'darwin' or sys.platform == "linux" or sys.platform == "linux2"):
            Dakota = 'dakota'
            workflow_driver1 = 'workflow_driver1'
            osType = 'Linux'
        else:
            Dakota = 'dakota'
            workflow_driver = workflow_driver + ".bat"
            workflow_driver1 = 'workflow_driver1.bat'
            osType = 'Windows'

    elif runType in ['runningRemote',]:
        Dakota = 'dakota'
        workflow_driver1 = 'workflow_driver1'
        osType = 'Linux'

    cwd = os.getcwd()
    print('CWD: ' + cwd)

    thisScriptDir = os.path.dirname(os.path.realpath(__file__))
    
    preprocessorCommand = '"{}/preprocessDakota" "{}" "{}" "{}" "{}" "{}" '.format(thisScriptDir,
                                                                        inputFile,
                                                                        workflow_driver,
                                                                        workflow_driver1,
                                                                        runType,
                                                                        osType)

    subprocess.Popen(preprocessorCommand, shell=True).wait()

    if runType in ['runningLocal']:
        os.chmod(workflow_driver,  stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR | stat.S_IXOTH)
        os.chmod(workflow_driver1, stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR | stat.S_IXOTH)

    command = Dakota + ' -input dakota.in -output dakota.out -error dakota.err'

    #Change permission of workflow driver
    st = os.stat(workflow_driver)
    os.chmod(workflow_driver, st.st_mode | stat.S_IEXEC)
    os.chmod(workflow_driver1, st.st_mode | stat.S_IEXEC)

    # copy the dakota input file to the main working dir for the structure
    shutil.copy("dakota.in", "../")

    # If calibration data files exist, copy to the main working directory
    if os.path.isfile("calibrationDataFilesToMove.cal"):
        calDataFileList = open("calibrationDataFilesToMove.cal", 'r')
        datFileList = calDataFileList.readlines()
        for line in datFileList:
            datFile = line.strip()
            if datFile.split('.')[-1] == 'tmpFile':
                shutil.copy(datFile, "../{}".format(datFile[:-8]))
            else:
                shutil.copy(datFile, "../")

        # os.remove("calibrationDataFilesToMove.cal")

    # change dir to the main working dir for the structure
    os.chdir("../")

    cwd = os.getcwd()

    if runType in ['runningLocal']:
        
        #    p = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
        #    for line in p.stdout:
        #        print(str(line))
        
        dakotaCommand = "dakota -input dakota.in -output dakota.out -error dakota.err"

        if 'parType' in data:
            print(data['parType'])
            if data['parType'] == 'parRUN':
                dakotaCommand = data['mpiExec'] + ' -n 1 ' + dakotaCommand

        print('running Dakota: ', dakotaCommand)

        try:
            result = subprocess.check_output(dakotaCommand, stderr=subprocess.STDOUT, shell=True)
            returncode = 0
        except subprocess.CalledProcessError as e:
            result = e.output
            print('RUNNING DAKOTA ERROR: ', result)
            returncode = e.returncode


        dakotaErrFile = os.path.join(os.getcwd(), 'dakota.err');
        dakotaOutFile = os.path.join(os.getcwd(), 'dakota.out');

        checkErrFile = os.path.getsize(dakotaErrFile)
        checkOutFile = os.path.exists(dakotaOutFile)

        if(checkOutFile == False and checkErrFile == 0 ):
            with open(dakotaErrFile, 'a') as file:
                file.write(result.decode("utf-8"))
        else:
            pass



if __name__ == '__main__':

    main(sys.argv[1:])
        
