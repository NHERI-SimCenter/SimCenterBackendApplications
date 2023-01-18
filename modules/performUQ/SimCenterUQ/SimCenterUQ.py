# written: UQ team @ SimCenter

# import functions for Python 2.X support
from __future__ import division, print_function
import sys
if sys.version.startswith('2'): 
    range=xrange
    string_types = basestring
else:
    string_types = str

import shutil
import json
import os
import stat
import sys
import platform
from subprocess import Popen, PIPE
from pathlib import Path
import subprocess
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
    workflowDriver = args.driverFile
    outputFile = args.workflowOutput
    
    with open(inputFile, 'r') as f:
        data = json.load(f)

    
    if runType in ['runningLocal',]:
        if (sys.platform == 'darwin' or sys.platform == "linux" or sys.platform == "linux2"):
            # MAC
            surrogate = 'surrogateBuild.py' 
            plom = 'runPLoM.py' # KZ: main script of PLoM
            #natafExe = os.path.join('nataf_gsa','nataf_gsa')   
            natafExe = 'nataf_gsa'          
            osType = 'Linux'
            workflowDriver1 = 'workflowDriver1'
            python = 'python3'

        else:
            
            surrogate = 'surrogateBuild.py'  
            plom = 'runPLoM.py' # KZ: main script of PLoM
            #natafExe = os.path.join('nataf_gsa','nataf_gsa.exe')   
            natafExe = 'nataf_gsa.exe'
            workflowDriver = workflowDriver + ".bat"            
            workflowDriver1 = 'workflowDriver1.bat'
            osType = 'Windows'
            python = 'python'            
        

        cwd = os.getcwd()
        workdir_main = str(Path(cwd).parents[0]) 
        print('CWD: ' + cwd)
        print('work_dir: ' + workdir_main)
        
        # open the input json file
        with open(inputFile) as data_file:    
            data = json.load(data_file)

        uq_data = data["UQ"]

        myScriptDir = os.path.dirname(os.path.realpath(__file__))

        if os.path.exists(workflowDriver):
            os.chmod(workflowDriver, stat.S_IXUSR | stat.S_IRUSR | stat.S_IXOTH)
            
            st = os.stat(workflowDriver)
            os.chmod(workflowDriver, st.st_mode | stat.S_IEXEC)
        else:            
            print(workflowDriver + " not found.")


        # change dir to the main working dir for the structure
        os.chdir("../")
    
        #    p = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
        #    for line in p.stdout:
        #        print(str(line))
        
        #    dakotaCommand = "dakota -input dakota.in -output dakota.out -error dakota.err"
        
        '''
        LATER, CHANGE THE LOCATION
        '''
        #  
        
        if uq_data['uqType'] == 'Train GP Surrogate Model':
            simCenterUQCommand = '"{}" "{}/{}" "{}" {} {} {} {} 1> logFileSimUQ.txt 2>&1'.format(python,myScriptDir,surrogate,workdir_main,inputFile, workflowDriver, osType, runType)
        elif uq_data['uqType'] == 'Sensitivity Analysis':
            simCenterUQCommand = '"{}/{}" "{}" {} {} {} {} 1> logFileSimUQ.txt 2>&1'.format(myScriptDir,natafExe,workdir_main,inputFile, workflowDriver, osType, runType)
        elif uq_data['uqType'] == 'Forward Propagation':
            simCenterUQCommand = '"{}/{}" "{}" {} {} {} {} 1> logFileSimUQ.txt 2>&1'.format(myScriptDir,natafExe,workdir_main,inputFile, workflowDriver, osType, runType)
        # KZ: training calling runPLoM.py to launch the model training
        elif uq_data['uqType'] == 'PLoM Model':
            simCenterUQCommand = '"{}" "{}" "{}" {} {} {} {}'.format(python, os.path.join(myScriptDir,plom).replace('\\','/'),workdir_main.replace('\\','/'),inputFile,workflowDriver,osType,runType)
            
        print('running SimCenterUQ: ', simCenterUQCommand)

        # subprocess.Popen(simCenterUQCommand, shell=True).wait()
        
        try:
            result = subprocess.check_output(simCenterUQCommand, stderr=subprocess.STDOUT, shell=True)
            returncode = 0
            print('DONE SUCESS')
        except subprocess.CalledProcessError as e:
            result = e.output
            returncode = e.returncode
            print('DONE FAIL')

if __name__ == '__main__':

    main(sys.argv[1:])            
