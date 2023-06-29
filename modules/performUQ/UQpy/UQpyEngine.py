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
import click

@click.command()
@click.option("--workflowInput", required=True, help="Path to JSON file containing the details of FEM and UQ tools.")
@click.option("--workflowOutput", required=True, help="Path to JSON file containing the details for post-processing.")
@click.option("--driverFile", required=True, help="ASCII file containing the details on how to run the FEM application.")
@click.option("--runType", required=True, type=click.Choice(['runningLocal','runningRemote']))
def main(workflowinput, workflowoutput, driverfile, runtype):
    # with open(inputFile, "r") as f:
    #     data = json.load(f)
    python = sys.executable

    # run on local computer
    osType = platform.system()
    if runtype in ['runningLocal',]:
        if (sys.platform == 'darwin' or sys.platform == "linux" or sys.platform == "linux2"):
            osType = 'Linux'
        else:
            workflowDriver = workflowDriver + ".bat"            
            osType = 'Windows'
    elif runtype in ['runningRemote',]:
        osType = 'Linux'        
        
    cwd = os.getcwd()
    templateDir = cwd
    tmpSimCenterDir = str(Path(cwd).parents[0])
    thisScriptDir = os.path.dirname(os.path.realpath(__file__))


    os.chmod("{}/preprocessUQpy.py".format(thisScriptDir),  stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR | stat.S_IXOTH)
    # 1. Create the python script
    preprocessorCommand = "'{} {}/preprocessUQpy.py' --workflowInput {} --driverFile {} --runType {} --osType {}".format(python, thisScriptDir,
                                                                        inputFile,
                                                                        workflowDriver,
                                                                        runType,
                                                                        osType)

    res = subprocess.Popen(preprocessorCommand, shell=True).wait()
    with open("preprocessResult.txt", "w") as f:
        f.write(str(res))

        
    if runtype in ['runningLocal']:
        os.chmod(driverfile,  stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR | stat.S_IXOTH)
    
    
    
    # 2. Run the python script
    UQpycommand = python + " UQpyAnalysis.py"
        
    #Change permission of workflow driver
    st = os.stat(driverfile)
    os.chmod(driverfile, st.st_mode | stat.S_IEXEC)

    # copy the analysis Python script created by UQpy to the main working dir for the structure
    shutil.move("UQpyAnalysis.py", "../")
        
    # change dir to the main working dir for the structure
    os.chdir("../")
        
    if runtype in ['runningLocal']:
    
        print('running UQpy: ', UQpycommand)
        try:
            result = subprocess.check_output(UQpycommand, stderr=subprocess.STDOUT, shell=True)
            returncode = 0
        except subprocess.CalledProcessError as e:
            result = e.output
            returncode = e.returncode

if __name__ == '__main__':
    main()            
