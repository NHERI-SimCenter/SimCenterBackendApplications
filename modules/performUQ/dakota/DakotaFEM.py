# import functions for Python 2.X support
from __future__ import division, print_function
import sys, os
if sys.version.startswith('2'): 
    range=xrange
    string_types = basestring
else:
    string_types = str

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import platform
import shutil
import subprocess
import stat
import argparse
from preprocessJSON import preProcessDakota

def main(args):

    #First we need to set the path and environment
    home = os.path.expanduser('~')
    env = os.environ
    if os.getenv("PEGASUS_WF_UUID") is not None:
        print("Pegasus job detected - Pegasus will set up the env")
    elif platform.system() == 'Darwin':
        env["PATH"] = env["PATH"] + ':{}/bin'.format(home)
        env["PATH"] = env["PATH"] + ':{}/dakota/bin'.format(home)
    elif platform.system() == 'Linux':
        env["PATH"] = env["PATH"] + ':{}/bin'.format(home)
        env["PATH"] = env["PATH"] + ':{}/dakota/dakota-6.5/bin'.format(home)
    elif platform.system() == 'Windows':
        pass
    else:
        print("PLATFORM {} NOT RECOGNIZED".format(platform.system))

    parser = argparse.ArgumentParser()

    parser.add_argument('--filenameBIM')
    parser.add_argument('--filenameSAM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameEDP')
    parser.add_argument('--filenameSIM')
    
    parser.add_argument('--driverFile')
    
    parser.add_argument('--method', default="LHS")
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=np.random.randint(1,1000))
    parser.add_argument('--samples2', type=int, default=None)
    parser.add_argument('--seed2', type=int, default=None)
    parser.add_argument('--ismethod', default=None)
    parser.add_argument('--dataMethod', default=None)
    parser.add_argument('--dataMethod2', default=None)
    
    parser.add_argument('--type')
    parser.add_argument('--concurrency', type=int, default=None)
    parser.add_argument('--keepSamples', default="True")
    parser.add_argument('--runType')
    
    args,unknowns = parser.parse_known_args()

    #Reading input arguments
    bimName = args.filenameBIM
    samName = args.filenameSAM
    evtName = args.filenameEVENT
    edpName = args.filenameEDP
    simName = args.filenameSIM
    driverFile = args.driverFile

    uqData = dict(
        method = args.method,
        
        samples = args.samples,
        samples2 = args.samples2,
        seed = args.seed,
        seed2 = args.seed2,
        ismethod = args.ismethod,
        dataMethod = args.dataMethod,
        dataMethod2 = args.dataMethod2,

        concurrency = args.concurrency,
        keepSamples = args.keepSamples not in ["False", 'False', "false", 'false', False]
    )

    runDakota = args.runType

    #Run Preprocess for Dakota
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    numRVs = preProcessDakota(bimName, evtName, samName, edpName, simName, driverFile, runDakota, uqData)

    #Setting Workflow Driver Name
    workflowDriverName = 'workflow_driver'
    if ((platform.system() == 'Windows') and (runDakota == 'run')):
        workflowDriverName = 'workflow_driver.bat'

    #Create Template Directory and copy files
    st = os.stat(workflowDriverName)
    os.chmod(workflowDriverName, st.st_mode | stat.S_IEXEC)
    #shutil.copy(workflowDriverName, "templatedir")
    shutil.copy("{}/dpreproSimCenter".format(scriptDir), os.getcwd())
    shutil.move(bimName, "bim.j")
    shutil.move(evtName, "evt.j")
    if os.path.isfile(samName): shutil.move(samName, "sam.j")
    shutil.move(edpName, "edp.j")
    #if os.path.isfile(simName): shutil.move(simName, "sim.j")

    # copy the dakota input file to the main working dir for the structure
    shutil.move("dakota.in", "../")

    # change dir to the main working dir for the structure
    os.chdir("../")

    if runDakota == "run":

        dakotaCommand = "dakota -input dakota.in -output dakota.out -error dakota.err"
        print('running Dakota: ', dakotaCommand)
        try:
            result = subprocess.check_output(dakotaCommand, stderr=subprocess.STDOUT, shell=True)
            returncode = 0
        except subprocess.CalledProcessError as e:
            result = e.output
            returncode = e.returncode
        result = result.decode(sys.stdout.encoding)

        print(result, returncode)

if __name__ == '__main__':

    main(sys.argv[1:])