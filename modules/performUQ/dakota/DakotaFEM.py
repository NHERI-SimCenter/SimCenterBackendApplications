# import functions for Python 2.X support
from __future__ import division, print_function
import sys, os
if sys.version.startswith('2'): 
    range=xrange
    string_types = basestring
else:
    string_types = str

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

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
    parser.add_argument('--samples', default=None)
    parser.add_argument('--seed', default=None)
    parser.add_argument('--samples2', default=None)
    parser.add_argument('--seed2', default=None)
    parser.add_argument('--ismethod', default=None)
    parser.add_argument('--dataMethod', default=None)
    parser.add_argument('--dataMethod2', default=None)
    
    parser.add_argument('--type')
    parser.add_argument('--concurrency', default=None)
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

        concurrency = args.concurrency
    )

    runDakota = args.runType

    #Run Preprocess for Dakota
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    numRVs = preProcessDakota(bimName, evtName, samName, edpName, simName, driverFile, uqData)

    #Setting Workflow Driver Name
    workflowDriverName = 'workflow_driver'
    if platform.system() == 'Windows':
        workflowDriverName = 'workflow_driver.bat'

    #Create Template Directory and copy files
    templateDir = "templatedir"
    #if os.path.exists(templateDir):
    #    shutil.rmtree(templateDir)

    #os.mkdir(templateDir)
    st = os.stat(workflowDriverName)
    os.chmod(workflowDriverName, st.st_mode | stat.S_IEXEC)
    shutil.copy(workflowDriverName, templateDir)
    shutil.copy("{}/dpreproSimCenter".format(scriptDir), os.getcwd())
    shutil.copy(bimName, "bim.j")
    shutil.copy(evtName, "evt.j")
    exists = os.path.isfile(samName)
    if exists:
        shutil.copy(samName, "sam.j")

    shutil.copy(edpName, "edp.j")

    exists = os.path.isfile(simName)
    if exists:
        shutil.copy(simName, "sim.j")

    shutil.copy("dakota.in", "../")
    os.chdir("../")

    if runDakota == "run":

        dakotaCommand = "dakota -input dakota.in -output dakota.out -error dakota.err"
        print(dakotaCommand)
        try:
            result = subprocess.check_output(dakotaCommand, stderr=subprocess.STDOUT, shell=True)
            returncode = 0
        except subprocess.CalledProcessError as e:
            result = e.output
            returncode = e.returncode
        #result = result.decode(sys.stdout.encoding)
        print(result, returncode)

        #Postprocess Dakota results
        #postprocessCommand = '{}/postprocessDAKOTA {} {} {} {} dakotaTab.out'.format(
        #    scriptDir, numRVs, numSamples, bimName, edpName)

        #subprocess.Popen(postprocessCommand, shell=True).wait()

if __name__ == '__main__':

    main(sys.argv[1:])