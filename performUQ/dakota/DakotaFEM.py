# import functions for Python 2.X support
from __future__ import division, print_function
import sys
if sys.version.startswith('2'): 
    range=xrange
    string_types = basestring
else:
    string_types = str

import os
import platform
import shutil
import subprocess
import stat
import argparse
from preprocessJSON import preProcessDakota

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
parser.add_argument('--method')
parser.add_argument('--samples')
parser.add_argument('--seed')
parser.add_argument('--type')
parser.add_argument('--concurrency', default=None)
parser.add_argument('--runType')
args = parser.parse_args()

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
    seed = args.seed,
    concurrency = args.concurrency
)

runDakota = args.runType

#Run Preprocess for Dakota
scriptDir = os.path.dirname(os.path.realpath(__file__))
# preProcessArgs = ["python", "{}/preprocessJSON.py".format(scriptDir), bimName, evtName,\
# samName, edpName, lossName, simName, driverFile, scriptDir, bldgName]
# subprocess.call(preProcessArgs)
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
    subprocess.Popen(dakotaCommand, stderr=subprocess.STDOUT, shell=True).wait()

    #Postprocess Dakota results
    #postprocessCommand = '{}/postprocessDAKOTA {} {} {} {} dakotaTab.out'.format(
    #    scriptDir, numRVs, numSamples, bimName, edpName)

    #subprocess.Popen(postprocessCommand, shell=True).wait()
