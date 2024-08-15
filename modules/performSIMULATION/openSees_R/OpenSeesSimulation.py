import os
import subprocess
import sys

inputArgs = sys.argv

# set filenames
bimName = sys.argv[2]
samName = sys.argv[4]
evtName = sys.argv[6]
edpName = sys.argv[8]
simName = sys.argv[10]

scriptDir = os.path.dirname(os.path.realpath(__file__))

# If requesting random variables run getUncertainty
# Otherwise, Run Opensees
if ('-getRV' in inputArgs) or ('--getRV' in inputArgs):
    getUncertaintyCommand = (
        f'"{scriptDir}/getUncertainty" {bimName} {samName} {evtName} {simName}'
    )
    subprocess.Popen(args=getUncertaintyCommand, shell=True).wait()
else:
    # Run preprocessor
    preprocessorCommand = f'"{scriptDir}/mainPreprocessor" {bimName} {samName} {evtName} {edpName} example.tcl'
    subprocess.Popen(preprocessorCommand, shell=True).wait()

    # Run OpenSees
    subprocess.Popen('OpenSees example.tcl', shell=True).wait()

    # Run postprocessor
    postprocessorCommand = (
        f'"{scriptDir}/mainPostprocessor" {bimName} {samName} {evtName} {edpName}'
    )
    subprocess.Popen(postprocessorCommand, shell=True).wait()
