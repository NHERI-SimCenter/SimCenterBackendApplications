import os  # noqa: INP001, D100
import subprocess
import sys

inputArgs = sys.argv  # noqa: N816

# set filenames
bimName = sys.argv[2]  # noqa: N816
samName = sys.argv[4]  # noqa: N816
evtName = sys.argv[6]  # noqa: N816
edpName = sys.argv[8]  # noqa: N816
simName = sys.argv[10]  # noqa: N816

scriptDir = os.path.dirname(os.path.realpath(__file__))  # noqa: PTH120, N816

# If requesting random variables run getUncertainty
# Otherwise, Run Opensees
if ('-getRV' in inputArgs) or ('--getRV' in inputArgs):
    getUncertaintyCommand = (  # noqa: N816
        f'"{scriptDir}/getUncertainty" {bimName} {samName} {evtName} {simName}'
    )
    subprocess.Popen(args=getUncertaintyCommand, shell=True).wait()
else:
    # Run preprocessor
    preprocessorCommand = f'"{scriptDir}/mainPreprocessor" {bimName} {samName} {evtName} {edpName} example.tcl'  # noqa: N816
    subprocess.Popen(preprocessorCommand, shell=True).wait()  # noqa: S602

    # Run OpenSees
    subprocess.Popen('OpenSees example.tcl', shell=True).wait()  # noqa: S602, S607

    # Run postprocessor
    postprocessorCommand = (  # noqa: N816
        f'"{scriptDir}/mainPostprocessor" {bimName} {samName} {evtName} {edpName}'
    )
    subprocess.Popen(postprocessorCommand, shell=True).wait()  # noqa: S602
