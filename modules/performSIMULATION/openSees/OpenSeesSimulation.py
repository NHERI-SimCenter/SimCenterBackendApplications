import sys
import os
import subprocess

def main(args):

    # set filenames
    aimName = args[1]
    samName = args[3]
    evtName = args[5]
    edpName = args[7]
    simName = args[9]

    # remove path to AIM file, so recorders are not messed up
    #      .. AIM file ro be read is in current dir (copy elsewhere)
    aimName = os.path.basename(aimName)

    scriptDir = os.path.dirname(os.path.realpath(__file__))

    #If requesting random variables run getUncertainty
    #Otherwise, Run Opensees 
    if "--getRV" in args:
        getUncertaintyCommand = '"{}/OpenSeesPreprocessor" {} {} {} {}'.format(scriptDir, aimName, samName, evtName, simName)
        subprocess.Popen(getUncertaintyCommand, shell=True).wait()
    else:
        #Run preprocessor
        preprocessorCommand = '"{}/OpenSeesPreprocessor" {} {} {} {} {} example.tcl'.format(scriptDir, aimName, samName, evtName, edpName, simName)
        subprocess.Popen(preprocessorCommand, shell=True).wait()        

        #Run OpenSees
        subprocess.Popen("OpenSees example.tcl", shell=True).wait()

        #Run postprocessor
        postprocessorCommand = '"{}/OpenSeesPostprocessor" {} {} {} {}'.format(scriptDir, aimName, samName, evtName, edpName)
        subprocess.Popen(postprocessorCommand, shell=True).wait()

if __name__ == '__main__':

    main(sys.argv[1:])
