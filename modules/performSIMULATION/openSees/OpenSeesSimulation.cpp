
int main(int argc, char **argv) {
  char *bimName = argv[2];
  char *samName = argv[4];
  char *evtName = argv[6];
  char *edpName = argv[8];
  char *simName = argv[10];
  
  if (argc == 12) {

  } else {

  }
    scriptDir = os.path.dirname(os.path.realpath(__file__))

    #If requesting random variables run getUncertainty
    #Otherwise, Run Opensees 
    if "--getRV" in args:
        getUncertaintyCommand = '"{}/OpenSeesPreprocessor" {} {} {} {} 1> ops.out 2>&1'.format(scriptDir, bimName, samName, evtName, simName)
        subprocess.Popen(getUncertaintyCommand, shell=True).wait()
    else:
        #Run preprocessor
        preprocessorCommand = '"{}/OpenSeesPreprocessor" {} {} {} {} {} example.tcl 1> ops.out 2>&1'.format(scriptDir, bimName, samName, evtName, edpName, simName)
        subprocess.Popen(preprocessorCommand, shell=True).wait()
        
        #Run OpenSees
        subprocess.Popen("OpenSees example.tcl", shell=True).wait()

        #Run postprocessor
        postprocessorCommand = '"{}/OpenSeesPostprocessor" {} {} {} {}'.format(scriptDir, bimName, samName, evtName, edpName)
        subprocess.Popen(postprocessorCommand, shell=True).wait()

if __name__ == '__main__':

    main(sys.argv[1:])
