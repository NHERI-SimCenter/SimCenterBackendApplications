############################################################################
# This python routine forms the backend where all the files are generated
# to run the Hydro-UQ simulation
#############################################################################
import json
import datetime
import os
import shutil
import sys
import numpy as np
import argparse
import pip

# Import user-defined classes
from GenUtilities import genUtilities # General utilities
from OpenFOAM import solver # OpenFOAM class

####################################################################
def install(package):
    '''
    Install all the requirements.
    '''

    # Install all python libraries required
    # python3 -m pip3 install --user -r requirements.txt

    # if hasattr(pip, 'main'):
    #     pip.main(['install', package])
    # else:
    #     pip._internal.main(['install', package])

####################################################################
def main():
    '''
    This is the main routine which controls the flow of program.

    Objects:
        hydro_parser: Parse CLI arguments
        hydroutil: General utilities
        hydrosolver: Solver related file generation

    Variables:
        projname: Name of the project as given by the user
        logID: Integer ID for the log file
    '''

    # Get the system argument
    # Create the parser
    hydro_parser = argparse.ArgumentParser(description='Get the Dakota.json file')

    # Add the arguments
    hydro_parser.add_argument('-b',
                       metavar='path',
                       type=str,
                       help='the path to dakota.json file',
                       required=True)

    # Execute the parse_args() method
    args = hydro_parser.parse_args()

    # Open the JSON file
    # Load all the objects to the data variable
    # with open('dakota.json') as f:
    with open(args.b) as f:
        data = json.load(f)

    # Create the objects
    hydroutil = genUtilities() # General utilities
    hydrosolver = solver() # Solver object

    #***********************************
    # HYDRO-UQ LOG FILE: INITIALIZE
    #***********************************
    # Get the project name
    projname = hydroutil.extract_element_from_json(data, ["Events","ProjectName"])
    projname = ', '.join(projname)
    logID = 0

    # Initialize the log
    hydroutil.hydrolog(projname)

    # Start the log file with header and time and date
    logfiletext = hydroutil.general_header()
    hydroutil.flog.write(logfiletext)
    logID += 1
    hydroutil.flog.write('%d (%s): This log has started.\n' % (logID,datetime.datetime.now()))

    #***********************************
    # REQUIRED DIRECTORIES
    #***********************************
    # Create the OpenFOAM directories
    foldwrite = hydrosolver.dircreate()
    logID += 1
    hydroutil.flog.write('%d (%s): Following solver directories have been created: %s\n' % (logID,datetime.datetime.now(),', '.join(foldwrite)))

    #***********************************
    # SUPPLEMENTARY SOLVER SPECIFIC FILES
    #***********************************
    fileswrite = hydrosolver.filecreate(data)
    logID += 1
    hydroutil.flog.write('%d (%s): Following required files have been created: %s\n' % (logID,datetime.datetime.now(),', '.join(fileswrite)))

    #***********************************
    # MATERIAL MODEL RELATED FILES
    #***********************************
    fileswrite = hydrosolver.matmodel(data)
    logID += 1
    hydroutil.flog.write('%d (%s): Following material-related files have been created: %s\n' % (logID,datetime.datetime.now(),', '.join(fileswrite)))

    #***********************************
    # SIMULATION CONTROL RELATED FILES
    #***********************************
    fileswrite = hydrosolver.solvecontrol(data)
    logID += 1
    hydroutil.flog.write('%d (%s): Following solver-control related files have been created: %s\n' % (logID,datetime.datetime.now(),', '.join(fileswrite)))

    #***********************************
    # PARALLELIZATION CONTROL RELATED FILES
    #***********************************
    fileswrite = hydrosolver.parallel(data)
    logID += 1
    hydroutil.flog.write('%d (%s): Following parallel-compute related files have been created: %s\n' % (logID,datetime.datetime.now(),', '.join(fileswrite)))

    #***********************************
    # GEOMETRY RELATED FILES
    # Call this only if we are using Hydro mesher
    #***********************************
    mesher = hydroutil.extract_element_from_json(data, ["Events","MeshType"])
    if int(mesher[0]) == 0:
        fileswrite = hydrosolver.geometry(data)
        logID += 1
        hydroutil.flog.write('%d (%s): Following geometry-related files have been created: %s\n' % (logID,datetime.datetime.now(),', '.join(fileswrite)))

    else:
        hydroutil.flog.write('%d (%s): No gemoetric files have not been created since the user is providing the mesh\n' % (logID,datetime.datetime.now()))

    #***********************************
    # MESHING RELATED FILES
    # Call this only if we are using Hydro mesher
    #***********************************
    if int(mesher[0]) == 0:
        fileswrite = hydrosolver.meshing(data)
        logID += 1
        hydroutil.flog.write('%d (%s): Following meshing-related files have been created: %s\n' % (logID,datetime.datetime.now(),', '.join(fileswrite)))

    elif int(mesher[0]) == 2:

        fileswrite = np.array([])
        # Check if OF mesh dictionary exists then move to system
        if os.path.isfile("templateDir/blockMeshDict"):
            shutil.move("templateDir/blockMeshDict", "system/blockMeshDict")
            fileswrite = np.append(filewritten,['blockMeshDict'])

        if os.path.isfile("templateDir/surfaceFeatureExtractDict"):
            shutil.move("templateDir/surfaceFeatureExtractDict", "system/surfaceFeatureExtractDict")
            fileswrite = np.append(filewritten,['surfaceFeatureExtractDict'])

        if os.path.isfile("templateDir/snappyHexMeshDict"):
            shutil.move("templateDir/snappyHexMeshDict", "system/snappyHexMeshDict")
            fileswrite = np.append(filewritten,['snappyHexMeshDict'])
        
        if fileswrite.size != 0:
            # Confirm the copy
            hydroutil.flog.write('%d (%s): Following mesh dictionaries provided by the user have been copied to system folder: %s\n' % (logID,datetime.datetime.now(),', '.join(fileswrite)))

        else:
            hydroutil.flog.write('%d (%s): WARNING: Mesh dictionaries not found\n' % (logID,datetime.datetime.now()))

    else:
        hydroutil.flog.write('%d (%s): Mesh files are provided by the user and thus mesh-dictionary creation has been skipped\n' % (logID,datetime.datetime.now()))


####################################################################
if __name__ == "__main__":
    
    # Install the requirements
    # install('requirements.txt')
    
    # Call the main routine
    main()
