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
from zipfile36 import ZipFile

# Import user-defined classes
from GenUtilities import genUtilities # General utilities

####################################################################
def controlDict(data):
    '''
    Method to create the control dictionary for the solver

    Variables
    ------------
        fileID: FileID for the fvSchemes file
        ofheader: Header for the Hydro-UQ input dictionary
        startT: Start time of simulation
        endT: End of simulation
        deltaT: Time interval for simulation
        writeT: Write interval
        simtype: Type of simulation
        solver: Solver used for CFD simulation
    '''  

    header = """/*--------------------------*- NHERI SimCenter -*----------------------------*\ 
|       | H |
|       | Y | HydroUQ: Water-based Natural Hazards Modeling Application
|=======| D | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
|       | R | Version: 1.00
|       | O |
\*---------------------------------------------------------------------------*/ 
FoamFile
{
    version   2.0;
    format    ascii;
    class     dictionary;
    location  "system";
    object    controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""

    fileID = open("system/controlDict","w")
    fileID.write(header)
    fileID.write('#include\t"../constantsFile"\n\n')
    fileID.write('\napplication \t $solver;\n\n')
    fileID.write('startFrom \t latestTime;\n\n')
    fileID.write('startTime \t $startT;\n\n')
    fileID.write('stopAt \t endTime;\n\n')
    fileID.write('endTime \t $endT;\n\n')
    fileID.write('deltaT \t $deltaT;\n\n')
    fileID.write('writeControl \t adjustableRunTime;\n\n')
    fileID.write('writeInterval \t $writeT;\n\n')
    fileID.write('purgeWrite \t 0;\n\n')
    fileID.write('writeFormat \t ascii;\n\n')
    fileID.write('writePrecision \t 6;\n\n')
    fileID.write('writeCompression \t uncompressed;\n\n')
    fileID.write('timeFormat \t general;\n\n')
    fileID.write('timePrecision \t 6;\n\n')
    fileID.write('runTimeModifiable \t yes;\n\n')
    fileID.write('adjustTimeStep \t yes;\n\n')
    fileID.write('maxCo \t 1.0;\n\n')
    fileID.write('maxAlphaCo \t 1.0;\n\n')
    fileID.write('maxDeltaT \t 0.025;\n\n')
    fileID.write('functions\n{\n\t')
    fileID.write('buildingsForces\n\t{\n\t\t')
    fileID.write('type\tforces;\n\t\t')
    fileID.write('functionObjectLibs\t("libforces.so");\n\t\t')
    fileID.write('writeControl\ttimeStep;\n\t\t')
    fileID.write('writeInterval\t1;\n\t\t')
    fileID.write('patches\t("Right");\n\t\t') # This needs to be changed to Building
    fileID.write('rho\trhoInf;\n\t\t')
    fileID.write('log\ttrue;\n\t\t')
    fileID.write('rhoInf\t1;\n\t\t')
    fileID.write('CofR\t(0 0 0);\n\t\t')

    # Get the number of stories
    hydroutil = genUtilities() # General utilities
    stories = hydroutil.extract_element_from_json(data, ["GeneralInformation","stories"])

    fileID.write('binData\n\t\t{\n\t\t\t')
    fileID.write('nBin\t'+str(stories[0])+';\n\t\t\t')
    fileID.write('direction\t(1 0 0);\n\t\t\t')
    fileID.write('cumulative\tno;\n\t\t}\n\t}\n}')

    # Close the controlDict file
    fileID.close()  

####################################################################
if __name__ == "__main__":

    # Get the system argument
    # Create the parser
    hydro_parser = argparse.ArgumentParser(description='Get the Dakota.json file')

    # Add the arguments
    hydro_parser.add_argument('-b', metavar='path to input file', type=str, help='the path to dakota.json file', required=True)

    # Execute the parse_args() method
    args = hydro_parser.parse_args()

    # Open the JSON file
    # Load all the objects to the data variable
    # with open('dakota.json') as f:
    with open(args.b) as f:
       data = json.load(f)
    
    # Call function to create new control dict
    controlDict(data)