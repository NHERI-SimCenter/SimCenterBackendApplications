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
#from GenUtilities import genUtilities # General utilities
#from OpenFOAM import solver # OpenFOAM class

####################################################################
# def headerOF(self,OFclass,location,filename):
#     '''
#     Method to create a header for the input dictionaries.

#     Variables
#     -----------
#         header: FileID for the file being created
#     '''

#     header = """/*--------------------------*- NHERI SimCenter -*----------------------------*\ 
# |       | H |
# |       | Y | HydroUQ: Water-based Natural Hazards Modeling Application
# |=======| D | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
# |       | R | Version: 1.00
# |       | O |
# \*---------------------------------------------------------------------------*/ 
# FoamFile
# {{
#     version   2.0;
#     format    ascii;
#     class     {};
#     location  "{}";
#     object    {};
# }}
# // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n""".format(OFclass,location,filename)

#     return header

####################################################################
# def controlDict(self):
#     '''
#     Method to create the control dictionary for the solver

#     Variables
#     ------------
#         fileID: FileID for the fvSchemes file
#         ofheader: Header for the Hydro-UQ input dictionary
#         startT: Start time of simulation
#         endT: End of simulation
#         deltaT: Time interval for simulation
#         writeT: Write interval
#         simtype: Type of simulation
#         solver: Solver used for CFD simulation
#     '''

    # Create the transportProperties file
    # fileID = open("system/controldict","w")

    # Get the turbulence model
    # hydroutil = genUtilities()

    # # Get required data
    # startT = ''.join(hydroutil.extract_element_from_json(data, ["Events","StartTime"]))
    # endT = ''.join(hydroutil.extract_element_from_json(data, ["Events","EndTime"]))
    # deltaT = ''.join(hydroutil.extract_element_from_json(data, ["Events","TimeInterval"]))
    # writeT = ''.join(hydroutil.extract_element_from_json(data, ["Events","WriteInterval"]))
    # simtype = ', '.join(hydroutil.extract_element_from_json(data, ["Events","SimulationType"]))
    # if(int(simtype) == 4):
    #     solver = "olaDyMFlow"
    # else:
    #     solver = "olaFlow"

    # # # Write the constants to the file
    # # var = np.array([['startT', startT], ['endT', endT], ['deltaT', deltaT],['deltaT2', 1], ['writeT', writeT], ['solver', '"'+solver+'"']])
    # # self.constvarfileOF(var,"ControlDict")

    # Write the dictionary file
    # ofheader = self.headerOF("dictionary","system","controlDict")
    # fileID.write(ofheader)
    # fileID.write('\napplication \t $solver;\n\n')
    # fileID.write('startFrom \t latestTime;\n\n')
    # fileID.write('startTime \t $startT;\n\n')
    # fileID.write('stopAt \t endTime;\n\n')
    # fileID.write('endTime \t $endT;\n\n')
    # fileID.write('deltaT \t $deltaT;\n\n')
    # fileID.write('writeControl \t adjustableRunTime;\n\n')
    # fileID.write('writeInterval \t $writeT;\n\n')
    # fileID.write('purgeWrite \t 0;\n\n')
    # fileID.write('writeFormat \t ascii;\n\n')
    # fileID.write('writePrecision \t t;\n\n')
    # fileID.write('writeCompression \t uncompressed;\n\n')
    # fileID.write('timeFormat \t general;\n\n')
    # fileID.write('timePrecision \t 6;\n\n')
    # fileID.write('runTimeModifiable \t yes;\n\n')
    # fileID.write('adjustTimeStep \t yes;\n\n')
    # fileID.write('maxCo \t 1.0;\n\n')
    # fileID.write('maxAlphaCo \t 1.0;\n\n')
    # fileID.write('maxDeltaT \t 1;\n\n')
    # fileID.write('libs\n(\n\t"libwaves.so"\n)\n')

    # Add post-processing stuff

    # Close the controlDict file
    # fileID.close()

####################################################################
# def main():
#     # Get the system argument
#     # Create the parser
#     # hydro_parser = argparse.ArgumentParser(description='Get the Dakota.json file')

#     # # Add the arguments
#     # hydro_parser.add_argument('-b',
#     #                    metavar='path',
#     #                    type=str,
#     #                    help='the path to dakota.json file',
#     #                    required=True)

#     # # Execute the parse_args() method
#     # args = hydro_parser.parse_args()

#     # # Open the JSON file
#     # # Load all the objects to the data variable
#     # # with open('dakota.json') as f:
#     # with open(args.b) as f:
#     #     data = json.load(f)

#     self.controlDict()

    

####################################################################
if __name__ == "__main__":
    
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
    fileID.write('writePrecision \t t;\n\n')
    fileID.write('writeCompression \t uncompressed;\n\n')
    fileID.write('timeFormat \t general;\n\n')
    fileID.write('timePrecision \t 6;\n\n')
    fileID.write('runTimeModifiable \t yes;\n\n')
    fileID.write('adjustTimeStep \t yes;\n\n')
    fileID.write('maxCo \t 1.0;\n\n')
    fileID.write('maxAlphaCo \t 1.0;\n\n')
    fileID.write('maxDeltaT \t 1;\n\n')
    fileID.write('libs\n(\n\t"libwaves.so"\n)\n')
    # Close the controlDict file
    fileID.close()