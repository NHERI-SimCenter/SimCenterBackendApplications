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

####################################################################
def install(package):
    """Install all the requirements."""

    # Install all python libraries required
    # python3 -m pip3 install --user -r requirements.txt

    # if hasattr(pip, 'main'):
    #     pip.main(['install', package])
    # else:
    #     pip._internal.main(['install', package])

####################################################################
def main():
    """This is the main routine which controls the flow of program."""

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

####################################################################
if __name__ == "__main__":
    
    # Install the requirements
    # install('requirements.txt')
    
    # Call the main routine
    main()
