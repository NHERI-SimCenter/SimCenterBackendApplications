# Modified by: Stevan Gavrilovic @ SimCenter, UC Berkeley
# Last revision: 09/2020

##########################################################################
#                       Relevant Publications                            #
##########################################################################

# Add relevant publications below
# [1]. Guan, X., Burton, H., and Thomas, S. (2020).
# “Python-based computational platform to automate seismic design,
# nonlinear structural model construction and analysis of steel moment
# resisting frames.” Engineering Structures. (Under Review)

import argparse, posixpath, ntpath, json

import sys
import time
import os
import sys
import shutil
import pathlib
import subprocess
import pickle

from global_variables import baseDirectory
from seismic_design import seismic_design
from global_variables import (
    SECTION_DATABASE,
    COLUMN_DATABASE,
    BEAM_DATABASE,
    RV_ARRAY,
)
from model_generation import model_generation


def main(BIM_file, EVENT_file, SAM_file, model_file, filePath, getRV):
    start_time = time.time()

    # Get the current directory
    workingDirectory = os.getcwd()

    rootSIM = {}

    # Try to open the BIM json
    with open(BIM_file, 'r', encoding='utf-8') as f:
        rootBIM = json.load(f)
    try:
        rootSIM = rootBIM['Modeling']
    except:
        raise ValueError('AutoSDA - structural information missing')

    # Extract the path for the directory containing the folder with the building data .csv files
    #    pathDataFolder = rootSIM['pathDataFolder']
    pathDataFolder = os.path.join(os.getcwd(), rootSIM['folderName'])

    #    pathDataFolder = workingDirectory + "/" + rootSIM['folderName']

    # Get the random variables from the input file
    try:
        rootRV = rootBIM['randomVariables']
    except:
        raise ValueError('AutoSDA - randomVariables section missing')

    # Populate the RV array with name/value pairs.
    # If a random variable is used here, the RV array will contain its current value
    for rv in rootRV:
        # Try to get the name and value of the random variable
        rvName = rv['name']
        curVal = rv['value']

        # Check if the current value a realization of a RV, i.e., is not a RV label
        # If so, then set the current value as the mean
        if 'RV' in str(curVal):
            curVal = float(rv['mean'])

        RV_ARRAY[rvName] = curVal

    # Count the starting time of the main program
    start_time = time.time()

    if getRV is False:
        # *********************** Design Starts Here *************************
        print('Starting seismic design')
        seismic_design(baseDirectory, pathDataFolder, workingDirectory)
        print('Seismic design complete')

        # ******************* Nonlinear Model Generation Starts Here ******
        # Nonlinear .tcl models are generated for EigenValue, Pushover, and Dynamic Analysis
        print('Generating nonlinear model')
        model_generation(baseDirectory, pathDataFolder, workingDirectory)

        # ******************* Perform Eigen Value Analysis ****************
        # print("Eigen Value Analysis for Building")
        # analysis_type = 'EigenValueAnalysis'
        # target_model = pathDataFolder + "/BuildingNonlinearModels/"+ analysis_type
        # os.chdir(target_model)
        # subprocess.Popen("OpenSees Model.tcl", shell=True).wait()
        #
        # ******************* Perform Nonlinear Pushover Analysis *********
        # print("Pushover Analysis for Building")
        # analysis_type = 'PushoverAnalysis'
        # target_model = pathDataFolder + "/BuildingNonlinearModels/" + analysis_type
        # os.chdir(target_model)
        # subprocess.Popen("OpenSees Model.tcl", shell=True).wait()

        print('The design and model construction has been accomplished.')

    end_time = time.time()
    print('Running time is: %s seconds' % round(end_time - start_time, 2))

    # Now create the SAM file for export
    root_SAM = {}

    root_SAM['mainScript'] = 'Model.tcl'
    root_SAM['type'] = 'OpenSeesInput'
    root_SAM['units'] = {
        'force': 'kips',
        'length': 'in',
        'temperature': 'C',
        'time': 'sec',
    }

    # Number of dimensions (KZ & AZ: changed to integer)
    root_SAM['ndm'] = 2

    # Number of degrees of freedom at each node (KZ & AZ: changed to integer)
    root_SAM['ndf'] = 3

    # Get the number of stories
    numStories = rootSIM['numStories']
    node_map = []

    # Using nodes on column #1 to calculate story drift
    # (1, i, 1, 0)      # Node tag at ground floor (different from those on upper stories)
    # (1, i, 1, 1)      # Node at bottom of current story
    # (1, i + 1, 1, 1)  # Node at top of current story
    for i in range(1, numStories + 2):
        nodeTagBot = 0
        if i == 1:
            # Node tag at ground floor is different from those on upper stories (1, i, 1, 0)
            nodeTagBot = 1010 + 100 * i
        else:
            # KZ & AZ: minor patch for story numbers greater than 10
            if i > 9:
                nodeTagBot = 10011 + 100 * i
            else:
                nodeTagBot = 1011 + 100 * i

        # Create the node and add it to the node mapping array
        node_entry = {}
        node_entry['node'] = nodeTagBot
        node_entry['cline'] = 'response'
        node_entry['floor'] = '{}'.format(i - 1)
        node_map.append(node_entry)

        ## KZ & AZ: Add centroid for roof drift
        node_entry_c = {}
        node_entry_c['node'] = nodeTagBot
        node_entry_c['cline'] = 'centroid'
        node_entry_c['floor'] = '{}'.format(i - 1)
        node_map.append(node_entry_c)

    root_SAM['NodeMapping'] = node_map
    root_SAM['numStory'] = numStories

    # Go back to the current directory before saving the SAM file
    os.chdir(workingDirectory)

    with open(SAM_file, 'w') as f:
        json.dump(root_SAM, f, indent=2)

    # Copy over the .tcl files of the building model into the working directory
    if getRV is False:
        pathToMainScriptFolder = (
            workingDirectory + '/BuildingNonlinearModels/DynamicAnalysis/'
        )

        if os.path.isdir(pathToMainScriptFolder):
            print(pathToMainScriptFolder)
            src_files = os.listdir(pathToMainScriptFolder)
            for file_name in src_files:
                full_file_name = os.path.join(pathToMainScriptFolder, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, workingDirectory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameAIM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameSAM')
    parser.add_argument('--fileName')
    parser.add_argument('--filePath')
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    sys.exit(
        main(
            args.filenameAIM,
            args.filenameEVENT,
            args.filenameSAM,
            args.fileName,
            args.filePath,
            args.getRV,
        )
    )
