# Modified by: Stevan Gavrilovic @ SimCenter, UC Berkeley  # noqa: INP001, D100
# Last revision: 09/2020

##########################################################################
#                       Relevant Publications                            #
##########################################################################

# Add relevant publications below
# [1]. Guan, X., Burton, H., and Thomas, S. (2020).
# “Python-based computational platform to automate seismic design,
# nonlinear structural model construction and analysis of steel moment
# resisting frames.” Engineering Structures. (Under Review)

import argparse
import json
import os
import shutil
import sys
import time

from global_variables import (
    RV_ARRAY,
    baseDirectory,
)
from model_generation import model_generation
from seismic_design import seismic_design


def main(BIM_file, EVENT_file, SAM_file, model_file, filePath, getRV):  # noqa: ANN001, ANN201, ARG001, C901, N803, D103, PLR0912, PLR0913, PLR0915
    start_time = time.time()

    # Get the current directory
    workingDirectory = os.getcwd()  # noqa: PTH109, N806

    rootSIM = {}  # noqa: N806

    # Try to open the BIM json
    with open(BIM_file, encoding='utf-8') as f:  # noqa: PTH123
        rootBIM = json.load(f)  # noqa: N806
    try:
        rootSIM = rootBIM['Modeling']  # noqa: N806
    except:  # noqa: E722
        raise ValueError('AutoSDA - structural information missing')  # noqa: B904, EM101, TRY003

    # Extract the path for the directory containing the folder with the building data .csv files  # noqa: E501
    #    pathDataFolder = rootSIM['pathDataFolder']  # noqa: ERA001
    pathDataFolder = os.path.join(os.getcwd(), rootSIM['folderName'])  # noqa: PTH109, PTH118, N806

    #    pathDataFolder = workingDirectory + "/" + rootSIM['folderName']  # noqa: ERA001

    # Get the random variables from the input file
    try:
        rootRV = rootBIM['randomVariables']  # noqa: N806
    except:  # noqa: E722
        raise ValueError('AutoSDA - randomVariables section missing')  # noqa: B904, EM101, TRY003

    # Populate the RV array with name/value pairs.
    # If a random variable is used here, the RV array will contain its current value
    for rv in rootRV:
        # Try to get the name and value of the random variable
        rvName = rv['name']  # noqa: N806
        curVal = rv['value']  # noqa: N806

        # Check if the current value a realization of a RV, i.e., is not a RV label
        # If so, then set the current value as the mean
        if 'RV' in str(curVal):
            curVal = float(rv['mean'])  # noqa: N806

        RV_ARRAY[rvName] = curVal

    # Count the starting time of the main program
    start_time = time.time()

    if getRV is False:
        # *********************** Design Starts Here *************************
        print('Starting seismic design')  # noqa: T201
        seismic_design(baseDirectory, pathDataFolder, workingDirectory)
        print('Seismic design complete')  # noqa: T201

        # ******************* Nonlinear Model Generation Starts Here ******
        # Nonlinear .tcl models are generated for EigenValue, Pushover, and Dynamic Analysis  # noqa: E501
        print('Generating nonlinear model')  # noqa: T201
        model_generation(baseDirectory, pathDataFolder, workingDirectory)

        # ******************* Perform Eigen Value Analysis ****************
        # print("Eigen Value Analysis for Building")  # noqa: ERA001
        # analysis_type = 'EigenValueAnalysis'  # noqa: ERA001
        # target_model = pathDataFolder + "/BuildingNonlinearModels/"+ analysis_type  # noqa: ERA001
        # os.chdir(target_model)  # noqa: ERA001
        # subprocess.Popen("OpenSees Model.tcl", shell=True).wait()  # noqa: ERA001
        #
        # ******************* Perform Nonlinear Pushover Analysis *********
        # print("Pushover Analysis for Building")  # noqa: ERA001
        # analysis_type = 'PushoverAnalysis'  # noqa: ERA001
        # target_model = pathDataFolder + "/BuildingNonlinearModels/" + analysis_type  # noqa: ERA001
        # os.chdir(target_model)  # noqa: ERA001
        # subprocess.Popen("OpenSees Model.tcl", shell=True).wait()  # noqa: ERA001

        print('The design and model construction has been accomplished.')  # noqa: T201

    end_time = time.time()
    print('Running time is: %s seconds' % round(end_time - start_time, 2))  # noqa: T201, UP031

    # Now create the SAM file for export
    root_SAM = {}  # noqa: N806

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
    numStories = rootSIM['numStories']  # noqa: N806
    node_map = []

    # Using nodes on column #1 to calculate story drift
    # (1, i, 1, 0)      # Node tag at ground floor (different from those on upper stories)  # noqa: ERA001, E501
    # (1, i, 1, 1)      # Node at bottom of current story  # noqa: ERA001
    # (1, i + 1, 1, 1)  # Node at top of current story  # noqa: ERA001
    for i in range(1, numStories + 2):
        nodeTagBot = 0  # noqa: N806
        if i == 1:
            # Node tag at ground floor is different from those on upper stories (1, i, 1, 0)  # noqa: E501
            nodeTagBot = 1010 + 100 * i  # noqa: N806
        elif i > 9:  # noqa: PLR2004
            nodeTagBot = 10011 + 100 * i  # noqa: N806
        else:
            nodeTagBot = 1011 + 100 * i  # noqa: N806

        # Create the node and add it to the node mapping array
        node_entry = {}
        node_entry['node'] = nodeTagBot
        node_entry['cline'] = 'response'
        node_entry['floor'] = f'{i - 1}'
        node_map.append(node_entry)

        ## KZ & AZ: Add centroid for roof drift
        node_entry_c = {}
        node_entry_c['node'] = nodeTagBot
        node_entry_c['cline'] = 'centroid'
        node_entry_c['floor'] = f'{i - 1}'
        node_map.append(node_entry_c)

    root_SAM['NodeMapping'] = node_map
    root_SAM['numStory'] = numStories

    # Go back to the current directory before saving the SAM file
    os.chdir(workingDirectory)

    with open(SAM_file, 'w') as f:  # noqa: PTH123
        json.dump(root_SAM, f, indent=2)

    # Copy over the .tcl files of the building model into the working directory
    if getRV is False:
        pathToMainScriptFolder = (  # noqa: N806
            workingDirectory + '/BuildingNonlinearModels/DynamicAnalysis/'
        )

        if os.path.isdir(pathToMainScriptFolder):  # noqa: PTH112
            print(pathToMainScriptFolder)  # noqa: T201
            src_files = os.listdir(pathToMainScriptFolder)
            for file_name in src_files:
                full_file_name = os.path.join(pathToMainScriptFolder, file_name)  # noqa: PTH118
                if os.path.isfile(full_file_name):  # noqa: PTH113
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
