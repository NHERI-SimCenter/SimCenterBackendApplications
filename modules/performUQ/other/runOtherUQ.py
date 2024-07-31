# written: Michael Gardner @ UNR  # noqa: INP001, D100

import argparse
import json
import os

from configureAndRunUQ import configureAndRunUQ


def main():  # noqa: ANN201, D103
    # KEEP THIS FOR NOW--MAYBE BACKEND WILL BE UPDATED ACCEPT DIFFERENT ARGUMENTS...
    # parser = argparse.ArgumentParser(description='Generate workflow driver based on input configuration')
    # parser.add_argument('--mainWorkDir', '-m', required=True, help="Main work directory")
    # parser.add_argument('--tempWorkDir', '-t', required=True, help="Temporary work directory")
    # parser.add_argument('--runType', '-r', required=True, help="Type of run")
    # parser.add_argument('--inputFile', '-i', required=True, help="Input JSON file with configuration from UI")
    # Options for run type
    runTypeOptions = ['runningLocal', 'runningRemote']  # noqa: N806

    # args = parser.parse_args()

    # workDirMain = args.mainWorkDir
    # workDirTemp = args.tempWorkDir
    # runType = args.runType
    # inputFile = args.inputFile

    parser = argparse.ArgumentParser()

    parser.add_argument('--workflowInput')
    parser.add_argument('--workflowOutput')
    parser.add_argument('--driverFile')
    parser.add_argument('--runType')

    args, unknowns = parser.parse_known_args()

    inputFile = args.workflowInput  # noqa: N806
    runType = args.runType  # noqa: N806
    workflowDriver = args.driverFile  # noqa: N806, F841
    outputFile = args.workflowOutput  # noqa: N806, F841

    cwd = os.getcwd()  # noqa: PTH109
    workDirTemp = cwd  # noqa: N806

    if runType not in runTypeOptions:
        raise ValueError('ERROR: Input run type has to be either local or remote')  # noqa: EM101, TRY003

    # change workdir to the templatedir
    # os.chdir(workDirTemp)
    # cwd = os.getcwd()

    # Open input file
    inputdata = {}  # noqa: F841
    with open(inputFile) as data_file:  # noqa: PTH123
        inputData = json.load(data_file)  # noqa: N806

    applicationsData = inputData['Applications']  # noqa: N806

    # Get data to pass to UQ driver
    uqData = inputData['UQ']  # noqa: N806
    simulationData = applicationsData['FEM']  # noqa: N806
    randomVarsData = inputData['randomVariables']  # noqa: N806
    demandParams = inputData['EDP']  # noqa: N806
    localAppDir = inputData['localAppDir']  # noqa: N806
    remoteAppDir = inputData['remoteAppDir']  # noqa: N806

    # Run UQ based on data and selected UQ engine--if you need to preprocess files with custom delimiters, use preprocessUQ.py
    configureAndRunUQ(
        uqData,
        simulationData,
        randomVarsData,
        demandParams,
        workDirTemp,
        runType,
        localAppDir,
        remoteAppDir,
    )


if __name__ == '__main__':
    main()
