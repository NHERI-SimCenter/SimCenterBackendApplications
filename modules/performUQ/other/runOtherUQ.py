# written: Michael Gardner @ UNR

import json
import os
import sys
import platform
import argparse
from configureAndRunUQ import configureAndRunUQ
from pathlib import Path

def main():
    # KEEP THIS FOR NOW--MAYBE BACKEND WILL BE UPDATED ACCEPT DIFFERENT ARGUMENTS...
    # parser = argparse.ArgumentParser(description='Generate workflow driver based on input configuration')
    # parser.add_argument('--mainWorkDir', '-m', required=True, help="Main work directory")
    # parser.add_argument('--tempWorkDir', '-t', required=True, help="Temporary work directory")
    # parser.add_argument('--runType', '-r', required=True, help="Type of run")
    # parser.add_argument('--inputFile', '-i', required=True, help="Input JSON file with configuration from UI")
    # Options for run type
    runTypeOptions=["runningLocal", "runningRemote"]
    
    # args = parser.parse_args()

    # workDirMain = args.mainWorkDir
    # workDirTemp = args.tempWorkDir
    # runType = args.runType
    # inputFile = args.inputFile

    parser = argparse.ArgumentParser()

    parser.add_argument("--workflowInput")
    parser.add_argument("--workflowOutput")
    parser.add_argument("--driverFile")
    parser.add_argument("--runType")

    args, unknowns = parser.parse_known_args()

    inputFile = args.workflowInput
    runType = args.runType
    workflowDriver = args.driverFile
    outputFile = args.workflowOutput

    cwd = os.getcwd()
    workDirTemp = cwd
    
    if runType not in runTypeOptions:
        raise ValueError("ERROR: Input run type has to be either local or remote")
    
    # change workdir to the templatedir
    # os.chdir(workDirTemp)
    # cwd = os.getcwd()
    
    # Open input file
    inputdata = {}
    with open(inputFile) as data_file:
        inputData = json.load(data_file)
    
    applicationsData = inputData["Applications"]

    # Get data to pass to UQ driver
    uqData = inputData["UQ"]
    simulationData = applicationsData["FEM"]
    randomVarsData = inputData["randomVariables"]
    demandParams = inputData["EDP"]
    localAppDir = inputData["localAppDir"]
    remoteAppDir = inputData["remoteAppDir"]

    # Run UQ based on data and selected UQ engine--if you need to preprocess files with custom delimiters, use preprocessUQ.py
    configureAndRunUQ(uqData, simulationData, randomVarsData, demandParams, workDirTemp, runType, localAppDir, remoteAppDir)

if __name__ == '__main__':
    main()
