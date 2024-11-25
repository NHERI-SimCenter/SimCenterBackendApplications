#!/usr/bin/env python3  # noqa: D100

import argparse
import json
import os
import re
import subprocess
import sys
from fractions import Fraction

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class FloorForces:  # noqa: D101
    def __init__(self):
        self.X = [0]
        self.Y = [0]
        self.Z = [0]


def directionToDof(direction):  # noqa: N802
    """Converts direction to degree of freedom"""  # noqa: D400, D401
    directionMap = {'X': 1, 'Y': 2, 'Z': 3}  # noqa: N806

    return directionMap[direction]


def addFloorForceToEvent(  # noqa: N802
    timeSeriesArray,  # noqa: N803
    patternsArray,  # noqa: N803
    force,
    direction,
    floor,
    dT,  # noqa: N803
):
    """Add force (one component) time series and pattern in the event file"""  # noqa: D400
    seriesName = 'HydroForceSeries_' + str(floor) + direction  # noqa: N806
    timeSeries = {'name': seriesName, 'dT': dT, 'type': 'Value', 'data': force}  # noqa: N806

    timeSeriesArray.append(timeSeries)
    patternName = 'HydroForcePattern_' + str(floor) + direction  # noqa: N806
    pattern = {
        'name': patternName,
        'timeSeries': seriesName,
        'type': 'HydroFloorLoad',
        'floor': str(floor),
        'dof': directionToDof(direction),
    }

    patternsArray.append(pattern)


def addFloorForceToEvent(patternsArray, force, direction, floor):  # noqa: ARG001, N802, N803, F811
    """Add force (one component) time series and pattern in the event file"""  # noqa: D400
    seriesName = 'HydroForceSeries_' + str(floor) + direction  # noqa: N806
    patternName = 'HydroForcePattern_' + str(floor) + direction  # noqa: N806
    pattern = {
        'name': patternName,
        'timeSeries': seriesName,
        'type': 'HydroFloorLoad',
        'floor': str(floor),
        'dof': directionToDof(direction),
    }

    patternsArray.append(pattern)


def addFloorPressure(pressureArray, floor):  # noqa: N802, N803
    """Add floor pressure in the event file"""  # noqa: D400
    floorPressure = {'story': str(floor), 'pressure': [0.0, 0.0]}  # noqa: N806

    pressureArray.append(floorPressure)


def writeEVENT(forces, eventFilePath):  # noqa: N802, N803
    """This method writes the EVENT.json file"""  # noqa: D400, D401, D404
    timeSeriesArray = []  # noqa: N806, F841
    patternsArray = []  # noqa: N806
    pressureArray = []  # noqa: N806
    hydroEventJson = {  # noqa: N806
        'type': 'Hydro',  # Using HydroUQ
        'subtype': 'MPMEvent',  # Using ClaymoreUW Material Point Method
        # "timeSeries": [], # From GeoClawOpenFOAM
        'pattern': patternsArray,
        'pressure': pressureArray,
        # "dT": deltaT, # From GeoClawOpenFOAM
        'numSteps': len(forces[0].X),
        'units': {'force': 'Newton', 'length': 'Meter', 'time': 'Sec'},
    }

    # Creating the event dictionary that will be used to export the EVENT json file
    eventDict = {'randomVariables': [], 'Events': [hydroEventJson]}  # noqa: N806

    # Adding floor forces
    for floorForces in forces:  # noqa: N806
        floor = forces.index(floorForces) + 1
        addFloorForceToEvent(patternsArray, floorForces.X, 'X', floor)
        addFloorForceToEvent(patternsArray, floorForces.Y, 'Y', floor)
        # addFloorPressure(pressureArray, floor) # From GeoClawOpenFOAM

    with open(eventFilePath, 'w', encoding='utf-8') as eventsFile:  # noqa: PTH123, N806
        json.dump(eventDict, eventsFile)


def GetFloorsCount(BIMFilePath):  # noqa: N802, N803, D103
    with open(BIMFilePath, encoding='utf-8') as BIMFile:  # noqa: PTH123, N806
        bim = json.load(BIMFile)
    return int(bim['GeneralInformation']['stories'])

def GetExecutableFile(BIMFilePath):  # noqa: N802, N803, D103
    filePath = BIMFilePath  # noqa: N806
    with open(filePath, encoding='utf-8') as file:  # noqa: PTH123
        evt = json.load(file)
    file.close  # noqa: B018
    
    executableNameKey = 'executableFile'
    executablePathKey = executableNameKey + 'Path'
    
    for event in evt['Events']:
        executableName = event[executableNameKey]
        executablePath = event[executablePathKey]
        
        return os.path.join(executablePath, executableName)
    
    defaultExecutablePath = f'{os.path.realpath(os.path.dirname(__file__))}'  # noqa: ISC003, PTH120
    defaultExecutableName = 'osu_lwf.exe'
    return defaultExecutablePath + defaultExecutableName

def GetSceneFile(BIMFilePath):  # noqa: N802, N803, D103
    filePath = BIMFilePath  # noqa: N806
    with open(filePath, encoding='utf-8') as file:  # noqa: PTH123
        evt = json.load(file)
    file.close  # noqa: B018

    fileNameKey = 'configFile' 
    filePathKey = fileNameKey + 'Path'
    
    for event in evt['Events']:
        fileName = event[fileNameKey]
        filePath = event[filePathKey]
        
        return os.path.join(filePath, fileName)
    
    defaultScriptPath = f'{os.path.realpath(os.path.dirname(__file__))}'  # noqa: ISC003, PTH120
    defaultScriptName = 'scene.json'
    return defaultScriptPath + defaultScriptName 


def GetTimer(BIMFilePath):  # noqa: N802, N803, D103
    filePath = BIMFilePath  # noqa: N806
    with open(filePath, encoding='utf-8') as file:  # noqa: PTH123
        evt = json.load(file)
    file.close  # noqa: B018
    
    timerKey = 'maxMinutes'

    for event in evt['Events']:
        timer = event[timerKey]
        maxSeconds = timer * 60
        return maxSeconds
    
    return 0


def main():  # noqa: D103
    """
    Entry point to generate event file using MPMEvent.
    """
    return 0


if __name__ == '__main__':
    """
    Entry point to generate event file using Stochastic Waves
    """
    # CLI parser
    parser = argparse.ArgumentParser(
        description='Get sample EVENT file produced by StochasticWave'
    )
    parser.add_argument(
        '-b',
        '--filenameAIM',
        help='BIM File',
        required=True,
        default='AIM.json',
    )
    parser.add_argument(
        '-e',
        '--filenameEVENT',
        help='Event File',
        required=True,
        default='EVENT.json',
    )
    parser.add_argument('--getRV', help='getRV', required=False, action='store_true')
    # parser.add_argument('--filenameSAM', default=None)

    # parsing arguments
    arguments, unknowns = parser.parse_known_args()

    # import subprocess

    # Get json of filenameAIM
    executableName = GetExecutableFile(arguments.filenameAIM)  # noqa: N816
    scriptName = GetSceneFile(arguments.filenameAIM)  # noqa: N816        
    maxSeconds = GetTimer(arguments.filenameAIM)  # noqa: N816
    
    if arguments.getRV == True:  # noqa: E712
        print('RVs requested')  # noqa: T201
        # Read the number of floors
        floorsCount = GetFloorsCount(arguments.filenameAIM)  # noqa: N816
        filenameEVENT = arguments.filenameEVENT  # noqa: N816

        
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                'timeout',
                str(maxSeconds),
                executableName,
                '-f',
                scriptName,
                # f'{os.path.realpath(os.path.dirname(__file__))}'  # noqa: ISC003, PTH120
                # + '/taichi_script.py',
            ],
            stdout=subprocess.PIPE,
            check=False,
        )

        forces = []
        for i in range(floorsCount):
            forces.append(FloorForces(recorderID=(i + 1)))  # noqa: PERF401

        # write the event file
        writeEVENT(forces, filenameEVENT, floorsCount)

    else:
        print('No RVs requested')  # noqa: T201
        filenameEVENT = arguments.filenameEVENT  # noqa: N816
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
               'timeout',
                str(maxSeconds),
                executableName,
                '-f',
                scriptName,
                # f'{os.path.realpath(os.path.dirname(__file__))}'  # noqa: ISC003, PTH120
                # + '/taichi_script.py',
            ],
            stdout=subprocess.PIPE,
            check=False,
        )

        forces = []
        floorsCount = 1  # noqa: N816
        for i in range(floorsCount):
            forces.append(FloorForces(recorderID=(i + 1)))

        # write the event file
        writeEVENT(forces, filenameEVENT, floorsCount=floorsCount)
        # writeEVENT(forces, arguments.filenameEVENT)
