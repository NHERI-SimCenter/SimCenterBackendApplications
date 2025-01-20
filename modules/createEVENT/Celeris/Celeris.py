#!/usr/bin/env python3  # noqa: EXE001

"""
Author: Justin Bonus, 2024

This script generates an EVENT.json file using the Celeris EVT to run CelerisAi (Python and Taichi Lang).

Permission for respectful distribution, modification, and training related to the SimCenter's mission
was granted by Patrick Lynett and Willington Renteria on 2024-9-25.

MIT License

Copyright (c) 2024 WILLINGTON RENTERIA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""  # noqa: D404

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

# Check if taichi is installed before importing taichi
try:
    import taichi as ti
except ImportError:
    print('Taichi is not installed. Please install it using "pip install taichi".')  # noqa: T201
    print()  # noqa: T201
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'taichi'], check=False)  # noqa: S603
    try:
        import taichi as ti
    except ImportError:
        print('Taichi installation failed. Please install it manually.')  # noqa: T201
        sys.exit(1)
    print('Taichi is installed successfully.')  # noqa: T201
    print()  # noqa: T201


class FloorForces:  # noqa: D101
    def __init__(self, recorderID=-1):  # noqa: N803
        if recorderID < 0:
            print(  # noqa: T201
                'No recorder ID, or a negative ID, provided, defaulting to 0 for all forces.'
            )
            self.X = [0.0]
            self.Y = [0.0]
            self.Z = [0.0]
        else:
            self.X = []
            self.Y = []
            self.Z = []
            # prepend zeros to the list to account for the timeSeries transient analysis req in OpenSees
            prependZero = False  # noqa: N806
            if prependZero:
                self.X.append(0.0)
                self.Y.append(0.0)
                self.Z.append(0.0)

            # Read in forces.[out or evt] file and add to EVENT.json
            # now using intermediary forces.evt for output of preceding Python calcs,
            # prevents confusion with forces.out made by FEM tab
            if os.path.exists('forces.evt'):  # noqa: PTH110
                with open('forces.evt') as file:  # noqa: PTH123
                    print('Reading forces from forces.evt to EVENT.json')  # noqa: T201
                    lines = file.readlines()
                    j = 0
                    for line in lines:
                        # Ensure not empty line
                        strip_line = line.strip()
                        if not strip_line:
                            print('Empty line found in forces.evt... skip')  # noqa: T201
                            continue
                        # Assume there is no header in the file
                        # Assume recorder IDs are sequential, starting from 1
                        if (j + 1) == recorderID:
                            # Strip away leading / trailing white-space,
                            # Delimit by regex to capture " ", \s, "  ", tabs, etc.
                            # Each value should be a number, rep. the force on recorder j at a time-step i
                            # clean_line = re.split() # default is '\s+', which is any whitespace
                            clean_line = re.split(r';\s|;|,\s|,|\s+', strip_line)
                            # clean_line = re.split(r';|,\s', strip_line)
                            # clean_line = re.split("\s+", strip_line)

                            for k in range(len(clean_line)):
                                self.X.append(float(clean_line[k]))
                                self.Y.append(0.0)
                                self.Z.append(0.0)
                        j = j + 1

                # must not have empty lists for max and min
                if len(self.X) == 0:
                    print(  # noqa: T201
                        'No forces found in the file for recorder ',
                        recorderID,
                        ', defaulting to 0.0 for all forces.',
                    )
                    self.X = [0.0]
                    self.Y = [0.0]
                    self.Z = [0.0]
                else:
                    # for a timeSeries with N elements, we append an element at N+1 to represent the max force of the series
                    self.X.append(max(self.X))
                    self.Y.append(max(self.Y))
                    self.Z.append(max(self.Z))

                    print(  # noqa: T201
                        'Length: ',
                        len(self.X),
                        ', Max force: ',
                        max(self.X),
                        max(self.Y),
                        max(self.Z),
                        ', Min force: ',
                        min(self.X),
                        min(self.Y),
                        min(self.Z),
                        ', Last force: ',
                        self.X[-1],
                        self.Y[-1],
                        self.Z[-1],
                    )
                file.close  # noqa: B018
            else:
                print(  # noqa: T201
                    'No forces.evt file found, defaulting to 0.0 for all forces.'
                )
                self.X.append(0.0)
                self.Y.append(0.0)
                self.Z.append(0.0)


def directionToDof(direction):  # noqa: N802
    """
    Converts direction to degree of freedom
    """  # noqa: D200, D400, D401
    directionMap = {'X': 1, 'Y': 2, 'Z': 3}  # noqa: N806

    return directionMap[direction]


def addFloorForceToEvent(  # noqa: N802
    patternsList,  # noqa: N803
    timeSeriesList,  # noqa: N803
    force,
    direction,
    floor,
):
    """
    Add force (one component) time series and pattern in the event file
    Use of Wind is just a placeholder for now, since its more developed than Hydro
    """  # noqa: D205, D400
    seriesName = '1'  # noqa: N806
    patternName = '1'  # noqa: N806
    seriesName = 'WindForceSeries_' + str(floor) + direction  # noqa: N806
    patternName = 'WindForcePattern_' + str(floor) + direction  # noqa: N806

    pattern = {
        'name': patternName,
        'timeSeries': seriesName,
        'numSteps': len(force.X),
        'dT': 0.01,
        'type': 'WindFloorLoad',
        'floor': str(floor),
        'story': str(floor),
        'dof': directionToDof(direction),
        'units': {'force': 'Newton', 'length': 'Meter', 'time': 'Sec'},
    }
    sensorData = {  # noqa: N806
        'name': seriesName,
        'pattern': patternName,
        'type': 'Value',
        'dof': directionToDof(direction),
        'floor': str(floor),
        'story': str(floor),
        'dT': 0.01,
        'dt': 0.01,
        'numSteps': len(force.X),
        'data': force.X,
    }

    patternsList.append(pattern)
    timeSeriesList.append(sensorData)


def writeEVENT(forces, eventFilePath='EVENT.json', floorsCount=1):  # noqa: N802, N803
    """
    This method writes the EVENT.json file
    """  # noqa: D200, D400, D401, D404
    # Adding floor forces
    patternsArray = []  # noqa: N806
    timeSeriesArray = []  # noqa: N806
    # timeSeriesType = "Value" # ? saw in old evt files

    # pressure = [{"pressure": [0.0, 0.0], "story": 1}]
    pressure = []

    for it in range(floorsCount):
        floorForces = forces[it]  # noqa: N806
        addFloorForceToEvent(
            patternsArray, timeSeriesArray, floorForces, 'X', it + 1
        )

    # subtype = "StochasticWindModel-KwonKareem2006"
    eventClassification = 'Hydro'  # noqa: N806
    eventType = 'Celeris'  # noqa: N806
    eventSubtype = 'Celeris'  # noqa: N806, F841
    # timeSeriesName = "HydroForceSeries_1X"
    # patternName = "HydroForcePattern_1X"

    hydroEventJson = {  # noqa: N806
        'type': eventClassification,
        'subtype': eventType,
        'eventClassification': eventClassification,
        'pattern': patternsArray,
        'timeSeries': timeSeriesArray,
        'pressure': pressure,
        'numSteps': len(forces[0].X),
        'dT': 0.01,
        'dt': 0.01,
        'units': {'force': 'Newton', 'length': 'Meter', 'time': 'Sec'},
    }

    # Creating the event dictionary that will be used to export the EVENT json file
    eventDict = {'randomVariables': [], 'Events': [hydroEventJson]}  # noqa: N806

    filePath = eventFilePath  # noqa: N806
    with open(filePath, 'w', encoding='utf-8') as file:  # noqa: PTH123
        json.dump(eventDict, file)
    file.close  # noqa: B018


def GetFloorsCount(BIMFilePath):  # noqa: N802, N803, D103
    filePath = BIMFilePath  # noqa: N806
    with open(filePath, encoding='utf-8') as file:  # noqa: PTH123
        bim = json.load(file)
    file.close  # noqa: B018

    return int(bim['GeneralInformation']['stories'])


def GetCelerisScript(BIMFilePath):  # noqa: N802, N803, D103
    filePath = BIMFilePath  # noqa: N806
    with open(filePath, encoding='utf-8') as file:  # noqa: PTH123
        evt = json.load(file)
    file.close  # noqa: B018

    fileNameKey = 'simulationScript'  # noqa: N806
    filePathKey = fileNameKey + 'Path'  # noqa: N806

    for event in evt['Events']:
        fileName = event[fileNameKey]  # noqa: N806
        filePath = event[filePathKey]  # noqa: N806
        return os.path.join(filePath, fileName)  # noqa: PTH118

    defaultScriptPath = f'{os.path.realpath(os.path.dirname(__file__))}'  # noqa: N806, PTH120
    defaultScriptName = 'setrun.py'  # noqa: N806
    return defaultScriptPath + defaultScriptName


def main():
    """
    Entry point to generate event file using Celeris.
    """  # noqa: D200
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
    scriptName = GetCelerisScript(arguments.filenameAIM)  # noqa: N816

    filePath = arguments.filenameAIM  # noqa: N816
    with open(filePath, encoding='utf-8') as file:  # noqa: PTH123
        evt = json.load(file)
    file.close  # noqa: B018

    configFilename = 'config.json'  # noqa: N816
    bathymetryFilename = 'bathymetry.txt'  # noqa: N816
    waveFilename = 'wave.txt'  # noqa: N816
    caseDirectory = './examples/CrescentCity'  # noqa: N816

    for event in evt['Events']:
        # Redesign the input structure in backend CelerisAi later.
        # For now assume waveFile, bathymetryFile, configFile, etc. are in the same directory.
        caseDirectory = event['configFilePath']  # noqa: N816
        configFilename = event['configFile']  # noqa: N816
        bathymetryFilename = event['bathymetryFile']  # noqa: N816
        waveFilename = event['waveFile']  # noqa: N816

    print('Running Celeris with script:', scriptName)  # noqa: T201
    print('Running Celeris with directory:', caseDirectory)  # noqa: T201
    print('Running Celeris with config file:', configFilename)  # noqa: T201
    print('Running Celeris with bathymetry:', bathymetryFilename)  # noqa: T201
    print('Running Celeris with waves:', waveFilename)  # noqa: T201

    if arguments.getRV == True:  # noqa: E712
        print('RVs requested')  # noqa: T201
        # Read the number of floors
        floorsCount = GetFloorsCount(arguments.filenameAIM)  # noqa: N816
        filenameEVENT = arguments.filenameEVENT  # noqa: N816

        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                'python3',
                scriptName,
                '-d',
                caseDirectory,
                '-f',
                configFilename,
                '-b',
                bathymetryFilename,
                '-w',
                waveFilename,
                # f'{os.path.realpath(os.path.dirname(__file__))}'
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
                'python3',
                scriptName,
                '-d',
                caseDirectory,
                '-f',
                configFilename,
                '-b',
                bathymetryFilename,
                '-w',
                waveFilename,
                # f'{os.path.realpath(os.path.dirname(__file__))}'
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
