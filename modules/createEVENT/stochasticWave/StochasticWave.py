#!/usr/bin/env python3  # noqa: EXE001, D100

"""Generate the event file using Stochastic Waves."""

import argparse
import json
import os
import re
import sys

import StochasticWaveLoadsJONSWAP
from StochasticWaveLoadsJONSWAP import ReadAIM

"""
Portions of this backend module are implemented courtesy of the welib python package:

Copyright 2019 E. Branlard
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


class FloorForces:  # noqa: D101
    def __init__(self):
        self.X = [0]
        self.Y = [0]
        self.Z = [0]


def validateCaseDirectoryStructure(caseDir):  # noqa: N802, N803
    """
    This method validates that the provided case directory is valid and contains the 0, constant and system directory
    It also checks that system directory contains the controlDict
    """  # noqa: D205, D400, D401, D404
    if not os.path.isdir(caseDir):  # noqa: PTH112
        print('Invalid case directory: ', caseDir)  # noqa: T201
        return False

    return True


def ReadForces(  # noqa: N802
    buildingForcesPath,  # noqa: N803
    floorsCount,  # noqa: N803
    startTime,  # noqa: ARG001, N803
    lengthScale,  # noqa: N803
    velocityScale,  # noqa: N803
    deltaT=1.0,  # noqa: N803
):
    """
    This method will read the forces from the output files in the MPM case output (post processing).
    It will scale dT and forces using the 2 scale factors: dT *= velocityScale/lengthScale; force *= lengthScale/(velocityScale*velocityScale)

    The scaling is also changed from model-scale to full-scale instead of the other way around
    """  # noqa: D205, D400, D401, D404
    forces = []

    timeFactor = 1.0 / (lengthScale / velocityScale)  # noqa: N806
    forceFactor = 1.0 / (lengthScale * velocityScale) ** 2.0  # noqa: N806

    if not os.path.exists(buildingForcesPath):  # noqa: PTH110
        print('Forces file not found: ', buildingForcesPath)  # noqa: T201
        for j in range(floorsCount):  # noqa: B007
            forces.append(FloorForces())  # noqa: PERF401
        deltaT = deltaT * timeFactor  # noqa: N806
        return [deltaT, forces]

    with open(buildingForcesPath) as file:  # noqa: PTH123
        print(  # noqa: T201
            'Reading recorder force time-series from forces.evt. To be placed into EVENT.json later.'
        )
        lines = file.readlines()
        j = 0
        for line in lines:
            print('Reading line ', j)  # noqa: T201
            # Ensure not empty line
            strip_line = line.strip()
            if not strip_line:
                print('Empty line found in ', buildingForcesPath, '.. skip')  # noqa: T201
                continue

            if (j + 1) > floorsCount:
                print(  # noqa: T201
                    'Number of floors exceeded in forces.evt, skipping the rest of the forces'
                )
                break

            forces.append(FloorForces())

            # Assume there is no header in the file
            # Assume recorder IDs are sequential, starting from 1
            clean_line = re.split(r';\s|;|,\s|,|\s+', strip_line)

            for k in range(len(clean_line)):
                forces[j].X.append(float(clean_line[k]) * forceFactor)
                forces[j].Y.append(0.0 * forceFactor)
                forces[j].Z.append(0.0 * forceFactor)
            j = j + 1
    file.close()

    deltaT = deltaT * timeFactor  # noqa: N806

    return [deltaT, forces]


def directionToDof(direction):  # noqa: N802
    """Converts direction to degree of freedom"""  # noqa: D400, D401
    directionMap = {'X': 1, 'Y': 2, 'Z': 3}  # noqa: N806

    return directionMap[direction]


def dofToDirection(dof):  # noqa: N802
    """Converts degree of freedom to direction"""  # noqa: D400, D401
    directionMap = {1: 'X', 2: 'Y', 3: 'Z'}  # noqa: N806

    return directionMap[dof]


def addFloorForceToEvent(  # noqa: N802
    timeSeriesArray,  # noqa: N803
    patternsArray,  # noqa: N803
    force,
    direction,
    floor=1,
    dT=1.0,  # noqa: N803
):
    """Add force (one component) time series and pattern in the event file
    Use of Wind is just a placeholder for now, since its more developed than Hydro
    """  # noqa: D205, D400
    seriesName = 'WindForceSeries_' + str(floor) + direction  # noqa: N806
    timeSeries = {  # noqa: N806
        'name': seriesName,
        'dT': dT,
        'numSteps': len(force),
        'type': 'Value',
        'data': force,
    }
    timeSeriesArray.append(timeSeries)

    patternName = 'WindForcePattern_' + str(floor) + direction  # noqa: N806
    pattern = {
        'name': patternName,
        'timeSeries': seriesName,
        'type': 'WindFloorLoad',
        'floor': str(floor),
        'dof': directionToDof(direction),
    }
    patternsArray.append(pattern)


def addFloorPressure(pressureArray, floor):  # noqa: N802, N803
    """
    Add floor pressure in the event file
    """  # noqa: D200, D400
    floorPressure = {  # noqa: N806
        'story': str(floor),
        'pressure': [0.0, 0.0],
    }

    pressureArray.append(floorPressure)


def readRV(aimFilePath='AIM.json', eventFilePath='EVENT.json'):  # noqa: D103, N802, N803
    with open(aimFilePath, encoding='utf-8') as aim:  # noqa: PTH123
        aimDict = json.load(aim)  # noqa: N806

    deltaT = aimDict['Events'][0]['timeStep']  # noqa: N806

    """Write the EVENT into the event JSON file."""
    timeSeriesArray = []  # noqa: N806
    patternsArray = []  # noqa: N806
    pressureArray = []  # noqa: N806

    eventDict = {'randomVariables': [], 'Events': []}  # noqa: N806

    forces = []
    floorsCount = GetFloorsCount(aimFilePath)  # noqa: N806
    for i in range(floorsCount):  # noqa: B007
        forces.append(FloorForces())  # noqa: PERF401

    event_json = {
        'type': 'Hydro',
        'subtype': 'StochasticWave',
        'eventClassification': 'Hydro',
        'dT': deltaT,
        'timeSeries': timeSeriesArray,
        'pattern': patternsArray,
        'pressure': pressureArray,
        'units': {'force': 'Newton', 'length': 'Meter', 'time': 'Sec'},
    }
    # Adding floor forces
    for floorForces in forces:  # noqa: N806
        floor = forces.index(floorForces) + 1
        addFloorForceToEvent(
            timeSeriesArray, patternsArray, floorForces.X, 'X', floor, deltaT
        )
        addFloorForceToEvent(
            timeSeriesArray, patternsArray, floorForces.Y, 'Y', floor, deltaT
        )
        addFloorPressure(pressureArray, floor)

    eventDict['Events'].append(event_json)

    filePath = eventFilePath  # noqa: F841, N806
    with open(eventFilePath, 'w', encoding='utf-8') as f:  # noqa: PTH123
        json.dump(eventDict, f, indent=2)


def writeEVENT(forces, eventFilePath, deltaT=1.0):  # noqa: D103, N802, N803
    sys.path.insert(0, os.getcwd())  # noqa: PTH109

    """Write the EVENT into the event JSON file."""
    timeSeriesArray = []  # noqa: N806
    patternsArray = []  # noqa: N806
    pressureArray = []  # noqa: N806

    event_json = {
        'type': 'Hydro',
        'subtype': 'StochasticWave',
        'eventClassification': 'Hydro',
        'timeSeries': timeSeriesArray,
        'pattern': patternsArray,
        'pressure': pressureArray,
        'dT': deltaT,
        'dt': deltaT,
        'numSteps': len(forces[0].X),
        'units': {'force': 'Newton', 'length': 'Meter', 'time': 'Sec'},
    }

    eventDict = {'randomVariables': [], 'Events': [event_json]}  # noqa: N806

    # Adding floor forces
    for floorForces in forces:  # noqa: N806
        floor = forces.index(floorForces) + 1
        addFloorForceToEvent(
            timeSeriesArray, patternsArray, floorForces.X, 'X', floor, deltaT
        )
        addFloorForceToEvent(
            timeSeriesArray, patternsArray, floorForces.Y, 'Y', floor, deltaT
        )
        addFloorPressure(pressureArray, floor)

    with open(eventFilePath, 'w', encoding='utf-8') as f:  # noqa: PTH123
        json.dump(eventDict, f)


def GetTimeStep(BIMFilePath):  # noqa: N802, N803, D103
    filePath = BIMFilePath  # noqa: N806
    with open(filePath, encoding='utf-8') as file:  # noqa: PTH123
        bim = json.load(file)
    file.close  # noqa: B018

    return float(bim['Events'][0]['timeStep'])


def GetFloorsCount(BIMFilePath):  # noqa: N802, N803, D103
    filePath = BIMFilePath  # noqa: N806
    with open(filePath, encoding='utf-8') as file:  # noqa: PTH123
        bim = json.load(file)
    file.close  # noqa: B018

    return int(bim['GeneralInformation']['stories'])


def GetEvent(  # noqa: N802
    caseDir,  # noqa: N803
    forcesOutputName,  # noqa: N803
    floorsCount,  # noqa: N803
    startTime,  # noqa: N803
    lengthScale,  # noqa: N803
    velocityScale,  # noqa: N803
    dT,  # noqa: N803
):
    """
    Reads MPM output and generate an EVENT file for the building
    """  # noqa: D200, D400, D401
    print('Check case directory: ', caseDir)  # noqa: T201
    if not validateCaseDirectoryStructure(caseDir):
        print('Invalid Case Directory!')  # noqa: T201
        sys.exit(-1)

    print('Join case directory with forces output file: ', forcesOutputName)  # noqa: T201
    buildingForcesPath = os.path.join(caseDir, forcesOutputName)  # noqa: PTH118, N806

    print('Check building forces filepath: ', buildingForcesPath)  # noqa: T201
    if not os.path.exists(buildingForcesPath):  # noqa: PTH110
        print('Forces file not found: ', buildingForcesPath)  # noqa: T201

    [deltaT, forces] = ReadForces(  # noqa: N806
        buildingForcesPath, floorsCount, startTime, lengthScale, velocityScale, dT
    )

    # Write the EVENT file
    writeEVENT(forces, 'EVENT.json', deltaT)
    print('StochasticWave event is written to EVENT.json')  # noqa: T201


def ReadBIM(BIMFilePath):  # noqa: N802, N803, D103
    with open(BIMFilePath) as BIMFile:  # noqa: PTH123, N806
        bim = json.load(BIMFile)
    eventType = bim['Applications']['Events'][0]['Application']  # noqa: N806
    if eventType == 'StochasticWave' or eventType == 'StochasticWaveJONSWAP':  # noqa: PLR1714
        forcesOutputName = 'forces.evt'  # noqa: N806
        print('StochasticWave event type: ', eventType)  # noqa: T201
        return [
            forcesOutputName,
            int(bim['GeneralInformation']['stories']),
            0.0,
            1.0,
            1.0,
        ]
    print('Unsupported event type: ', eventType)  # noqa: T201
    sys.exit(-1)


def main():
    """Generate the event file using Stochastic Waves."""
    return 0


if __name__ == '__main__':
    """
    Entry point to generate event file using Stochastic Waves when run as a script.
    """

    # CLI parser
    parser = argparse.ArgumentParser(
        description='Get sample EVENT file produced by the StochasticWave module'
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
    parser.add_argument(
        '--getRV',
        help='getRV',
        nargs='?',
        required=False,
        const=True,
        default=False,
    )

    arguments = parser.parse_args()

    print('No RVs requested in StochasticWave.py')  # noqa: T201
    currentDir = os.getcwd()  # noqa: PTH109, N816
    print('Current Directory: ', currentDir)  # noqa: T201
    filenameAIM = arguments.filenameAIM  # noqa: N816
    filenameEVENT = arguments.filenameEVENT  # noqa: N816

    print('Reading AIM file')  # noqa: T201
    if arguments.getRV == True:  # noqa: E712
        StochasticWaveLoadsJONSWAP.ReadAIM(arguments.filenameAIM, True)  # noqa: FBT003
    else:
        StochasticWaveLoadsJONSWAP.ReadAIM(arguments.filenameAIM, False)  # noqa: FBT003

    # write the event file
    print('Reading BIM file')  # noqa: T201
    [forcesOutputName, floors, startTime, lengthScale, velocityScale] = ReadBIM(  # noqa: N816
        arguments.filenameAIM
    )
    print('Process ', filenameEVENT)  # noqa: T201
    sys.exit(
        GetEvent(
            currentDir,
            forcesOutputName,
            floors,
            startTime,
            lengthScale,
            velocityScale,
            dT=GetTimeStep(arguments.filenameAIM),
        )
    )

    print('StochasticWave.py completed')  # noqa: T201
