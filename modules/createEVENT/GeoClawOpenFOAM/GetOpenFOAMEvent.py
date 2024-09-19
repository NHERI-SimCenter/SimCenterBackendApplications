#!/usr/bin/env python  # noqa: INP001, D100
import argparse
import json
import os
import re


class FloorForces:  # noqa: D101
    def __init__(self):
        self.X = [0]
        self.Y = [0]
        self.Z = [0]


def validateCaseDirectoryStructure(caseDir):  # noqa: N802, N803
    """This method validates that the provided case directory is valid and contains the 0, constant and system directory
    It also checks that system directory contains the controlDict
    """  # noqa: D205, D400, D401, D404
    if not os.path.isdir(caseDir):  # noqa: PTH112
        return False  # noqa: DOC201, RUF100

    caseDirList = os.listdir(caseDir)  # noqa: N806
    necessaryDirs = ['0', 'constant', 'system', 'postProcessing']  # noqa: N806
    if any(aDir not in caseDirList for aDir in necessaryDirs):
        return False

    controlDictPath = os.path.join(caseDir, 'system/controlDict')  # noqa: PTH118, N806
    if not os.path.exists(controlDictPath):  # noqa: SIM103, PTH110
        return False

    return True


def parseForceComponents(forceArray):  # noqa: N802, N803
    """This method takes the OpenFOAM force array and parse into components x,y,z"""  # noqa: D400, D401, D404
    components = forceArray.strip('()').split()
    x = float(components[0])
    y = float(components[1])
    z = float(components[2])
    return [x, y, z]  # noqa: DOC201, RUF100


def ReadOpenFOAMForces(buildingForcesPath, floorsCount, startTime):  # noqa: N802, N803
    """This method will read the forces from the output files in the OpenFOAM case output (post processing)"""  # noqa: D400, D401, D404
    deltaT = 0  # noqa: N806
    forces = []
    for i in range(floorsCount):  # noqa: B007
        forces.append(FloorForces())  # noqa: PERF401
    forcePattern = re.compile(r'\([0-9.e\+\-\s]+\)')  # noqa: N806

    with open(buildingForcesPath) as forcesFile:  # noqa: PTH123, N806
        forceLines = forcesFile.readlines()  # noqa: N806
        needsDeltaT = True  # noqa: N806
        for line in forceLines:
            if line.startswith('#'):
                continue
            elif needsDeltaT:  # noqa: RET507
                deltaT = float(line.split()[0])  # noqa: N806
                needsDeltaT = False  # noqa: N806

            t = float(line.split()[0])
            if t > startTime:
                detectedForces = re.findall(forcePattern, line)  # noqa: N806

                for i in range(floorsCount):
                    # Read the different force types (pressure, viscous and porous!)
                    pressureForce = detectedForces[6 * i]  # noqa: N806
                    viscousForce = detectedForces[6 * i + 1]  # noqa: N806
                    porousForce = detectedForces[6 * i + 2]  # noqa: N806

                    # Parse force components
                    [fprx, fpry, fprz] = parseForceComponents(pressureForce)
                    [fvx, fvy, fvz] = parseForceComponents(viscousForce)
                    [fpox, fpoy, fpoz] = parseForceComponents(porousForce)

                    # Aggregate forces in X, Y, Z directions
                    forces[i].X.append(fprx + fvx + fpox)
                    forces[i].Y.append(fpry + fvy + fpoy)
                    forces[i].Z.append(fprz + fvz + fpoz)

    return [deltaT, forces]  # noqa: DOC201, RUF100


def directionToDof(direction):  # noqa: N802
    """Converts direction to degree of freedom"""  # noqa: D400, D401
    directioMap = {'X': 1, 'Y': 2, 'Z': 3}  # noqa: N806

    return directioMap[direction]  # noqa: DOC201, RUF100


def addFloorForceToEvent(  # noqa: N802
    timeSeriesArray,  # noqa: N803
    patternsArray,  # noqa: N803
    force,
    direction,
    floor,
    dT,  # noqa: N803
):
    """Add force (one component) time series and pattern in the event file"""  # noqa: D400
    seriesName = 'WaterForceSeries_' + str(floor) + direction  # noqa: N806
    timeSeries = {'name': seriesName, 'dT': dT, 'type': 'Value', 'data': force}  # noqa: N806

    timeSeriesArray.append(timeSeries)

    patternName = 'WaterForcePattern_' + str(floor) + direction  # noqa: N806
    pattern = {
        'name': patternName,
        'timeSeries': seriesName,
        'type': 'WaterFloorLoad',
        'floor': str(floor),
        'dof': directionToDof(direction),
    }

    patternsArray.append(pattern)


def addFloorPressure(pressureArray, floor):  # noqa: N802, N803
    """Add floor pressure in the event file"""  # noqa: D400
    floorPressure = {'story': str(floor), 'pressure': [0.0, 0.0]}  # noqa: N806

    pressureArray.append(floorPressure)


def writeEVENT(forces, deltaT):  # noqa: N802, N803
    """This method writes the EVENT.json file"""  # noqa: D400, D401, D404
    timeSeriesArray = []  # noqa: N806
    patternsArray = []  # noqa: N806
    pressureArray = []  # noqa: N806
    waterEventJson = {  # noqa: N806
        'type': 'Hydro',
        'subtype': 'OpenFOAM CFD Hydro Event',
        'timeSeries': timeSeriesArray,
        'pattern': patternsArray,
        'pressure': pressureArray,
        'dT': deltaT,
        'numSteps': len(forces[0].X),
        'units': {'force': 'Newton', 'length': 'Meter', 'time': 'Sec'},
    }

    # Creating the event dictionary that will be used to export the EVENT json file
    eventDict = {'randomVariables': [], 'Events': [waterEventJson]}  # noqa: N806

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

    with open('EVENT.json', 'w') as eventsFile:  # noqa: PTH123, N806
        json.dump(eventDict, eventsFile)


def GetOpenFOAMEvent(floorsCount, startTime):  # noqa: N802, N803
    """Read OpenFOAM output and generate an EVENT file for the building"""  # noqa: D400
    forcesOutputName = 'buildingsForces'  # noqa: N806

    if floorsCount == 1:
        buildingForcesPath = os.path.join(  # noqa: PTH118, N806
            'postProcessing', forcesOutputName, '0', 'forces.dat'
        )
    else:
        buildingForcesPath = os.path.join(  # noqa: PTH118, N806
            'postProcessing', forcesOutputName, '0', 'forces_bins.dat'
        )

    [deltaT, forces] = ReadOpenFOAMForces(  # noqa: N806
        buildingForcesPath, floorsCount, startTime
    )

    # Write the EVENT file
    writeEVENT(forces, deltaT)

    print('OpenFOAM event is written to EVENT.json')  # noqa: T201


def ReadBIM(BIMFilePath):  # noqa: N802, N803, D103
    with open(BIMFilePath) as BIMFile:  # noqa: PTH123, N806
        bim = json.load(BIMFile)

    return [
        int(bim['GeneralInformation']['stories']),
        float(bim['Events'][0]['StartTime']),
    ]


if __name__ == '__main__':
    """
    Entry point to read the forces from OpenFOAM case and use it for the EVENT
    """
    # CLI parser
    parser = argparse.ArgumentParser(
        description='Get EVENT file from OpenFOAM output'
    )
    parser.add_argument('-b', '--bim', help='path to BIM file', required=False)

    # parsing arguments
    arguments, unknowns = parser.parse_known_args()
    [floors, startTime] = ReadBIM(arguments.bim)  # noqa: N816

    GetOpenFOAMEvent(floors, startTime)
