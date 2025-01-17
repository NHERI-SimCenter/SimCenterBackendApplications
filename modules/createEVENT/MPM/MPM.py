#!/usr/bin/env python3 # noqa: EXE001, D100
import argparse
import json


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
        'subtype': 'MPM',  # Using ClaymoreUW Material Point Method
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


if __name__ == '__main__':
    """
    Entry point to generate event file using HydroUQ MPM (ClaymoreUW Material Point Method)
    """
    # CLI parser
    parser = argparse.ArgumentParser(
        description='Get sample EVENT file produced by HydroUQ MPM'
    )
    parser.add_argument('-b', '--filenameAIM', help='BIM File', required=True)
    parser.add_argument('-e', '--filenameEVENT', help='Event File', required=True)
    parser.add_argument(
        '--getRV',
        help='getRV',
        required=False,
        action='store_true',
        default=False,
    )

    # Parsing arguments
    arguments, unknowns = parser.parse_known_args()

    if arguments.getRV == True:  # noqa: E712
        # Read the number of floors
        # Reads BIM file
        floorsCount = GetFloorsCount(arguments.filenameAIM)  # noqa: N816
        forces = []
        for i in range(floorsCount):  # noqa: B007
            forces.append(FloorForces())  # noqa: PERF401
        # Write the event file
        writeEVENT(forces, arguments.filenameEVENT)
