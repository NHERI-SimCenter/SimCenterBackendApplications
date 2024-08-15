import argparse
import json


class FloorForces:
    def __init__(self):
        self.X = [0]
        self.Y = [0]
        self.Z = [0]


def directionToDof(direction):
    """Converts direction to degree of freedom"""
    directioMap = {'X': 1, 'Y': 2, 'Z': 3}

    return directioMap[direction]


def addFloorForceToEvent(
    timeSeriesArray,
    patternsArray,
    force,
    direction,
    floor,
    dT,
):
    """Add force (one component) time series and pattern in the event file"""
    seriesName = 'HydroForceSeries_' + str(floor) + direction
    timeSeries = {'name': seriesName, 'dT': dT, 'type': 'Value', 'data': force}

    timeSeriesArray.append(timeSeries)
    patternName = 'HydroForcePattern_' + str(floor) + direction
    pattern = {
        'name': patternName,
        'timeSeries': seriesName,
        'type': 'HydroFloorLoad',
        'floor': str(floor),
        'dof': directionToDof(direction),
    }

    patternsArray.append(pattern)


def addFloorForceToEvent(patternsArray, force, direction, floor):
    """Add force (one component) time series and pattern in the event file"""
    seriesName = 'HydroForceSeries_' + str(floor) + direction
    patternName = 'HydroForcePattern_' + str(floor) + direction
    pattern = {
        'name': patternName,
        'timeSeries': seriesName,
        'type': 'HydroFloorLoad',
        'floor': str(floor),
        'dof': directionToDof(direction),
    }

    patternsArray.append(pattern)


def addFloorPressure(pressureArray, floor):
    """Add floor pressure in the event file"""
    floorPressure = {'story': str(floor), 'pressure': [0.0, 0.0]}

    pressureArray.append(floorPressure)


def writeEVENT(forces, eventFilePath):
    """This method writes the EVENT.json file"""
    timeSeriesArray = []
    patternsArray = []
    pressureArray = []
    hydroEventJson = {
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
    eventDict = {'randomVariables': [], 'Events': [hydroEventJson]}

    # Adding floor forces
    for floorForces in forces:
        floor = forces.index(floorForces) + 1
        addFloorForceToEvent(patternsArray, floorForces.X, 'X', floor)
        addFloorForceToEvent(patternsArray, floorForces.Y, 'Y', floor)
        # addFloorPressure(pressureArray, floor) # From GeoClawOpenFOAM

    with open(eventFilePath, 'w', encoding='utf-8') as eventsFile:
        json.dump(eventDict, eventsFile)


def GetFloorsCount(BIMFilePath):
    with open(BIMFilePath, encoding='utf-8') as BIMFile:
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
        '--getRV', help='getRV', required=False, action='store_true', default=False
    )

    # Parsing arguments
    arguments, unknowns = parser.parse_known_args()

    if arguments.getRV == True:
        # Read the number of floors
        # Reads BIM file
        floorsCount = GetFloorsCount(arguments.filenameAIM)
        forces = []
        for i in range(floorsCount):
            forces.append(FloorForces())
        # Write the event file
        writeEVENT(forces, arguments.filenameEVENT)
