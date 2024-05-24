from __future__ import print_function
import os, sys
import re
import json
import argparse

class FloorForces:
    def __init__(self):
        self.X = [0]
        self.Y = [0]
        self.Z = [0]

def directionToDof(direction):
    """
    Converts direction to degree of freedom
    """
    directioMap = {
        "X": 1,
        "Y": 2,
        "Z": 3
    }

    return directioMap[direction]


def addFloorForceToEvent(patternsArray, force, direction, floor):
    """
    Add force (one component) time series and pattern in the event file
    """
    seriesName = "WindForceSeries_" + str(floor) + direction
    patternName = "WindForcePattern_" + str(floor) + direction
    pattern = {
        "name": patternName,
        "timeSeries": seriesName,
        "type": "WindFloorLoad",
        "floor": str(floor),
        "dof": directionToDof(direction)
    }

    patternsArray.append(pattern)


def writeEVENT(forces, eventFilePath):
    """
    This method writes the EVENT.json file
    """
    patternsArray = []
    windEventJson = {
        "type" : "Hydro",
        "subtype": "CoupledDigitalTwin",
        "pattern": patternsArray,
        "pressure": [],
        "numSteps": len(forces[0].X),
        "units": {
            "force": "Newton",
            "length": "Meter",
            "time": "Sec"
        }
    }

    #Creating the event dictionary that will be used to export the EVENT json file
    eventDict = {"randomVariables":[], "Events": [windEventJson]}

    #Adding floor forces
    for floorForces in forces:
        floor = forces.index(floorForces) + 1
        addFloorForceToEvent(patternsArray, floorForces.X, "X", floor)
        addFloorForceToEvent(patternsArray, floorForces.Y, "Y", floor)

    with open(eventFilePath, "w", encoding='utf-8') as eventsFile:
        json.dump(eventDict, eventsFile)


def GetFloorsCount(BIMFilePath):
    with open(BIMFilePath,'r', encoding='utf-8') as BIMFile:
        bim = json.load(BIMFile) 
    
    return int(bim["GeneralInformation"]["stories"])
	
if __name__ == "__main__":
    """
    Entry point to generate event file using CFD
    """
    #CLI parser
    parser = argparse.ArgumentParser(description="Get sample EVENT file produced by CFD")
    parser.add_argument('-b', '--filenameAIM', help="BIM File", required=True)
    parser.add_argument('-e', '--filenameEVENT', help= "Event File", required=True)
    parser.add_argument('--getRV', help= "getRV", required=False, action='store_true')

    #parsing arguments
    arguments, unknowns = parser.parse_known_args()

    if arguments.getRV == True:
        #Read the number of floors
        floorsCount = GetFloorsCount(arguments.filenameAIM)
        forces = []
        for i in range(floorsCount):
            forces.append(FloorForces())
        #write the event file
        writeEVENT(forces, arguments.filenameEVENT)
    
    
    
