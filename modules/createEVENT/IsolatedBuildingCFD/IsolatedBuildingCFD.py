from __future__ import print_function
import os, sys
import re
import json
import argparse
import numpy

class FloorForces:
    def __init__(self):
        self.X = [0]
        self.Y = [0]
        self.Z = [0]
    
def validateCaseDirectoryStructure(caseDir):
    """
    This method validates that the provided case directory is valid and contains the 0, constant and system directory
    It also checks that system directory contains the controlDict
    """
    if not os.path.isdir(caseDir):
        return False
    
    caseDirList = os.listdir(caseDir)
    necessaryDirs = ["0", "constant", "system", "postProcessing"]
    if any(not aDir in caseDirList for aDir in necessaryDirs):
        return False

    controlDictPath = os.path.join(caseDir, "system/controlDict")
    if not os.path.exists(controlDictPath):
        return False
    
    return True

def parseForceComponents(forceArray):
    """
    This method takes the OpenFOAM force array and parse into components x,y,z
    """
    components = forceArray.strip('()').split()
    x = float(components[0])
    y = float(components[1])
    z = float(components[2])
    return [x, y, z]

def ReadOpenFOAMForces(buildingForcesPath, floorsCount, startTime, lengthScale, velocityScale):
    """
    This method will read the forces from the output files in the OpenFOAM case output (post processing). 
    It will scale dT and forces using the 2 scale factors: dT *= velocityScale/lengthScale; force *= lengthScale/(velocityScale*velocityScale)
    
    In newer version of OpenFOAM (>=9) the output force format has been changed, the porous forces are not written anymore. So, depending on the
    wether the porous is written or not  the expected format is changed. 
    
    The scaling is also changed from model-scale to full-scale instead of the other way around
    """
    deltaT = 0
    forces = []
    
    
    for i in range(floorsCount):
        forces.append(FloorForces())
    forcePattern = re.compile(r"\([0-9.e\+\-\s]+\)")
    
    #Block of forces and moments to read
    nBlocks = 6 # [pressure->[force, moment]; viscous->[force, moment]; porous->[force, moment]]
    
    
    timeFactor = 1.0/(lengthScale/velocityScale)
    forceFactor = 1.0/(lengthScale*velocityScale)**2.0
    
    timeCount = 0
        
    with open(buildingForcesPath, 'r') as forcesFile:
        forceLines = forcesFile.readlines()
        for line in forceLines:
            if line.startswith("# Time"):
                if "(pressure viscous)" in line:
                    nBlocks = 4
                elif "(pressure viscous porous)" in line:
                    nBlocks = 6
                    
            if line.startswith("#"):
                continue

            time = float(line.split()[0])
            timeCount += 1
            
            if timeCount==1:
                deltaT = time

            if timeCount==2:
                deltaT = time - deltaT
                
            
            if time > startTime:
                detectedForces = re.findall(forcePattern, line)

                for i in range(floorsCount):
                    # Read the different force types (pressure, viscous and porous!)
                    # porous part not needed for wind load 
                    pressureForce = detectedForces[nBlocks*i]
                    viscousForce = detectedForces[nBlocks*i + 1]
                    # porousForce = detectedForces[nBlocks*i + 2]

                    # Parse force components 
                    [fprx, fpry, fprz] = parseForceComponents(pressureForce)
                    [fvx, fvy, fvz] = parseForceComponents(viscousForce)
                    # [fpox, fpoy, fpoz] = parseForceComponents(porousForce)
                    
                    # Aggregate forces in X, Y, Z directions
                    forces[i].X.append((fprx + fvx)*forceFactor)
                    forces[i].Y.append((fpry + fvy)*forceFactor)
                    forces[i].Z.append((fprz + fvz)*forceFactor)

    deltaT = deltaT*timeFactor
    
    return [deltaT, forces]

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


def addFloorForceToEvent(timeSeriesArray, patternsArray, force, direction, floor, dT):
    """
    Add force (one component) time series and pattern in the event file
    """
    seriesName = "WindForceSeries_" + str(floor) + direction
    timeSeries = {
                "name": seriesName,
                "dT": dT,
                "type": "Value",
                "data": force
            }
    
    timeSeriesArray.append(timeSeries)
    
    patternName = "WindForcePattern_" + str(floor) + direction
    pattern = {
        "name": patternName,
        "timeSeries": seriesName,
        "type": "WindFloorLoad",
        "floor": str(floor),
        "dof": directionToDof(direction)
    }

    patternsArray.append(pattern)

def addFloorPressure(pressureArray, floor):
    """
    Add floor pressure in the event file
    """
    floorPressure = {
        "story":str(floor),
        "pressure":[0.0, 0.0]
    }

    pressureArray.append(floorPressure)


def writeEVENT(forces, deltaT):
    """
    This method writes the EVENT.json file
    """
    timeSeriesArray = []
    patternsArray = []
    pressureArray = []
    
    windEventJson = {
        "type" : "Wind",
        "subtype": "IsolatedBuildingCFD",
        "timeSeries": timeSeriesArray,
        "pattern": patternsArray,
        "pressure": pressureArray,
        "dT": deltaT,
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
        addFloorForceToEvent(timeSeriesArray, patternsArray, floorForces.X, "X", floor, deltaT)
        addFloorForceToEvent(timeSeriesArray, patternsArray, floorForces.Y, "Y", floor, deltaT)
        addFloorPressure(pressureArray, floor)

    with open("EVENT.json", "w") as eventsFile:
        json.dump(eventDict, eventsFile)


def GetOpenFOAMEvent(caseDir, forcesOutputName, floorsCount, startTime, lengthScale, velocityScale):
    """
    Reads OpenFOAM output and generate an EVENT file for the building
    """
    if not validateCaseDirectoryStructure(caseDir):
        print("Invalid OpenFOAM Case Directory!")
        sys.exit(-1)
        

    if floorsCount == 1:        
        buildingForcesPath = os.path.join(caseDir, "postProcessing", forcesOutputName, "0", "forces.dat")
    else:
        buildingForcesPath = os.path.join(caseDir, "postProcessing", forcesOutputName, "0", "forces_bins.dat")
        
    [deltaT, forces] = ReadOpenFOAMForces(buildingForcesPath, floorsCount, startTime, lengthScale, velocityScale)

    # Write the EVENT file
    writeEVENT(forces, deltaT)

    print("OpenFOAM event is written to EVENT.json")

def ReadBIM(BIMFilePath):
    with open(BIMFilePath,'r') as BIMFile:
        bim = json.load(BIMFile)

    eventType = bim["Applications"]["Events"][0]["Application"]
    forcesOutputName = 'buildingsForces'

    if eventType == "IsolatedBuildingCFD":
        nStories = int(bim["GeneralInformation"]["stories"])
        gScale = 1.0/float(bim["Events"][0]["GeometricData"]["geometricScale"])
        vScale = 1.0/float(bim["Events"][0]["windCharacteristics"]["velocityScale"])
        startTime = 0.05*float(bim["Events"][0]["numericalSetup"]["duration"]) #Discard the first 5% of the simulation
        forcesOutputName = 'storyForces'
        return [forcesOutputName, nStories, startTime, gScale, vScale]

    else:
        if "LengthScale" in bim["Events"][0]:
            return [forcesOutputName, int(bim["GeneralInformation"]["stories"]), float(bim["Events"][0]["start"]), float(bim["Events"][0]["LengthScale"]), float(bim["Events"][0]["VelocityScale"])]
        else:
            return [forcesOutputName, int(bim["GeneralInformation"]["stories"]), float(bim["Events"][0]["start"]), 1.0, 1.0]


import json
import os

def scale_event(filename_aim, filename_event):
    """ Scales event factor in a JSON event file based on AIM file parameters. """

    # Check if AIM file exists
    if not os.path.exists(filename_aim):
        print(f"Error: AIM file '{filename_aim}' does not exist.")
        return
    
    # Check if Event file exists
    if not os.path.exists(filename_event):
        print(f"Error: Event file '{filename_event}' does not exist.")
        return

    try:
        # Load AIM file
        with open(filename_aim, 'r') as input_file:
            bim = json.load(input_file)

        # Validate AIM structure
        if "Events" not in bim or not isinstance(bim["Events"], list) or not bim["Events"]:
            print(f"Error: 'Events' key is missing or invalid in '{filename_aim}'.")
            return
        
        event = bim["Events"][0]

        # Validate event type key
        if "type" not in event:
            print("Error: Missing 'type' key in event data.")
            return
        
        event_type = event["type"]

        print(f'event_type {event_type}')
        # Check for 'IsolatedBuildingCFD' event type
        if event_type == "IsolatedBuildingCFD":
            
            # Validate 'windCharacteristics' structure
            if "windCharacteristics" in event and "windSpeedScalingFactor" in event["windCharacteristics"]:

                scaling_factor = event["windCharacteristics"]["windSpeedScalingFactor"]

                print(f'scaling_factor {scaling_factor}')
                

                if scaling_factor != 1.0:
                    # Open event file and modify scale factor
                    with open(filename_event, 'r') as event_file:
                        event_data = json.load(event_file)

                    # Validate event structure
                    if "Events" not in event_data or not isinstance(event_data["Events"], list) or not event_data["Events"]:
                        print(f"Error: 'Events' key is missing or invalid in '{filename_event}'.")
                        return
                
                    event_0 = event_data["Events"][0]

                    # Modify or add 'factor' in the event
                    if "factor" in event_0:
                        event_0["factor"] *= scaling_factor  # Multiply existing value
                    else:
                        event_0["factor"] = scaling_factor  # Set new value

                    # Write the updated data back to the same file
                    with open(filename_event, 'w') as event_file:
                        json.dump(event_data, event_file, indent=2)

                    print(f"Updated event factor in '{filename_event}' with scaling factor {scaling_factor}.")
    
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error processing files: {e}")

        

if __name__ == "__main__":
    """
    Entry point to read the forces from OpenFOAM case and use it for the EVENT
    """
    
    #CLI parser
    parser = argparse.ArgumentParser(description="Get EVENT file from OpenFOAM output")
    parser.add_argument('-c', '--case', help="OpenFOAM case directory", required=False)
    parser.add_argument('-b', '--bim', help= "path to BIM file", required=False)
    parser.add_argument('--filenameAIM', help= "path to BIM file", required=False)
    parser.add_argument('--filenameEVENT', help= "path to EVENT file", required=False)        
    parser.add_argument("--getRV", action="store_true", help="used to get random variable, if not multiply EVENT by gs")
    

    #parsing arguments
    arguments, unknowns = parser.parse_known_args()

    # [forcesOutputName, floors, startTime, lengthScale, velocityScale] = ReadBIM(arguments.bim)

    # GetOpenFOAMEvent(arguments.case, forcesOutputName, floors, startTime, lengthScale, velocityScale)

    if not arguments.getRV:

        scale_event(filename_aim = arguments.filenameAIM, filename_event = arguments.filenameEVENT)

        
        
    
