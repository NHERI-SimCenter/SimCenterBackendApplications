#!/usr/bin/env python3

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
import taichi as ti

# import taichi_script

# from taichi.tools import *
# from params import *


class FloorForces:
    def __init__(self, recorderID=-1):
        if recorderID < 0:
            print(
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
            prependZero = False
            if prependZero:
                self.X.append(0.0)
                self.Y.append(0.0)
                self.Z.append(0.0)

            # Read in forces.[out or evt] file and add to EVENT.json
            # now using intermediary forces.evt for output of preceeding Python calcs,
            # prevents confusion with forces.out made by FEM tab
            if os.path.exists('forces.evt'):
                with open('forces.evt') as file:
                    print('Reading forces from forces.evt to EVENT.json')
                    lines = file.readlines()
                    j = 0
                    for line in lines:
                        # Ensure not empty line
                        strip_line = line.strip()
                        if not strip_line:
                            print('Empty line found in forces.evt... skip')
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
                    print(
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

                    print(
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
                file.close
            else:
                print('No forces.evt file found, defaulting to 0.0 for all forces.')
                self.X.append(0.0)
                self.Y.append(0.0)
                self.Z.append(0.0)


def directionToDof(direction):
    """
    Converts direction to degree of freedom
    """
    directioMap = {'X': 1, 'Y': 2, 'Z': 3}

    return directioMap[direction]


def addFloorForceToEvent(patternsList, timeSeriesList, force, direction, floor):
    """
    Add force (one component) time series and pattern in the event file
    Use of Wind is just a placeholder for now, since its more developed than Hydro
    """
    seriesName = '1'
    patternName = '1'
    seriesName = 'WindForceSeries_' + str(floor) + direction
    patternName = 'WindForcePattern_' + str(floor) + direction

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
    sensorData = {
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


def writeEVENT(forces, eventFilePath='EVENT.json', floorsCount=1):
    """
    This method writes the EVENT.json file
    """
    # Adding floor forces
    patternsArray = []
    timeSeriesArray = []
    # timeSeriesType = "Value" # ? saw in old evt files

    # pressure = [{"pressure": [0.0, 0.0], "story": 1}]
    pressure = []

    for it in range(floorsCount):
        floorForces = forces[it]
        addFloorForceToEvent(
            patternsArray, timeSeriesArray, floorForces, 'X', it + 1
        )

    # subtype = "StochasticWindModel-KwonKareem2006"
    eventClassification = 'Hydro'
    eventType = 'StochasticWave'
    eventSubtype = 'StochasticWaveJonswap'
    # subtype = "StochasticWaveJonswap" # ?
    # timeSeriesName = "HydroForceSeries_1X"
    # patternName = "HydroForcePattern_1X"

    hydroEventJson = {
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
    eventDict = {'randomVariables': [], 'Events': [hydroEventJson]}

    filePath = eventFilePath
    with open(filePath, 'w', encoding='utf-8') as file:
        json.dump(eventDict, file)
    file.close


def GetFloorsCount(BIMFilePath):
    filePath = BIMFilePath
    with open(filePath, encoding='utf-8') as file:
        bim = json.load(file)
    file.close

    return int(bim['GeneralInformation']['stories'])


def main():
    return 0
    # """
    # Entry point to generate event file using Stochastic Waves
    # """
    # #CLI parser
    # parser = argparse.ArgumentParser(description="Get sample EVENT file produced by StochasticWave")
    # parser.add_argument('-b', '--filenameAIM', help="BIM File", required=True)
    # parser.add_argument('-e', '--filenameEVENT', help= "Event File", required=True)
    # parser.add_argument('--getRV', help= "getRV", required=False, action='store_true')

    # #parsing arguments
    # arguments, unknowns = parser.parse_known_args()

    # exec(open("Ex1_WaveKinematics.py").read())
    # exec(open("Ex2_Jonswap_Spectrum.py").read())
    # exec(open("Ex3_WaveTimeSeries.py").read())
    # # exec(open("Ex4_WaveLoads.py").read())

    # # Run Ex4_WaveLoads.py with the given parameters
    # # result = Ex4_WaveLoads.main(arguments.water_depth, arguments.peak_period, arguments.significant_wave_height, arguments.pile_diameter, arguments.drag_coefficient, arguments.mass_coefficient, arguments.number_of_recorders_z, arguments.time)
    # import subprocess
    # result = subprocess.run(["python", "Ex4_WaveLoads.py", "-hw", 30.0, "-Tp", 12.7, "-Hs", 5.0, "-Dp", 1.0, "-Cd", 2.1, "-Cm", 2.1, "-nz", GetFloorsCount(arguments.filenameAIM), "-t", 10.0], stdout=subprocess.PIPE)

    # if arguments.getRV == True:
    #     #Read the number of floors
    #     floorsCount = GetFloorsCount(arguments.filenameAIM)
    #     forces = []
    #     for i in range(floorsCount):
    #         forces.append(FloorForces())

    #     #write the event file
    #     writeEVENT(forces, arguments.filenameEVENT)


if __name__ == '__main__':
    """
    Entry point to generate event file using Stochastic Waves
    """
    # CLI parser
    parser = argparse.ArgumentParser(
        description='Get sample EVENT file produced by StochasticWave'
    )
    parser.add_argument(
        '-b', '--filenameAIM', help='BIM File', required=True, default='AIM.json'
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

    # Run Ex4_WaveLoads.py with the given parameters
    # result = Ex4_WaveLoads.main(arguments.water_depth, arguments.peak_period, arguments.significant_wave_height, arguments.pile_diameter, arguments.drag_coefficient, arguments.mass_coefficient, arguments.number_of_recorders_z, arguments.time)

    # import subprocess

    if arguments.getRV == True:
        print('RVs requested in StochasticWave.py')
        # Read the number of floors
        floorsCount = GetFloorsCount(arguments.filenameAIM)
        filenameEVENT = arguments.filenameEVENT

        result = subprocess.run(
            [
                'ti',
                f'{os.path.realpath(os.path.dirname(__file__))}'
                + '/taichi_script.py',
            ],
            stdout=subprocess.PIPE,
            check=False,
        )

        forces = []
        for i in range(floorsCount):
            forces.append(FloorForces(recorderID=(i + 1)))

        # write the event file
        writeEVENT(forces, filenameEVENT, floorsCount)

    else:
        print('No RVs requested in StochasticWave.py')
        filenameEVENT = arguments.filenameEVENT
        result = subprocess.run(
            [
                'ti',
                f'{os.path.realpath(os.path.dirname(__file__))}'
                + '/taichi_script.py',
            ],
            stdout=subprocess.PIPE,
            check=False,
        )

        forces = []
        floorsCount = 1
        for i in range(floorsCount):
            forces.append(FloorForces(recorderID=(i + 1)))

        # write the event file
        writeEVENT(forces, filenameEVENT, floorsCount=floorsCount)
        # writeEVENT(forces, arguments.filenameEVENT)
