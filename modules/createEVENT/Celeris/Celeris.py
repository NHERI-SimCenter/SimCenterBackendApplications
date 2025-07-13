#!/usr/bin/env python3  # noqa: EXE001

"""
Author: Justin Bonus, 2024.

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
"""

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

# Check if taichi is installed before importing taichi
try:
    import taichi as ti
except ImportError:
    print('Taichi is not installed. Please install it using "pip install taichi".')  # noqa: T201
    print('Attempting to install taichi automatically for you...')  # noqa: T201
    print()  # noqa: T201
    # SYSEXECUTABLE = sys.executable  # noqa: N806
    # PYTHONPATH = os.environ.get('PYTHONPATH', '')  # noqa: N806
    # PYTHONHOME = os.environ.get('PYTHONHOME', '')  # noqa: N806
    # PYTHONSTARTUP = os.environ.get('PYTHONSTARTUP', '')  # noqa: N806
    # VIRTUAL_ENV = os.environ.get('VIRTUAL_ENV', '')  # noqa: N806
    # PIP_REQUIRE_VIRTUALENV = os.environ.get('PIP_REQUIRE_VIRTUALENV', '')  # noqa: N806
    # print('If you are using a virtual environment, make sure it is activated.')  # noqa: T201
    # print('Python executable being used to install taichi (i.e., sys.excutable):', SYSEXECUTABLE)  # noqa: T201
    # print('PYTHONPATH:', PYTHONPATH)
    # print('PYTHONHOME:', PYTHONHOME)  # noqa: T201
    # print('PYTHONSTARTUP:', PYTHONSTARTUP)
 
    # if VIRTUAL_ENV:
    #     print('VIRTUAL_ENV:', VIRTUAL_ENV)
    # else:
    #     print('No virtual environment detected. If you are using one, please activate it before running this script.')
 
    # if PIP_REQUIRE_VIRTUALENV:
    #     print('PIP_REQUIRE_VIRTUALENV:', PIP_REQUIRE_VIRTUALENV)
    # else:
    #     print('No PIP_REQUIRE_VIRTUALENV environment variable detected. If you are using a virtual environment, please activate it before running this script.')
    
    print()
    
    try:
        print('Try to install with $ python -m pip install taichi')
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'taichi'], check=False)  # noqa: S603
        try:
            import taichi as ti
        except ImportError:
            print('Taichi installation failed. Please install it manually.')  # noqa: T201
            # sys.exit(1)
    except:
        # Might be user permission issue
        print('Try to install with $ python -m pip install --user taichi')  # noqa: T201
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', 'taichi'], check=False)  # noqa: S603
            try:
                import taichi as ti
            except ImportError:
                print('Taichi installation failed. Please install it manually.')
                # sys.exit(1)
        except:
            print('Try to install with $ pip install taichi')  # noqa: T201
            try:
                subprocess.run(['pip', 'install', 'taichi'], check=False)  # noqa: S603
                try:
                    import taichi as ti
                except ImportError:
                    print('Taichi installation failed. Please install it manually.')
                    # sys.exit(1)
            except:
                print('ERROR: Cannot install taichi. There is likely an issue with you Python environment, OS, GLIBC, or pip installation.')  # noqa: T201
                print('INFO: Please manually install taichi into the Python environment specified in the desktop applications Files > Preferences tab')
                sys.exit(1)
    print()  # noqa: T201


class FloorForces:  # noqa: D101
    def __init__(self, recorderID=-1):  # noqa: N803
        if recorderID < 0 or recorderID is None:  # noqa: E701
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
            prependZero = True  # noqa: N806
            if prependZero:
                self.X.append(0.0)
                self.Y.append(0.0)
                self.Z.append(0.0)

            # Read in forces.[out or evt] file and add to EVENT.json
            # now using intermediary forces.evt for output of preceding Python calcs,
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
                        # Assume recorder IDs are sequential
                        # Often, the first ID (0) is set to 0.0 for all time-steps, as it usually maps to a rigid node at the structures base
                        if (j) == recorderID:
                            # Strip away leading / trailing white-space, Delimit by regex to capture " ", \s, "  ", tabs, etc.
                            # Each value should be a number, rep. the force on recorder j at a time-step i
                            clean_line = re.split(r';\s|;|,\s|,|\s+', strip_line)
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
                    print(  # noqa: T201
                        'Force time-series length: ',
                        len(self.X),
                        ', Max force (X,Y,Z): ',
                        max(self.X),
                        max(self.Y),
                        max(self.Z),
                        ', Min force (X,Y,Z): ',
                        min(self.X),
                        min(self.Y),
                        min(self.Z),
                        ', Last force (X,Y,Z): ',
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
    seriesName = 'WindForceSeries_' + str(floor) + direction  # noqa: N806
    patternName = 'WindForcePattern_' + str(floor) + direction  # noqa: N806

    pattern = {
        'name': patternName,
        'timeSeries': seriesName,
        'numSteps': len(force.X),
        'dT': dt,
        'dt': dt,
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
        'dT': dt,
        'dt': dt,
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
        addFloorForceToEvent(
            patternsArray, timeSeriesArray, floorForces, 'Y', it + 1
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
        'dT': dt,
        'dt': dt,
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


def GetCelerisScript(defaultScriptName = 'setrun.py'):  # noqa: N802, N803, D103
    defaultScriptPath = f'{os.path.realpath(os.path.dirname(__file__))}'  # noqa: N806, PTH120
    defaultScriptPath = os.path.join(defaultScriptPath, defaultScriptName)  # noqa: PTH118
    return defaultScriptPath  # noqa: PTH118


def dt(event=None):  # noqa: D102
    """
    Computes the time step based on the Courant criterion:
        dt = Courant * dx / sqrt(g * maxdepth)

    Returns:
        float: The computed time step.
    """
    if 'Courant_num' in event['config']:
        # Check is string
        if isinstance(event['config']['Courant_num'], str):
            Courant = 0.1
        else:
            Courant = event['config']['Courant_num']
    else:
        Courant = 0.1
        
    if 'base_depth' in event['config']:
        # Check is string
        if isinstance(event['config']['base_depth'], str):
            maxdepth = 1.0
        else:
            maxdepth = event['config']['base_depth']
    elif 'seaLevel' in event['config']:
        # Check is string
        if isinstance(event['config']['seaLevel'], str):
            maxdepth = 1.0
        else:
            maxdepth = event['config']['seaLevel']
    else:
        maxdepth = 1.0
    
    if (maxdepth <= 0.0):
        print('Warning: maxdepth is less than or equal to 0.0, setting it to 1.0. This will affect the time-step and simulation stability')  # noqa: T201
        maxdepth = 1.0
    dx = abs(event['config']['dx']) if 'dx' in event['config'] else 1.0
    gravity = abs(event['config']['g']) if 'g' in event['config'] else 9.80665  # Acceleration due to gravity in m/s^2
    dt = Courant * dx / np.sqrt(gravity * maxdepth)
    if dt <= 0.0:
        print('Warning: Computed dt is less than or equal to 0.0, setting it to 1.0e-3')  # noqa: T201
        dt = 1.0e-3
    return dt  # noqa: N806

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
    
    # parsing arguments
    arguments, unknowns = parser.parse_known_args()

    # Get json of filenameAIM
    filePath = arguments.filenameAIM  # noqa: N816
    with open(filePath, encoding='utf-8') as file:  # noqa: PTH123
        evt = json.load(file)
    file.close  # noqa: B018

    scriptName = GetCelerisScript()  # noqa: N816
    caseDirectory = './examples/CrescentCity'  # noqa: N816
    configFilename = 'config.json'  # noqa: N816
    bathymetryFilename = 'bathymetry.txt'  # noqa: N816
    waveFilename = 'wave.txt'  # noqa: N816

    for event in evt['Events']:
        # Redesign the input structure in backend CelerisAi later.
        # For now assume waveFile, bathymetryFile, configFile, etc. are in the same directory.
        caseDirectory = event['configFilePath']  # noqa: N816
        configDirectory = event['configFilePath']  # noqa: N816
        configFilename = event['configFile']  # noqa: N816
        bathymetryDirectory = event['bathymetryFilePath']  # noqa: N816
        bathymetryFilename = event['bathymetryFile']  # noqa: N816
        waveDirectory = event['waveFilePath']  # noqa: N816
        waveFilename = event['waveFile']  # noqa: N816
        
        configFilename = os.path.join(  # noqa: PTH118
            configDirectory, configFilename
        )  # noqa: N816, PTH118
        bathymetryFilename = os.path.join(  # noqa: PTH118
            bathymetryDirectory, bathymetryFilename
        )  # noqa: N816, PTH118
        waveFilename = os.path.join( # noqa: PTH118
            waveDirectory, waveFilename
        )  # noqa: N816, PTH118
        
        # Check if the config file exists
        if not os.path.exists(configFilename):  # noqa: PTH110
            print('Config file does not exist:', configFilename)
            # Use default config file
            configFilename = os.path.join(  # noqa: PTH118
                caseDirectory, 'config.json'
            )  # noqa: N816, PTH118
            print('Using default config file:', configFilename)  # noqa: T201
            
        # Check if the bathymetry file exists
        if not os.path.exists(bathymetryFilename):
            print('Bathymetry file does not exist:', bathymetryFilename)  # noqa: T201
            # Use default bathymetry file
            bathymetryFilename = os.path.join(  # noqa: PTH118
                caseDirectory, 'bathy.txt'
            ) # noqa: N816, PTH118
            print('Using default bathymetry file:', bathymetryFilename)
        
        # Check if the wave file exists
        if not os.path.exists(waveFilename):  # noqa: PTH110
            print('Wave file does not exist:', waveFilename)  # noqa: T201
            # Use default wave file
            waveFilename = os.path.join(  # noqa: PTH118
                caseDirectory, 'waves.txt'
            )
            print('Using default wave file:', waveFilename)  # noqa: T201
            
        # Determine dt for the force time series
        # Try to compute using Courant_num (CFL), otherwise look for dt in the config
        if 'Courant_num' in event['config']:
            print('Courant_num found in event file. Compute dt.')  # noqa: T201
            dt = dt(event=event)
        else:
            print('Courant_num not found in event file. Use provided dt')  # noqa: T201
            if 'dt' in event['config']:
                dt = event['config']['dt']  # noqa: N816
            else:
                print('dt not found in event file. Use default dt')  # noqa: T201
                dt = 1e-3  # noqa: N806

    print('Running Celeris with script:', scriptName)  # noqa: T201
    print('Running Celeris with directory:', caseDirectory)  # noqa: T201
    print('Running Celeris with config file:', configFilename)  # noqa: T201
    print('Running Celeris with bathymetry:', bathymetryFilename)  # noqa: T201
    print('Running Celeris with waves:', waveFilename)  # noqa: T201

    floorsCount = 1  # noqa: N816

    if arguments.getRV == True:  # noqa: E712
        print('RVs requested')  # noqa: T201
        # Read the number of floors
        floorsCount = GetFloorsCount(arguments.filenameAIM)  # noqa: N816
        filenameEVENT = arguments.filenameEVENT  # noqa: N816
        
        forces = []
        for i in range(floorsCount):
            forces.append(FloorForces(recorderID=(i + 1)))  # noqa: PERF401

        # write the event file
        writeEVENT(forces, filenameEVENT, floorsCount)

    else:
        print('No RVs requested')  # noqa: T201
        
        # filenameAIM = arguments.filenameAIM  # noqa: N816
        # Read in Events[0]["config"]
        configObj = evt['Events'][0]['config']  # noqa: N806
        # Write configObj to config_rv.json
        configRVFilename = 'config_rv.json'  # noqa: N816
        with open(caseDirectory + '/' + configRVFilename, 'w', encoding='utf-8') as file:
            json.dump(configObj, file)
        file.close
        
        filenameEVENT = arguments.filenameEVENT  # noqa: N816
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                sys.executable,
                scriptName,
                '-d',
                caseDirectory,
                '-f',
                configRVFilename,
                '-b',
                bathymetryFilename,
                '-w',
                waveFilename,
            ],
            stdout=subprocess.PIPE,
            check=False,
        )

        forces = []
        for i in range(floorsCount):
            forces.append(FloorForces(recorderID=(i + 1)))

        # write the event file
        writeEVENT(forces, filenameEVENT, floorsCount=floorsCount)
