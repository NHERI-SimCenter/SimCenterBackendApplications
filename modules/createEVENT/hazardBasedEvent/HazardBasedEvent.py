# This python script process the input and will use it to run SHA and ground motion selection  # noqa: CPY001, D100, INP001
# In addition to providing the event file

import glob
import json
import os
import re
import subprocess  # noqa: S404
import sys


def computeScenario(gmConfig, location):  # noqa: N802, N803, D103
    scriptDir = os.path.dirname(os.path.realpath(__file__))  # noqa: PTH120, N806
    eqHazardPath = f'{scriptDir}/GMU/EQHazard.jar'  # noqa: N806
    simulateIMPath = f'{scriptDir}/GMU/SimulateIM'  # noqa: N806
    selectRecordPath = f'{scriptDir}/GMU/SelectRecord'  # noqa: N806
    recordDatabasePath = f'{scriptDir}/GMU/NGAWest2-1000.csv'  # noqa: N806

    # Separate Selection Config
    selectionConfig = gmConfig['RecordSelection']  # noqa: N806
    del gmConfig['RecordSelection']

    gmConfig['Site'] = {}
    gmConfig['Site']['Type'] = 'SingleLocation'
    gmConfig['Site']['Location'] = {}
    gmConfig['Site']['Location']['Latitude'] = location[0]
    gmConfig['Site']['Location']['Longitude'] = location[1]

    # Adding the required output
    gmConfig['IntensityMeasure']['EnableJsonOutput'] = True
    with open(  # noqa: PTH123
        './HazardWorkDir/Hazard_Scenario.json', 'w', encoding='utf-8'
    ) as hazardFile:  # noqa: N806
        json.dump(gmConfig, hazardFile, indent=4)

    # Now we need to run the EQHazard Process
    hazardCommand = [  # noqa: N806
        'java',
        '-jar',
        eqHazardPath,
        './HazardWorkDir/Hazard_Scenario.json',
        './HazardWorkDir/Hazard_Output.json',
    ]
    hazardResult = subprocess.call(hazardCommand)  # noqa: S603, N806

    if hazardResult != 0:
        sys.stderr.write('Hazard analysis failed!')
        return -1

    # Now we need to run the SimulateIM Process
    # First we create a simulation config
    simConfig = {  # noqa: N806
        'GroundMotions': {'File': './HazardWorkDir/Hazard_Output.json'},
        'NumSimulations': 1,
        'SpatialCorrelation': True,
    }

    with open(  # noqa: PTH123
        './HazardWorkDir/Sim_Config.json', 'w', encoding='utf-8'
    ) as simConfigFile:  # noqa: N806
        json.dump(simConfig, simConfigFile, indent=4)
    simulateCommand = [  # noqa: N806
        simulateIMPath,
        './HazardWorkDir/Sim_Config.json',
        './HazardWorkDir/Hazard_Sim.json',
    ]
    simResult = subprocess.call(simulateCommand)  # noqa: S603, N806

    if simResult != 0:
        sys.stderr.write('Intensity measure simulation failed!')
        return -2

    # Now we can run record selection
    #
    selectionConfig['Target']['File'] = './HazardWorkDir/Hazard_Sim.json'
    selectionConfig['Database']['File'] = recordDatabasePath
    with open(  # noqa: PTH123
        './HazardWorkDir/Selection_Config.json', 'w', encoding='utf-8'
    ) as selectionConfigFile:  # noqa: N806
        json.dump(selectionConfig, selectionConfigFile, indent=4)
    selectionCommand = [  # noqa: N806
        selectRecordPath,
        './HazardWorkDir/Selection_Config.json',
        './HazardWorkDir/Records_Selection.json',
    ]
    simResult = subprocess.call(selectionCommand)  # noqa: S603, N806

    if simResult != 0:  # noqa: RET503
        sys.stderr.write('Intensity measure simulation failed!')
        return -2


def readNGAWest2File(ngaW2FilePath, scaleFactor):  # noqa: N802, N803, D103
    series = []
    dt = 0.0
    with open(ngaW2FilePath) as recordFile:  # noqa: N806, PLW1514, PTH123
        canRead = False  # We need to process the header first  # noqa: N806
        for line in recordFile:
            if canRead:
                series.extend(
                    [float(value) * scaleFactor * 9.81 for value in line.split()]
                )

            elif 'NPTS=' in line:
                dt = float(
                    re.match(r'NPTS=.+, DT=\s+([0-9\.]+)\s+SEC', line).group(1)
                )
                canRead = True  # noqa: N806

    return series, dt


def createNGAWest2Event(rsn, scaleFactor, recordsFolder, eventFilePath):  # noqa: N802, N803, D103
    pattern = os.path.join(recordsFolder, 'RSN') + str(rsn) + '_*.AT2'  # noqa: PTH118
    recordFiles = glob.glob(pattern)  # noqa: PTH207, N806
    if len(recordFiles) != 2:  # noqa: PLR2004
        print(  # noqa: T201
            'Error finding NGA West 2 files.\n'
            f'Please download the files for record {rsn} '
            f'from NGA West 2 website and place them in the records folder ({recordsFolder})'
        )
        exit(-1)  # noqa: PLR1722

    h1, dt1 = readNGAWest2File(recordFiles[0], scaleFactor)
    h2, dt2 = readNGAWest2File(recordFiles[1], scaleFactor)

    patternH1 = {}  # noqa: N806
    patternH1['type'] = 'UniformAcceleration'
    patternH1['timeSeries'] = 'accel_X'
    patternH1['dof'] = 1

    patternH2 = {}  # noqa: N806
    patternH2['type'] = 'UniformAcceleration'
    patternH2['timeSeries'] = 'accel_Y'
    patternH2['dof'] = 2

    seriesH1 = {}  # noqa: N806
    seriesH1['name'] = 'accel_X'
    seriesH1['type'] = 'Value'
    seriesH1['dT'] = dt1
    seriesH1['data'] = h1

    seriesH2 = {}  # noqa: N806
    seriesH2['name'] = 'accel_Y'
    seriesH2['type'] = 'Value'
    seriesH2['dT'] = dt2
    seriesH2['data'] = h2

    event = {}
    event['name'] = 'NGAW2_' + str(rsn)
    event['type'] = 'Seismic'
    event['description'] = (
        'NGA West 2 record '
        + str(rsn)
        + ' scaled by a factor of '
        + str(scaleFactor)
    )
    event['dT'] = dt1
    event['numSteps'] = len(h1)
    event['timeSeries'] = [seriesH1, seriesH2]
    event['pattern'] = [patternH1, patternH2]
    event['units'] = {'length': 'm', 'time': 'sec'}

    eventsDict = {}  # noqa: N806
    eventsDict['Events'] = [event]
    eventsDict['RandomVariables'] = []

    with open(eventFilePath, 'w', encoding='utf-8') as eventFile:  # noqa: PTH123, N806
        json.dump(eventsDict, eventFile, indent=4)


def main():  # noqa: D103
    inputArgs = sys.argv  # noqa: N806

    # Process only if --getRV is passed
    if '--getRV' not in inputArgs:
        sys.exit(0)

    # First let's process the arguments
    argBIM = inputArgs.index('--filenameAIM') + 1  # noqa: N806
    bimFilePath = inputArgs[argBIM]  # noqa: N806
    argEVENT = inputArgs.index('--filenameEVENT') + 1  # noqa: N806
    eventFilePath = inputArgs[argEVENT]  # noqa: N806

    # Ensure a hazard cache folder exist
    if not os.path.exists('./HazardWorkDir'):  # noqa: PTH110
        os.mkdir('./HazardWorkDir')  # noqa: PTH102

    with open(bimFilePath, encoding='utf-8') as bimFile:  # noqa: PTH123, N806
        bim = json.load(bimFile)
        location = [
            bim['GeneralInformation']['location']['latitude'],
            bim['GeneralInformation']['location']['longitude'],
        ]

    scriptDir = os.path.dirname(os.path.realpath(__file__))  # noqa: PTH120, N806
    recordsFolder = f'{scriptDir}/GMU/NGAWest2Records'  # noqa: N806

    computeScenario(bim['Events'][0]['GroundMotion'], location)

    # We need to read the building location

    # Now we can start processing the event
    with open('./HazardWorkDir/Records_Selection.json') as selectionFile:  # noqa: N806, PLW1514, PTH123
        recordSelection = json.load(selectionFile)  # noqa: N806

    selectedRecord = recordSelection['GroundMotions'][0]  # noqa: N806
    rsn = selectedRecord['Record']['Id']
    scaleFactor = selectedRecord['ScaleFactor']  # noqa: N806

    createNGAWest2Event(rsn, scaleFactor, recordsFolder, eventFilePath)


if __name__ == '__main__':
    main()
