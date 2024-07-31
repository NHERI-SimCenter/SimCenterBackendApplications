import argparse  # noqa: INP001, D100
import json
import os
from textwrap import wrap

from scipy import spatial


def ReadSMC(smcFilePath):  # noqa: ANN001, ANN201, N802, N803, D103
    with open(smcFilePath, 'r+') as smcFile:  # noqa: PTH123, N806
        series = []
        smcLines = smcFile.readlines()  # noqa: N806
        dT = 1.0 / float(smcLines[17].strip().split()[1])  # noqa: N806
        nCommentLines = int(smcLines[12].strip().split()[7])  # noqa: N806
        for line in smcLines[(27 + nCommentLines) :]:
            for value in wrap(line, 10, drop_whitespace=False):
                if value.strip():
                    series.append(float(value) / 100.0)  # noqa: PERF401

        return [series, dT]


def ReadCOSMOS(cosmosFilePath):  # noqa: ANN001, ANN201, N802, N803, D103
    with open(cosmosFilePath, 'r+') as cosmosFile:  # noqa: PTH123, N806
        series = []
        cosmosLines = cosmosFile.readlines()  # noqa: N806
        headerSize = int(cosmosLines[0][46:48])  # noqa: N806
        intSize = int(cosmosLines[headerSize][37:40])  # noqa: N806
        realSize = int(cosmosLines[headerSize + intSize + 1][34:37])  # noqa: N806
        commentSize = int(cosmosLines[headerSize + intSize + realSize + 2][0:4])  # noqa: N806
        totalHeader = headerSize + intSize + realSize + commentSize + 3  # noqa: N806
        recordSize = int(cosmosLines[totalHeader].strip().split()[0])  # noqa: N806
        dT = float(cosmosLines[37].strip().split()[1]) / 1000.0  # noqa: N806

        for line in cosmosLines[totalHeader + 1 : totalHeader + recordSize + 1]:
            series.append(float(line.strip()) / 100.0)  # noqa: PERF401

        return [series, dT]


def createEvent(recordsFolder, h1File, h2File, eventFilePath):  # noqa: ANN001, ANN201, N802, N803, D103
    if h1File.endswith('.smc'):
        h1, dt1 = ReadSMC(os.path.join(recordsFolder, h1File))  # noqa: PTH118
    else:
        h1, dt1 = ReadCOSMOS(os.path.join(recordsFolder, h1File))  # noqa: PTH118

    if h2File.endswith('.smc'):
        h2, dt2 = ReadSMC(os.path.join(recordsFolder, h2File))  # noqa: PTH118
    else:
        h2, dt2 = ReadCOSMOS(os.path.join(recordsFolder, h2File))  # noqa: PTH118

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
    event['name'] = h1File
    event['type'] = 'Seismic'
    event['description'] = h1File
    event['dT'] = dt1
    event['numSteps'] = len(h1)
    event['timeSeries'] = [seriesH1, seriesH2]
    event['pattern'] = [patternH1, patternH2]

    eventsDict = {}  # noqa: N806
    eventsDict['Events'] = [event]
    eventsDict['RandomVariables'] = []

    with open(eventFilePath, 'w') as eventFile:  # noqa: PTH123, N806
        json.dump(eventsDict, eventFile, indent=4)


def main():  # noqa: ANN201, D103
    # Input Argument Specifications
    gmArgsParser = argparse.ArgumentParser(  # noqa: N806
        'Characterize ground motion using seismic hazard analysis and record selection'
    )
    gmArgsParser.add_argument(
        '-filenameAIM', '--filenameAIM', required=True, help='Path to the BIM file'
    )
    gmArgsParser.add_argument(
        '-filenameEVENT',
        '--filenameEVENT',
        required=True,
        help='Path to the EVENT file',
    )
    gmArgsParser.add_argument(
        '-groundMotions',
        '--groundMotions',
        required=True,
        help='Path to the ground motions configuration file',
    )
    gmArgsParser.add_argument(
        '-recordsFolder',
        '--recordsFolder',
        required=True,
        help='Path to the ground motions records folder',
    )
    gmArgsParser.add_argument(
        '-getRV',
        '--getRV',
        action='store_true',
        help='Flag showing whether or not this call is to get the random variables definition',
    )

    # Parse the arguments
    gmArgs = gmArgsParser.parse_args()  # noqa: N806

    # Check getRV flag
    if not gmArgs.getRV:
        # We will use the template files so no changes are needed
        # We do not have any random variables for this event for now
        return 0

    # First let's process the arguments
    bimFilePath = gmArgs.filenameAIM  # noqa: N806
    eventFilePath = gmArgs.filenameEVENT  # noqa: N806
    gmConfigPath = gmArgs.groundMotions  # noqa: N806
    recordsFolder = gmArgs.recordsFolder  # noqa: N806

    with open(gmConfigPath) as gmConfigFile:  # noqa: PTH123, N806
        gmConfig = json.load(gmConfigFile)  # noqa: N806

    # We need to read the building location
    with open(bimFilePath) as bimFile:  # noqa: PTH123, N806
        bim = json.load(bimFile)
        location = [
            bim['GI']['location']['latitude'],
            bim['GI']['location']['longitude'],
        ]

    siteLocations = []  # noqa: N806
    for gm in gmConfig['GroundMotion']:
        siteLocations.append(  # noqa: PERF401
            [gm['Location']['Latitude'], gm['Location']['Longitude']]
        )

    # we need to find the nearest neighbor
    sitesTree = spatial.KDTree(siteLocations)  # noqa: N806

    nearest = sitesTree.query(location)
    nearestGM = gmConfig['GroundMotion'][nearest[1]]  # noqa: N806
    h1File = nearestGM['Records']['Horizontal1']  # noqa: N806
    h2File = nearestGM['Records']['Horizontal2']  # noqa: N806

    createEvent(os.path.abspath(recordsFolder), h1File, h2File, eventFilePath)  # noqa: RET503, PTH100


if __name__ == '__main__':
    main()
