# python code to open the TPU .mat file  # noqa: INP001, D100
# and put data into a SimCenter JSON file for
# wind tunnel data

import os
import sys

import scipy.io as sio

inputArgs = sys.argv  # noqa: N816

print('Number of arguments: %d' % len(sys.argv))  # noqa: T201
print('The arguments are: %s' % str(sys.argv))  # noqa: T201, UP031

# set filenames
matFileIN = sys.argv[1]  # noqa: N816
jsonFileOUT = sys.argv[2]  # noqa: N816

dataDir = os.getcwd()  # noqa: PTH109, N816
scriptDir = os.path.dirname(os.path.realpath(__file__))  # noqa: PTH120, N816


def parseTPU_HighRise_MatFile(matFileIn, windFileOutName):  # noqa: N802, N803, D103
    file = open(windFileOutName, 'w', encoding='utf-8')  # noqa: SIM115, PTH123
    file.write('{\n')
    mat_contents = sio.loadmat(matFileIn)
    depth = mat_contents['Building_depth'][0][0]
    height = mat_contents['Building_height'][0][0]
    breadth = mat_contents['Building_breadth'][0][0]
    period = mat_contents['Sample_period'][0][0]
    frequency = mat_contents['Sample_frequency'][0][0]
    angle = mat_contents['Wind_direction_angle'][0][0]
    # uH = mat_contents['Uh_AverageWindSpeed'][0][0];
    uH = float(mat_contents['Uh_AverageWindSpeed'][0])  # noqa: N806
    print(uH)  # noqa: T201
    print(depth)  # noqa: T201
    print(height)  # noqa: T201
    file.write('"windSpeed":%f,' % uH)  # noqa: UP031
    file.write('"depth":%f,' % depth)  # noqa: UP031
    file.write('"height":%f,' % height)  # noqa: UP031
    file.write('"breadth":%f,' % breadth)  # noqa: UP031
    file.write('"period":%f,' % period)  # noqa: UP031
    file.write('"units":{"length":"m","time":"sec"},')
    file.write('"frequency":%f,' % frequency)  # noqa: UP031
    file.write('"incidenceAngle":%f,' % angle)  # noqa: UP031
    file.write('"tapLocations": [')
    locations = mat_contents['Location_of_measured_points']
    numLocations = locations.shape[1]  # noqa: N806
    # get xMax and yMax .. assuming first sensor is 1m from building edge
    # location on faces cannot be obtained from the inputs, at least not with
    # current documentation, awaing email from TPU

    xMax = max(locations[0]) + 1  # noqa: N806, F841
    yMax = max(locations[1]) + 1  # noqa: N806, F841

    for loc in range(numLocations):
        tag = locations[2][loc]
        xLoc = locations[0][loc]  # noqa: N806
        yLoc = locations[1][loc]  # noqa: N806
        face = locations[3][loc]

        X = xLoc  # noqa: N806
        Y = yLoc  # noqa: N806, F841
        if face == 2:  # noqa: PLR2004
            xLoc = X - breadth  # noqa: N806
        elif face == 3:  # noqa: PLR2004
            xLoc = X - breadth - depth  # noqa: N806
        elif face == 4:  # noqa: PLR2004
            xLoc = X - 2 * breadth - depth  # noqa: N806

        if loc == numLocations - 1:
            file.write(
                '{"id":%d,"xLoc":%f,"yLoc":%f,"face":%d}]' % (tag, xLoc, yLoc, face)
            )
        else:
            file.write(
                '{"id":%d,"xLoc":%f,"yLoc":%f,"face":%d},' % (tag, xLoc, yLoc, face)
            )

    file.write(',"pressureCoefficients": [')
    coefficients = mat_contents['Wind_pressure_coefficients']
    numLocations = coefficients.shape[1]  # noqa: N806
    numValues = coefficients.shape[0]  # noqa: N806
    for loc in range(numLocations):
        file.write('{"id": %d , "data":[' % (loc + 1))
        for i in range(numValues - 1):
            file.write('%f,' % coefficients[i, loc])  # noqa: UP031
        if loc != numLocations - 1:
            file.write('%f]},' % coefficients[numValues - 1, loc])  # noqa: UP031
        else:
            file.write('%f]}]' % coefficients[numValues - 1, loc])  # noqa: UP031

    file.write('}')
    file.close()


if __name__ == '__main__':
    parseTPU_HighRise_MatFile(matFileIN, jsonFileOUT)
