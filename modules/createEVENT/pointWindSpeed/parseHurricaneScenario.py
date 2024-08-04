# python code to open the TPU .mat file  # noqa: CPY001, D100, EXE002, INP001
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


def parseMatFile(matFileIn, windFileOutName):  # noqa: N802, N803, D103
    file = open(windFileOutName, 'w')  # noqa: PLW1514, PTH123, SIM115
    mat_contents = sio.loadmat(matFileIn)
    print(mat_contents['wind'])  # noqa: T201
    windData = mat_contents['wind'][0][0]  # noqa: N806
    f = windData[0]
    lat = windData[1]
    long = windData[2]
    numLocations = lat.shape[0]  # noqa: N806
    print(lat.shape)  # noqa: T201
    file.write('{')
    file.write('"wind":[')
    for i in range(numLocations):
        locSpeed = f[i]  # noqa: N806
        locLat = lat[i]  # noqa: N806
        locLong = long[i]  # noqa: N806

        if i == numLocations - 1:
            file.write(
                '{"lat":%f,"long":%f,"windSpeed":%d}]' % (locLat, locLong, locSpeed)
            )
        else:
            file.write(
                '{"lat":%f,"long":%f,"windSpeed":%d},' % (locLat, locLong, locSpeed)
            )

    file.write('}')
    file.close()


if __name__ == '__main__':
    parseMatFile(matFileIN, jsonFileOUT)
