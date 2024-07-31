# python code to open the TPU .mat file
# and put data into a SimCenter JSON file for
# wind tunnel data

import os
import sys

import scipy.io as sio

inputArgs = sys.argv

print('Number of arguments: %d' % len(sys.argv))
print('The arguments are: %s' % str(sys.argv))

# set filenames
matFileIN = sys.argv[1]
jsonFileOUT = sys.argv[2]

dataDir = os.getcwd()
scriptDir = os.path.dirname(os.path.realpath(__file__))


def parseMatFile(matFileIn, windFileOutName):
    file = open(windFileOutName, 'w')
    mat_contents = sio.loadmat(matFileIn)
    print(mat_contents['wind'])
    windData = mat_contents['wind'][0][0]
    f = windData[0]
    lat = windData[1]
    long = windData[2]
    numLocations = lat.shape[0]
    print(lat.shape)
    file.write('{')
    file.write('"wind":[')
    for i in range(numLocations):
        locSpeed = f[i]
        locLat = lat[i]
        locLong = long[i]

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
