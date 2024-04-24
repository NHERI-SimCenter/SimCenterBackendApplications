# python code to open the TPU .mat file
# and put data into a SimCenter JSON file for
# wind tunnel data

import sys
import os
import subprocess
import json
import stat
import shutil
import numpy as np
import scipy.io as sio
from pprint import pprint

inputArgs = sys.argv

print ("Number of arguments: %d" % len(sys.argv))
print ("The arguments are: %s"  %str(sys.argv))

# set filenames
matFileIN = sys.argv[1]
jsonFileOUT = sys.argv[2]

dataDir = os.getcwd()
scriptDir = os.path.dirname(os.path.realpath(__file__))

def parseTPU_HighRise_MatFile(matFileIn, windFileOutName):

    file = open(windFileOutName,"w", encoding='utf-8');
    file.write("{\n");

    mat_contents = sio.loadmat(matFileIn);
    depth = mat_contents['Building_depth'][0][0];
    height = mat_contents['Building_height'][0][0];
    breadth = mat_contents['Building_breadth'][0][0];
    period = mat_contents['Sample_period'][0][0];
    frequency = mat_contents['Sample_frequency'][0][0];
    angle = mat_contents['Wind_direction_angle'][0][0];
    #uH = mat_contents['Uh_AverageWindSpeed'][0][0];
    uH = float(mat_contents['Uh_AverageWindSpeed'][0]);
    print(uH)
    print(depth)
    print(height)
    file.write("\"windSpeed\":%f," % uH);
    file.write("\"depth\":%f," % depth);
    file.write("\"height\":%f," % height);
    file.write("\"breadth\":%f," % breadth);
    file.write("\"period\":%f," % period);
    file.write("\"units\":{\"length\":\"m\",\"time\":\"sec\"},");
    file.write("\"frequency\":%f," % frequency);
    file.write("\"incidenceAngle\":%f," % angle);
    file.write("\"tapLocations\": [");

    locations = mat_contents['Location_of_measured_points'];
    numLocations = locations.shape[1];

    # get xMax and yMax .. assuming first sensor is 1m from building edge
    # location on faces cannot be obtained from the inputs, at least not with 
    # current documentation, awaing email from TPU

    xMax = max(locations[0])+1
    yMax = max(locations[1])+1
    
    for loc in range(0, numLocations):
        tag = locations[2][loc]
        xLoc = locations[0][loc]
        yLoc = locations[1][loc]
        face = locations[3][loc]

        X = xLoc
        Y = yLoc
        if (face == 2):
            xLoc = X - breadth
        elif (face == 3):
            xLoc = X - breadth - depth
        elif (face == 4):
            xLoc = X - 2*breadth - depth
        
        if (loc == numLocations-1):
            file.write("{\"id\":%d,\"xLoc\":%f,\"yLoc\":%f,\"face\":%d}]" % (tag, xLoc, yLoc, face))
        else:
            file.write("{\"id\":%d,\"xLoc\":%f,\"yLoc\":%f,\"face\":%d}," % (tag, xLoc, yLoc, face))


    file.write(",\"pressureCoefficients\": [");
    coefficients = mat_contents['Wind_pressure_coefficients'];
    numLocations = coefficients.shape[1];
    numValues = coefficients.shape[0];

    for loc in range(0, numLocations):
        file.write("{\"id\": %d , \"data\":[" % (loc+1))
        for i in range(0, numValues-1):
            file.write("%f," % coefficients[i,loc])
        if (loc != numLocations-1):
            file.write("%f]}," % coefficients[numValues-1,loc])
        else:
            file.write("%f]}]" % coefficients[numValues-1,loc])

    file.write("}")
    file.close()

if __name__ == '__main__':    
    parseTPU_HighRise_MatFile(matFileIN,jsonFileOUT)

  
