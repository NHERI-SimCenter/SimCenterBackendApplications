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

# set filenames
bimName = sys.argv[2]
evtName = sys.argv[4]

dataDir = os.getcwd()
scriptDir = os.path.dirname(os.path.realpath(__file__))

def parseDEDM_MatFile(matFileIn, windFileOut):
    print("HELLO PROCESSING")
    mat_contents = sio.loadmat(matFileIn);
    dt = mat_contents['dt_ultimate'][0][0];
    print("dT=%s" % dt)
    windDirections = mat_contents['wind_directions'];
    numDirections = windDirections.size;
    for dirn in range(0, numDirections):
        # get forces
        FxForcesUltimate = mat_contents['full_scale_force_x_ultimate'][dirn];
        FyForcesUltimate = mat_contents['full_scale_force_y_ultimate'][dirn];
        MzForcesUltimate = mat_contents['full_scale_force_t_ultimate'][dirn];
        # Set number of floors
        numFloor = 1
        numSteps = 0
        if FxForcesUltimate.ndim != 1:
            numFloor = FxForcesUltimate.shape[0]
            numSteps = FxForcesUltimate[0].size
        else:
            numSteps = FxForcesUltimate.size

        windFileOutName = windFileOut + "." + str(windDirections[dirn][0]) + ".json";
        file = open(windFileOutName,"w");
        file.write("{\n");
        file.write("\"type\":\"Wind\",\n");
        file.write("\"name\":\"" + windFileOutName + "\",\n");
        file.write("\"dT\":" + str(dt) + ",\n");
        file.write("\"numSteps\":" + str(numSteps) + ",\n");
        file.write("\"timeSeries\":[\n");
        for floor in range(1, numFloor+1):
            floorForces = FxForcesUltimate if numFloor is 1 else FxForcesUltimate[floor-1]
            
            file.write("{\"name\":\"" + str(floor) + "_Fx\",\n");            
            file.write("\"type\":\"Value\",\n");            
            file.write("\"dT\":" + str(dt) + ",\n");      
            file.write("\"numSteps\":" + str(floorForces.size) + ",\n");
            file.write("\"data\":[\n");
            numForces = floorForces.size;
            
            for i in range(0, numForces-1):
                file.write(str(floorForces[i]) + ",")
            file.write(str(floorForces[numForces-1]))
            file.write("]\n},\n");


            floorForces = MzForcesUltimate if numFloor is 1 else MzForcesUltimate[floor-1]
            
            file.write("{\"name\":\"" + str(floor) + "_Mz\",\n");            
            file.write("\"type\":\"Value\",\n");            
            file.write("\"dT\":" + str(dt) + ",\n");      
            file.write("\"numSteps\":" + str(floorForces.size) + ",\n");
            file.write("\"data\":[\n");
            numForces = floorForces.size;
            
            for i in range(0, numForces-1):
                file.write(str(floorForces[i]) + ",")
            file.write(str(floorForces[numForces-1]))
            file.write("]\n},\n");


            floorForces = FyForcesUltimate if numFloor is 1 else FyForcesUltimate[floor-1]            
            # floorForces = yForcesUltimate[floor-1]
            file.write("{\"name\":\"" + str(floor) + "_Fy\",\n");            
            file.write("\"type\":\"Value\",\n");            
            file.write("\"dT\":" + str(dt) + ",\n");            
            file.write("\"data\":[\n");
            numForces = floorForces.size;

            for i in range(0, numForces-1):
                file.write(str(floorForces[i]) + ",")
            file.write(str(floorForces[numForces-1]))
            file.write("]}\n");

            if (floor != numFloor):
                file.write(",\n");
            
        file.write("],\n");

        file.write("\"pattern\":[\n");
        for floor in range(1, numFloor+1):
            file.write("{\"name\":\"" + str(floor) + "_Fx\",\n");            
            file.write("\"timeSeries\":\"" + str(floor) + "_Fx\",\n");            
            file.write("\"type\":\"WindFloorLoad\",\n");            
            file.write("\"floor\":\"" + str(floor) + "\",\n");
            file.write("\"dof\":1,\n");
            file.write("\"value\":1.0\n},\n");

            file.write("{\"name\":\"" + str(floor) + "_Mz\",\n");            
            file.write("\"timeSeries\":\"" + str(floor) + "_Mz\",\n");            
            file.write("\"type\":\"WindFloorLoad\",\n");            
            file.write("\"floor\":\"" + str(floor) + "\",\n");
            file.write("\"dof\":6,\n");
            file.write("\"value\":1.0\n},\n");

            file.write("{\"name\":\"" + str(floor) + "_Fy\",\n");            
            file.write("\"timeSeries\":\"" + str(floor) + "_Fy\",\n");            
            file.write("\"type\":\"WindFloorLoad\",\n");            
            file.write("\"floor\":\"" + str(floor) + "\",\n");
            file.write("\"dof\":2,\n");
            file.write("\"value\":1.0}\n");
            
            if (floor != numFloor):
                file.write(",\n");

        file.write("]\n");

        
        file.write("}\n");
        file.close() 
        
        # create a json object for the event
               
    print (dt)
    print (windDirections)


if "--getRV" in inputArgs:
    getDataFromDEDM_HRP = '"{}/DEDM_HRP" --filenameBIM {} --filenameEVENT {} --getRV'.format(scriptDir, bimName, evtName)
    subprocess.Popen(getDataFromDEDM_HRP, shell=True).wait()
    print("DONE. NOW PROCESSING RETURN")
    parseDEDM_MatFile("tmpSimCenterDEDM.mat",evtName)
    os.remove("tmpSimCenterDEDM.mat")
else:
    getDataFromDEDM_HRP = '"{}/DEDM_HRP" --filenameBIM {} --filenameEVENT {}'.format(scriptDir, bimName, evtName)
    subprocess.Popen(getDataFromDEDM_HRP, shell=True).wait()
  
