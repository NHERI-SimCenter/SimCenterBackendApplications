import sys
import os
import subprocess
import json
from calibration import createMaterial
from postProcess import postProcess

def main(args):

    # set filenames
    srtName = args[1]
    evtName = args[3]

    RFflag = False

    with open(srtName) as json_file:
        data = json.load(json_file)

    for material in data["Events"][0]["materials"]:
        if material["type"] == "PM4Sand_Random" or material["type"] == "PDMY03_Random" or material["type"] == "Elastic_Random":
            RFflag = True
            break
    if RFflag:
        #create material file based on 1D Gaussian field
        soilData = data["Events"][0]
        createMaterial(soilData)

    #Run OpenSees
    subprocess.Popen("OpenSees model.tcl", shell=True).wait()

    #Run postprocessor to create EVENT.json
    postProcess(evtName)

if __name__ == '__main__':

    main(sys.argv[1:])