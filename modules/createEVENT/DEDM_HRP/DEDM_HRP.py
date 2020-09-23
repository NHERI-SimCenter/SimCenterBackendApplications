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

# read the relevant parameters from the BIM file
with open(bimName, 'r') as f:
    bim_data = json.load(f)

incidenceAngle = bim_data['Events'][0]['incidenceAngle']
checkedPlan = bim_data['Events'][0]['checkedPlan']

if checkedPlan == 1 and incidenceAngle > 45:
    incidenceAngle = 90 - incidenceAngle

dataDir = os.getcwd()
scriptDir = os.path.dirname(os.path.realpath(__file__))

def parseDEDM_MatFile(matFileIn, windFileOut):
    print("HELLO PROCESSING - BRILLIANT PYTHON;)")
    mat_contents = sio.loadmat(matFileIn);
    dt = mat_contents['dt_ultimate'][0][0];
    print("dT=%s" % dt)

    windDirections = [windDir[0] for windDir in mat_contents['wind_directions']]

    for dirn, windAngle in enumerate(windDirections):

        if incidenceAngle != windAngle:
            continue

        # get forces
        FxForcesUltimate = mat_contents['full_scale_force_x_ultimate'][dirn];
        FyForcesUltimate = mat_contents['full_scale_force_y_ultimate'][dirn];
        MzForcesUltimate = mat_contents['full_scale_force_t_ultimate'][dirn];

        # Set number of floors & steps
        if FxForcesUltimate.ndim != 1:
            numFloor = FxForcesUltimate.shape[0]
            numSteps = FxForcesUltimate[0].size
        else:
            numFloor = 1
            numSteps = FxForcesUltimate.size

        windFileOutName = windFileOut + "." + str(windAngle) + ".json";

        event_output ={}

        event_output.update({
            'type': 'Wind',
            'name': windFileOutName,
            'dT'  : dt,
            'numSteps': numSteps,
            'timeSeries': [],
            'pattern': []
            })

        for floor in range(1, numFloor+1):

            for force_label, force_source, dof in zip(
                ['Fx', 'Mz', 'Fy'],
                [FxForcesUltimate, MzForcesUltimate, FyForcesUltimate],
                [1, 6, 2]):

                floorForces = force_source if numFloor is 1 else force_source[floor-1]

                ts_i = {
                    'name': f'{floor}_{force_label}',
                    'type': 'Value',
                    'dT': dt,
                    'numSteps': floorForces.size,
                    'data': floorForces.tolist()
                }

                event_output['timeSeries'].append(ts_i)

                pat_i = {
                    'name': f'{floor}_{force_label}',
                    'timeSeries': f'{floor}_{force_label}',
                    'type': 'WindFloorLoad',
                    'floor': str(floor),
                    'dof': dof,
                    'value': 1.0
                }

                event_output['pattern'].append(pat_i)

        with open(windFileOutName, 'w') as f:
            json.dump(event_output, f, indent=2)
               
    print('available wind directions: ', windDirections)


if "--getRV" in inputArgs:
    getDataFromDEDM_HRP = '"{}/DEDM_HRP" --filenameBIM {} --filenameEVENT {} --getRV'.format(scriptDir, bimName, evtName)
    subprocess.Popen(getDataFromDEDM_HRP, shell=True).wait()
    print("DONE. NOW PROCESSING RETURN")
    parseDEDM_MatFile("tmpSimCenterDEDM.mat",evtName)
    os.remove("tmpSimCenterDEDM.mat")
else:
    getDataFromDEDM_HRP = '"{}/DEDM_HRP" --filenameBIM {} --filenameEVENT {}'.format(scriptDir, bimName, evtName)
    subprocess.Popen(getDataFromDEDM_HRP, shell=True).wait()
  
