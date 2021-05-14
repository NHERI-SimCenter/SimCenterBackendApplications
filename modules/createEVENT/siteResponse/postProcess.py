# This script create evt.j for workflow
import numpy as np
import json
import os
import shutil

def postProcess(evtName):
    #acc = np.loadtxt("acceleration.out")
    #os.remove("acceleration.out")  # remove acceleration file to save space
    acc = np.loadtxt("out_tcl/acceleration.out")
    shutil.rmtree("out_tcl")  # remove output files to save space
    time = acc[:,0]
    acc_surf = acc[:,-2] / 9.81
    dT = time[1] - time[0]

    timeSeries = dict(
        name = "accel_X",
        type = "Value",
        dT = dT,
        data = acc_surf.tolist()
    )

    patterns = dict(
        type = "UniformAcceleration",
        timeSeries = "accel_X",
        dof = 1
    )

    evts = dict(
        RandomVariables = [],
        name = "SiteResponseTool",
        type = "Seismic",
        description = "Surface acceleration",
        dT = dT,
        numSteps = len(acc_surf),
        timeSeries = [timeSeries],
        pattern = [patterns]
    )

    dataToWrite = dict(Events = [evts])

    with open(evtName, "w") as outfile:
        json.dump(dataToWrite, outfile, indent=4)

    return 0


if __name__ == "__main__":
    postProcess("EVENT.json")
