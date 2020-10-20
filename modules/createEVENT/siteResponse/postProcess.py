# This script create evt.j for workflow
import numpy as np
import json
import shutil
#from response_spectrum import response_spectrum


def postProcess(evtName):
    acc = np.loadtxt("out_tcl/acceleration.out")
    createProfile()
    #shutil.rmtree("out_tcl")  # remove output files to save space
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

def createProfile():
    # Sa
    periods = np.logspace(-3,1,100)

    acc = np.loadtxt('out_tcl/acceleration.out')
    acc = acc / 9.81
    time = acc[:-2,0]
    acc_surf = acc[:-2,-2]
    #p_surf, _, _, Sa_surf = response_spectrum(acc_surf, time[1]-time[0])

    #Sa_interp = np.interp(periods, p_surf, Sa_surf)
    #pga
    profile_pga = np.amax(acc[:, 1::4], axis=0)
    #peak shear strain
    strain = np.loadtxt('out_tcl/strain.out')
    profile_mStrain = np.amax(strain[:, 3::3], axis=0)
    #Ru
    stressData = np.loadtxt("out_tcl/stress.out")
    initStress = stressData[0,:]
    pwpData = np.loadtxt("out_tcl/porePressure.out")
    initPWP = pwpData[0,:]
    # read node info in elevation
    nodeInfo = np.loadtxt("nodesInfo.dat")
    profile_thickness= nodeInfo[-1, 2]

    #Convert from elevation to depth
    nodesY = profile_thickness - nodeInfo[0::2, 2]
    elemY = 0.5 * (nodesY[0:-1] + nodesY[1:])

    # ru based on change in sigma_v
    profile_ru_sigma = np.amax(1.0 - stressData[:, 2::3] / stressData[0, 2::3], axis=0)

    #xvals = np.linspace(0, profile_depth, 50)
    #Ru_interp = np.interp(xvals, np.flip(elemY), np.flip(profile_ru_sigma))

    #dataToWrite = np.append(Sa_interp, Ru_interp)
    #np.savetxt('Sa.out', np.transpose([periods, Sa_interp]))
    np.savetxt('Ru.out', np.transpose([elemY, profile_ru_sigma]))
    np.savetxt('maxStrain.out', np.transpose([elemY, profile_mStrain]))
    np.savetxt('pga.out', np.transpose([nodesY, profile_pga]))

if __name__ == "__main__":
    postProcess("EVENT.json")
