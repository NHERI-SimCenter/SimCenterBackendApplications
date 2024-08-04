# This script create evt.j for workflow  # noqa: CPY001, D100, INP001
import json
import shutil

import numpy as np


def postProcess(evtName):  # noqa: N802, N803, D103
    # acc = np.loadtxt("acceleration.out")
    # os.remove("acceleration.out")  # remove acceleration file to save space
    acc = np.loadtxt('out_tcl/acceleration.out')
    shutil.rmtree('out_tcl')  # remove output files to save space
    time = acc[:, 0]
    acc_surf = acc[:, -2] / 9.81
    dT = time[1] - time[0]  # noqa: N806

    timeSeries = dict(name='accel_X', type='Value', dT=dT, data=acc_surf.tolist())  # noqa: C408, N806

    patterns = dict(type='UniformAcceleration', timeSeries='accel_X', dof=1)  # noqa: C408

    evts = dict(  # noqa: C408
        RandomVariables=[],
        name='SiteResponseTool',
        type='Seismic',
        description='Surface acceleration',
        dT=dT,
        numSteps=len(acc_surf),
        timeSeries=[timeSeries],
        pattern=[patterns],
    )

    dataToWrite = dict(Events=[evts])  # noqa: C408, N806

    with open(evtName, 'w') as outfile:  # noqa: PLW1514, PTH123
        json.dump(dataToWrite, outfile, indent=4)

    return 0


if __name__ == '__main__':
    postProcess('EVENT.json')
