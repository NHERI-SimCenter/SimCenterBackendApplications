import json  # noqa: CPY001, D100, INP001
import subprocess  # noqa: S404
import sys

from calibration import createMaterial
from postProcess import postProcess


def main(args):  # noqa: D103
    # set filenames
    srtName = args[1]  # noqa: N806
    evtName = args[3]  # noqa: N806

    RFflag = False  # noqa: N806

    with open(srtName, encoding='utf-8') as json_file:  # noqa: PTH123
        data = json.load(json_file)

    for material in data['Events'][0]['materials']:
        if (
            material['type'] == 'PM4Sand_Random'
            or material['type'] == 'PDMY03_Random'
            or material['type'] == 'Elastic_Random'
        ):
            RFflag = True  # noqa: N806
            break
    if RFflag:
        # create material file based on 1D Gaussian field
        soilData = data['Events'][0]  # noqa: N806
        createMaterial(soilData)

    # Run OpenSees
    subprocess.Popen('OpenSees model.tcl', shell=True).wait()  # noqa: S602, S607

    # Run postprocessor to create EVENT.json
    postProcess(evtName)


if __name__ == '__main__':
    main(sys.argv[1:])
