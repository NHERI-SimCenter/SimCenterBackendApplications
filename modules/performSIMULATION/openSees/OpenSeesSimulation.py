#!/usr/bin/env python3  # noqa: D100, EXE001, RUF100

import os
import subprocess
import sys

# from pathlib import Path


def main(args):  # noqa: D103
    # set filenames
    aimName = args[1]  # noqa: N806
    samName = args[3]  # noqa: N806
    evtName = args[5]  # noqa: N806
    edpName = args[7]  # noqa: N806
    simName = args[9]  # noqa: N806

    # remove path to AIM file, so recorders are not messed up
    #      .. AIM file ro be read is in current dir (copy elsewhere)
    aimName = os.path.basename(aimName)  # noqa: PTH119, N806
    scriptDir = os.path.dirname(os.path.realpath(__file__))  # noqa: PTH120, N806

    # aimName = Path(args[1]).name
    # scriptDir = Path(__file__).resolve().parent

    # If requesting random variables run getUncertainty
    # Otherwise, Run Opensees
    if '--getRV' in args:
        getUncertaintyCommand = f'"{scriptDir}/OpenSeesPreprocessor" {aimName} {samName} {evtName} {simName} > workflow.err 2>&1'  # noqa: N806
        exit_code = subprocess.Popen(getUncertaintyCommand, shell=True).wait()  # noqa: S602
        # exit_code = subprocess.run(getUncertaintyCommand, shell=True).returncode
        # if not exit_code==0:
        #    exit(exit_code)
    else:
        # Run preprocessor
        preprocessorCommand = f'"{scriptDir}/OpenSeesPreprocessor" {aimName} {samName} {evtName} {edpName} {simName} example.tcl > workflow.err 2>&1'  # noqa: N806
        exit_code = subprocess.Popen(preprocessorCommand, shell=True).wait()  # noqa: S602
        # exit_code = subprocess.run(preprocessorCommand, shell=True).returncode # Maybe better for compatibility - jb
        # if not exit_code==0:
        #    exit(exit_code)

        # Run OpenSees
        exit_code = subprocess.Popen(  # noqa: S602
            'OpenSees example.tcl  >> workflow.err 2>&1',  # noqa: S607
            shell=True,
        ).wait()
        # Maybe better for compatibility, need to doublecheck - jb
        # exit_code = subprocess.run("OpenSees example.tcl >> workflow.err 2>&1", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode

        # if os.path.isfile("./workflow.err"):
        #    with open("./workflow.err", 'r') as file:
        #        lines = file.readlines()
        #        # Iterate through each line
        #        for line in lines:
        #            # Check if the keyword exists in the line
        #            if "error" in line.lower():
        #                exit_code = -1
        #                exit(exit_code)

        # Run postprocessor
        postprocessorCommand = f'"{scriptDir}/OpenSeesPostprocessor" {aimName} {samName} {evtName} {edpName}  >> workflow.err 2>&1'  # noqa: N806
        exit_code = subprocess.Popen(postprocessorCommand, shell=True).wait()  # noqa: S602, F841
        # exit_code = subprocess.run(postprocessorCommand, shell=True).returncode # Maybe better for compatibility - jb
        # if not exit_code==0:
        #     exit(exit_code)


if __name__ == '__main__':
    main(sys.argv[1:])
