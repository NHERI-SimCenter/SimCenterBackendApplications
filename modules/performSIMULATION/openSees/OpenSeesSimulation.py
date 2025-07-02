#!/usr/bin/env python3  # noqa: D100, EXE001, RUF100

import os
import subprocess
import sys
import json

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

        # Check if the SAM file is a FemoraInput type
        with open(samName, 'r') as samFile:
            samData = json.load(samFile)
        with open(aimName, 'r') as aimFile:
            aimData = json.load(aimFile)
        subtype = samData.get("subType", None)
        runtype = aimData.get("runType", None)
        if subtype == "FemoraInput":
            main_script = samData.get("mainScript", None)
            if main_script is None:
                print("Error: mainScript not found in SAM file.")
                exit(1)

            # remove the .* from the end of the main script
            # it can have other . in the name but the last one is the extension
            ext = "." + main_script.rsplit('.')[-1]
            main_script_new = main_script[:-len(ext)]  + "_example.tcl"
            samData["mainScript"] = main_script_new


            shared_namespace = {}
            random_variables = samData.get("randomVar", [])
            command1 = "import femora as fm;\nimport os;\n"

            # Loop through the random variables and create the commands
            for rv in random_variables:
                rv_name = rv.get("name", "")
                rv_val  = rv.get("value", "")
                # add the command to the shared namespace
                command1 += f"{rv_name} = {rv_val};\n"
                command1 += f"print('Random Variable({rv_name}) =', {rv_name});\n"
                # command1 += f"print('Random Variable({rv_name}) =", rv_name, "');\n"
            print("Command1:", command1)

            with open(main_script, 'r') as main_script_file:
                command2 = main_script_file.read()

            command3 = f"""\nfm.export_to_tcl(filename="{main_script_new}")"""

            command = command1 + command2 + command3
            # try:
            #     exec(command, shared_namespace)
            #     print("exec() finished successfully.")
            # except Exception as e:
            #     print("exec() failed with error:", type(e).__name__, "-", e)
            #     # append the error to the workflow.err file
            #     with open("femora.err", 'w') as err_file:
            #         err_file.write(f"Error: {type(e).__name__} - {e}\n")

            original_stdout = sys.stdout
            original_stderr = sys.stderr
            import traceback
            # Redirect both stdout and stderr to the same file
            with open("femora.log", "w") as log_file:
                sys.stdout = log_file
                sys.stderr = log_file

                try:
                    exec(command, shared_namespace)
                    print("Femora model created successfully.")
                except Exception as e:
                    print("Femora failed with error:", type(e).__name__, "-", e)
                    traceback.print_exc()  # log full traceback to the same file

            # Restore original stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            with open(samName, 'w') as samFile:
                json.dump(samData, samFile)
        
        preprocessorCommand = f'"{scriptDir}/OpenSeesPreprocessor" {aimName} {samName} {evtName} {edpName} {simName} example.tcl > workflow.err1 2>&1'  # noqa: N806
        exit_code = subprocess.Popen(preprocessorCommand, shell=True).wait()  # noqa: S602
        # exit_code = subprocess.run(preprocessorCommand, shell=True).returncode # Maybe better for compatibility - jb
        # if not exit_code==0:
        #    exit(exit_code)

        # Run OpenSees
        if subtype == "FemoraInput":
            coresPerModel = samData.get("coresPerModel", 1)
            # Run OpenSees in parallel using mpirun
            if runtype == "runningLocal":
                openSeesCommand = f'mpirun -np {coresPerModel} OpenSeesMP example.tcl >> workflow.err 2>&1'
            elif runtype == "runningRemote":
                # For remote runs, we assume OpenSeesMP is available on the remote machine
                openSeesCommand = f'ibrun -n {coresPerModel} OpenSeesMP example.tcl >> workflow.err 2>&1'
            else:
                print(f"Error: Unsupported runType '{runtype}' in AIM file.")
                exit(1)
        else:
            openSeesCommand = 'OpenSees example.tcl >> workflow.err 2>&1'


        exit_code = subprocess.Popen(  # noqa: S602
            openSeesCommand,  # noqa: S607
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
