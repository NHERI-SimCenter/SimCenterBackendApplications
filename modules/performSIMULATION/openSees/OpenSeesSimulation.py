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
        

        if subtype == "SSISimulation":
            # "building_type": "custom_3d_building"
            if samData.get("building_type", "") not in ["custom_3d_building"]:
                print(f"Error: building_type '{samData.get('building_type', '')}' is not supported for SSISimulation subtype.")
                print("Only 'custom_3d_building' is supported.")
                exit(1)
            if samData.get("soil_type", "") not in ["soil_foundation_type_1"]:
                print(f"Error: foundation_type '{samData.get('foundation_type', '')}' is not supported for SSISimulation subtype.")
                print("Only 'foundation_type_1' is supported.")
                exit(1)
            
            if samData.get("building_type", "") == "custom_3d_building" and samData.get("soil_type", "") == "soil_foundation_type_1":
                main_script = samData["structure_info"].get("model_file", None)
                main_script = os.path.basename(main_script)
                ext = "." + main_script.rsplit('.')[-1]
                main_script = main_script[:-len(ext)]  # remove the extension
                main_script_new = main_script  + "_example.tcl"
                if main_script is None:
                    print("Error: model_file not found in structure_info of SAM file.")
                    exit(1)
                from femora.components.simcenter.eeuq.soil_foundation_type_one import soil_foundation_type_one 
                num_cores = soil_foundation_type_one(model_filename=main_script_new,
                                                     info_file=samName,
                                                     EEUQ=True)
                num_cores = int(num_cores)
                samData["coresPerModel"] = num_cores
                samData["mainScript"] = main_script_new
           

            with open(samName, 'w') as samFile:
                json.dump(samData, samFile)


                

        
        preprocessorCommand = f'"{scriptDir}/OpenSeesPreprocessor" {aimName} {samName} {evtName} {edpName} {simName} example.tcl > workflow.err 2>&1'  # noqa: N806
        exit_code = subprocess.Popen(preprocessorCommand, shell=True).wait()  # noqa: S602
        # exit_code = subprocess.run(preprocessorCommand, shell=True).returncode # Maybe better for compatibility - jb
        # if not exit_code==0:
        #    exit(exit_code)

        # Run OpenSees
        if subtype == "FemoraInput" or subtype == "SSISimulation":
            coresPerModel = samData.get("coresPerModel", 1)
            
            # Run OpenSees in parallel using mpirun
            offset = 0
            numSamples = 0

            coresPerNode = aimData.get("coresPerNode", 1)
            nodeCount = aimData.get("nodeCount",1)
            totalCores = coresPerNode*nodeCount

            try:
                uq_data = aimData.get("UQ", {})
                print(f"uq_data {uq_data}")
                sampling = uq_data.get("samplingMethodData", {})
                numSamples = sampling.get("samples")
                print(f"numSamples {numSamples}")

                # Basic validation
                if numSamples is None:
                    raise KeyError("Missing one or more required keys: UQ.samplingMethodData.samples")

            except KeyError as e:
                print(f" Key Error Getting NumSamples for femora in OpenSeesSimulation: {e}")
            except Exception as e:
                print(f" Error Getting numSamples for femora in OpnSeesSimulation: {e}")


            print(f"numSamples {numSamples} coresPerModel {coresPerModel} totalCores {totalCores}")
            
            # offset depends on workdir number, workdirs start at 1
            # first 0 throgh numSamples-1 for dakota, rest for OpenSeesMP
            # numDakotaCores is how many dakota is started with: ibrun -n numDakoraCores dakota ...
            
            numDakotaCores=totalCores/(1+coresPerModel)
            cwd = os.getcwd()
            x = int(cwd.split('workdir.')[-1])
            offset = int(numDakotaCores + ((x-1)%numDakotaCores)*coresPerModel)
            
            if runtype == "runningLocal":

                if offset == 0:
                    openSeesCommand = f'mpirun -np {coresPerModel} OpenSeesMP example.tcl >> workflow.err 2>&1'
                else:
                    openSeesCommand = f'mpirun -o {offset} -np {coresPerModel} OpenSeesMP example.tcl >> workflow.err 2>&1'                
            elif runtype == "runningRemote":
                
                # For remote runs, we assume OpenSeesMP is available on the remote machine
                if offset == 0:
                    openSeesCommand = f'ibrun -n {coresPerModel} OpenSeesMP example.tcl >> workflow.err 2>&1'
                else:
                    openSeesCommand = f'ibrun -o {offset} -n {coresPerModel} OpenSeesMP example.tcl >> workflow.err 2>&1'
            else:
                print(f"Error: Unsupported runType '{runtype}' in AIM file.")
                exit(1)

            print(f" OpenSeesSimulation in workdir.{x}: {openSeesCommand}")
            
        else:
            openSeesCommand = 'OpenSees example.tcl >> workflow.err 2>&1'
            print(f" OpenSeesSimulation: {openSeesCommand}")


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

        postprocess_commands = []

        with open(edpName, 'r') as edpFile:
            edpData = json.load(edpFile)
            engdemand = edpData.get("EngineeringDemandParameters", [])
            for edp in engdemand:
                postprocessScript = edp.get("postprocessScript")
                if postprocessScript and postprocessScript.endswith(".py"):
                    args = " ".join(arg.get("type", "") for arg in edp.get("responses", []))
                    command = f'python {postprocessScript} {args} >> workflow.err 2>&1'
                    postprocess_commands.append(command)
        if exit_code == 0:
            for postprocess in postprocess_commands:
                subprocess.Popen(
                    postprocess,
                    shell=True
                ).wait()

        print("Postprocess commands:")
        print(postprocess_commands)





        # Run postprocessor
        postprocessorCommand = f'"{scriptDir}/OpenSeesPostprocessor" {aimName} {samName} {evtName} {edpName}  >> workflow.err 2>&1'  # noqa: N806
        exit_code = subprocess.Popen(postprocessorCommand, shell=True).wait()  # noqa: S602, F841
        # exit_code = subprocess.run(postprocessorCommand, shell=True).returncode # Maybe better for compatibility - jb
        # if not exit_code==0:
        #     exit(exit_code)


if __name__ == '__main__':
    main(sys.argv[1:])
