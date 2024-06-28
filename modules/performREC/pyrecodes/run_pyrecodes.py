import json, os, shapely, argparse, sys, ujson, importlib
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
# Delete below when pyrecodes can be installed as stand alone
import sys
sys.path.insert(0, '/Users/jinyanzhao/Desktop/SimCenterBuild/r2d_pyrecodes/')
from pyrecodes import main


def run_pyrecodes(rec_config, inputRWHALE, parallelType, mpiExec, numPROC):

    # Initiate directory
    rec_ouput_dir = os.path.join(inputRWHALE['runDir'],"Results", "Recovery")
    if not os.path.exists(rec_ouput_dir):
        os.mkdir(rec_ouput_dir)

    # Find the realizations to run
    damage_input = rec_config.pop('DamageInput')
    realizations_to_run = select_realizations_to_run(\
        damage_input,inputRWHALE)
    
    # Replace SimCenterDefault with correct path
    cmp_lib = rec_config["ComponentLibrary"]
    if cmp_lib.startswith('SimCenterDefault'):
        cmp_lib_name = cmp_lib.split('/')[1]
        cmp_lib_dir = os.path.dirname(os.path.realpath(__file__))
        cmp_lib = os.path.join(cmp_lib_dir, cmp_lib_name)
        rec_config["ComponentLibrary"] = cmp_lib
    # loop through each realizations. Needs to be parallelized
    # Create the base of system configuration json
    system_configuration = create_system_configuration(rec_config)
    # Create the base of main json
    main_json = dict()
    main_json.update({"ComponentLibrary": {
        "ComponentLibraryCreatorClass": "JSONComponentLibraryCreator",
        "ComponentLibraryFile": rec_config["ComponentLibrary"]
    }})

    # initialize a dict to accumulate recovery results stats
    result_det_path = os.path.join(inputRWHALE['runDir'],"Results", 
                                    f"Results_det.json")
    with open(result_det_path, 'r') as f:
        results_det = json.load(f)
    result_agg = dict()
    resilience_results = dict()

    # Loop through realizations and run pyrecodes
    numP = 1
    procID = 0
    doParallel = False
    mpi_spec = importlib.util.find_spec("mpi4py")
    found = mpi_spec is not None
    if found and parallelType == 'parRUN':
        import mpi4py
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        numP = comm.Get_size()
        procID = comm.Get_rank()
        if numP < 2:
            doParallel = False
            numP = 1
            procID = 0
        else:
            doParallel = True
    count = 0
    needsInitiation = True
    ind_in_rank = 0
    for ind, rlz_ind in enumerate(realizations_to_run):
        # Create a realization directory
        if count % numP == procID:
            rlz_dir =  os.path.join(rec_ouput_dir,str(rlz_ind))
            if not os.path.exists(rlz_dir):
                os.mkdir(rlz_dir)

            # Update the system_configuration json
            damage_rlz_file = os.path.join(inputRWHALE['runDir'],"Results",\
                                    f"Results_{int(rlz_ind)}.json")
            DamageInput = {"Type": "R2DDamageInput",
                "Parameters": {"DamageFile": damage_rlz_file}}
            system_configuration.update({"DamageInput":DamageInput})

            # Write the system_configureation to a file
            system_configuration_file = os.path.join(rlz_dir, \
                                                    "SystemConfiguration.json")
            with open(system_configuration_file, 'w') as f:
                ujson.dump(system_configuration, f)
            
            # Update the main json
            main_json.update({"System": {
                "SystemCreatorClass": "ConcreteSystemCreator",
                "SystemClass": "BuiltEnvironmentSystem",
                "SystemConfigurationFile": system_configuration_file
            }})

            # Write the main json to a file
            main_file = os.path.join(rlz_dir, "main.json")
            with open(main_file, 'w') as f:
                ujson.dump(main_json, f)

            system = main.run(main_file)

            system.calculate_resilience()

            # Append the recovery time to results_rlz 
            if needsInitiation:
                needsInitiation = False
                num_of_rlz_per_rank = int(np.floor(len(realizations_to_run)/numP))
                if procID < len(realizations_to_run)%numP:
                    num_of_rlz_per_rank += 1
                # Initialize resilience_results
                resilience_results_buffer = dict()
                resilience_calculator_id = 0
                resilience_results.update({
                        "time_steps": list(range(0, system.MAX_TIME_STEP+1))
                    })
                resources_to_plot = system.resilience_calculators[resilience_calculator_id].system_supply.keys()
                for resource_name in resources_to_plot: 
                    resilience_results_buffer.update({
                        resource_name: {
                            "Supply": np.zeros([num_of_rlz_per_rank, system.MAX_TIME_STEP+1]),
                            "Demand": np.zeros([num_of_rlz_per_rank, system.MAX_TIME_STEP+1]),
                            "Consumption": np.zeros([num_of_rlz_per_rank, system.MAX_TIME_STEP+1])
                        }
                    })
                # Initialize result_agg
                result_agg_buffer = dict()
                for asset_type, item in results_det.items():
                    asset_type_result = dict()
                    for asset_subtype, asset_subtype_item in item.items():
                        asset_subtype_result = dict()
                        for aim_id, aim in asset_subtype_item.items():
                            asset_subtype_result.update({aim_id:{
                                "RecoveryDuration":np.zeros(num_of_rlz_per_rank)
                            }})
                        asset_type_result.update({asset_subtype:asset_subtype_result})
                    result_agg_buffer.update({asset_type:asset_type_result})
                del results_det
                
            resilience_result_rlz_i = dict()
            for resource_name in resources_to_plot:
                resilience_result_rlz_i.update({
                        "time_steps": list(range(0, system.time_step+1)),
                        resource_name: {
                            "Supply": system.resilience_calculators[resilience_calculator_id].system_supply[resource_name][:system.time_step+1],
                            "Demand": system.resilience_calculators[resilience_calculator_id].system_demand[resource_name][:system.time_step+1],
                            "Consumption": system.resilience_calculators[resilience_calculator_id].system_consumption[resource_name][:system.time_step+1]
                        }
                    }
                    )
                resilience_results_buffer[resource_name]['Supply'][ind_in_rank,:system.time_step+1] = \
                    system.resilience_calculators[resilience_calculator_id].system_supply[resource_name][:system.time_step+1]
                resilience_results_buffer[resource_name]['Demand'][ind_in_rank,:system.time_step+1] = \
                    system.resilience_calculators[resilience_calculator_id].system_demand[resource_name][:system.time_step+1]
                resilience_results_buffer[resource_name]['Consumption'][ind_in_rank,:system.time_step+1] = \
                    system.resilience_calculators[resilience_calculator_id].system_consumption[resource_name][:system.time_step+1]
            resilience_result_rlz_i_file = os.path.join(rlz_dir, "ResilienceResult.json")
            with open(resilience_result_rlz_i_file, 'w') as f:
                ujson.dump(resilience_result_rlz_i, f)
            result_file_name = os.path.join(inputRWHALE['runDir'],"Results", 
                                            f"Results_{rlz_ind}.json")
            with open(result_file_name, 'r') as f:
                results = json.load(f)
            for comp in system.components:
                if getattr(comp, 'r2d_comp', False) is True:
                    recovery_duration = getattr(comp, 'recoverd_time_step',system.MAX_TIME_STEP) - \
                        system.DISASTER_TIME_STEP
                    recovery_duration = max(0, recovery_duration)
                    results[comp.asset_type][comp.asset_subtype][comp.aim_id].update({
                        "Recovery": {"Duration":recovery_duration}
                    })
                    result_agg_buffer[comp.asset_type][comp.asset_subtype][comp.aim_id]\
                        ['RecoveryDuration'][ind_in_rank] = recovery_duration
            with open(result_file_name, 'w') as f:
                ujson.dump(results, f)

            ind_in_rank += 1
        count = count + 1

    # wait for all to finish
    if doParallel:
        comm.Barrier()

    # if rank 0, gather result_agg and resilience_results, write to file
    # note that the gathered results dosen't follow the order in realization_to_run
    # but this order is not needed when calculating mean and std
    if doParallel:
        # gather results_agg
        for asset_type, item in result_agg_buffer.items():
            asset_type_result = dict()
            for asset_subtype, asset_subtype_item in item.items():
                asset_subtype_result = dict()
                for aim_id, aim in asset_subtype_item.items():
                    asset_subtype_result.update({aim_id:{
                        "RecoveryDuration":comm.gather(result_agg_buffer[asset_type][asset_subtype], root=0)
                    }})
                asset_type_result.update({asset_subtype:asset_subtype_result})
            result_agg.update({asset_type:asset_type_result})
            # gather resilience_resutls
        for resource_name in resources_to_plot:
            if procID == 0:
                resilience_results.update({
                resource_name: {
                    "Supply": np.zeros([len(realizations_to_run), system.MAX_TIME_STEP+1]),
                    "Demand": np.zeros([len(realizations_to_run), system.MAX_TIME_STEP+1]),
                    "Consumption": np.zeros([len(realizations_to_run), system.MAX_TIME_STEP+1])
                    }
                })
            comm.gather(resilience_results_buffer[resource_name]["Supply"],
                                          resilience_results[resource_name]["Supply"], root=0)
            comm.gather(resilience_results_buffer[resource_name]["Demand"],
                                          resilience_results[resource_name]["Demand"], root=0)
            comm.gather(resilience_results_buffer[resource_name]["Consumption"],
                                          resilience_results[resource_name]["Consumption"], root=0)
    else:
        for resource_name in resources_to_plot: 
            resilience_results.update({
                resource_name: resilience_results_buffer[resource_name]
            })
        result_agg = result_agg_buffer

    if procID==0:
    # Calculate stats of the results and add to results_det.json
        with open(result_det_path, 'r') as f:
            results_det = json.load(f)
        for asset_type, item in result_agg.items():
            for asset_subtype, asset_subtype_item in item.items():
                for aim_id, aim in asset_subtype_item.items():
                    if 'R2Dres' not in results_det[asset_type][asset_subtype][aim_id].keys():
                        results_det[asset_type][asset_subtype][aim_id].update({'R2Dres':{}})
                    results_det[asset_type][asset_subtype][aim_id]['R2Dres'].update({
                        "R2Dres_mean_RecoveryDuration":aim['RecoveryDuration'].mean(),
                        "R2Dres_std_RecoveryDuration":aim['RecoveryDuration'].std()
                    })
        with open(result_det_path, 'w') as f:
            ujson.dump(results_det, f)
    
        recovery_result_path = os.path.join(rec_ouput_dir, "ResilienceResult.json")
        for resource_name in resources_to_plot: 
            resilience_results[resource_name].update({
                'R2Dres_mean_Supply':resilience_results[resource_name]['Supply'].mean(axis=0).tolist(),
                'R2Dres_std_Supply':resilience_results[resource_name]['Supply'].std(axis=0).tolist(),
                'R2Dres_mean_Demand':resilience_results[resource_name]['Demand'].mean(axis=0).tolist(),
                'R2Dres_std_Demand':resilience_results[resource_name]['Demand'].std(axis=0).tolist(),
                'R2Dres_mean_Consumption':resilience_results[resource_name]['Consumption'].mean(axis=0).tolist(),
                'R2Dres_std_Consumption':resilience_results[resource_name]['Consumption'].std(axis=0).tolist()
            })
            resilience_results[resource_name].pop("Supply")
            resilience_results[resource_name].pop("Demand")
            resilience_results[resource_name].pop("Consumption")

        
        with open(recovery_result_path, 'w') as f:
            ujson.dump(resilience_results, f)

    # Below are for development use
    from pyrecodes import GeoVisualizer as gvis
    geo_visualizer = gvis.R2D_GeoVisualizer(system.components)
    geo_visualizer.plot_component_localities()
    from pyrecodes import Plotter
    plotter_object = Plotter.Plotter()
    x_axis_label = 'Time step [day]'
    resources_to_plot = ['Shelter', 'FunctionalHousing', 'ElectricPower', 'PotableWater']
    resource_units = ['[beds/day]', '[beds/day]', '[MWh/day]', '[RC/day]']
    # define which resilience calculator to use to plot the supply/demand/consumption of the resources
    # they are ordered as in the system configuration file
    resilience_calculator_id = 0
    for i, resource_name in enumerate(resources_to_plot):    
        y_axis_label = f'{resource_name} {resource_units[i]} | {system.resilience_calculators[resilience_calculator_id].scope}'
        axis_object = plotter_object.setup_lor_plot_fig(x_axis_label, y_axis_label)    
        time_range = system.time_step+1
        time_steps_before_event = 10 # 
        plotter_object.plot_single_resource(list(range(-time_steps_before_event, time_range)), 
                                             resilience_results[resource_name]['R2Dres_mean_Supply'][:time_range], 
                                             resilience_results[resource_name]['R2Dres_mean_Demand'][:time_range],
                                             resilience_results[resource_name]['R2Dres_mean_Consumption'][:time_range],
                                             axis_object, warmup=time_steps_before_event)
    print()
def create_system_configuration(rec_config):
    content_config = rec_config.pop('Content')
    system_configuration = rec_config.copy()
    if content_config['Creator'] == 'FromJsonFile':
        with open(content_config['FilePath'], 'r') as f:
            content = json.load(f)
        system_configuration.update({"Content":content})
    elif content_config['Creator'] == 'LocalityGeoJSON':
        # think how users can input RecoveryResourceSupplier and Resources
        pass
        
    return system_configuration


def select_realizations_to_run(damage_input, inputRWHALE):
    rlzs_num = min([item['ApplicationData']['Realizations'] \
                       for _, item in inputRWHALE['Applications']['DL'].items()])
    rlzs_available = np.array(range(rlzs_num))
    if damage_input['Type'] == 'R2DDamageRealization':
        rlz_filter = damage_input['Parameters']['Filter']
        rlzs_requested = []
        for rlzs in rlz_filter.split(','):
            if "-" in rlzs:
                rlzs_low, rlzs_high = rlzs.split("-")
                rlzs_requested += list(range(int(rlzs_low), int(rlzs_high)+1))
            else:
                rlzs_requested.append(int(rlzs))
        rlzs_requested = np.array(rlzs_requested)
        rlzs_in_available = np.in1d(rlzs_requested, rlzs_available)
        if rlzs_in_available.sum() != 0:
            rlzs_to_run = rlzs_requested[
                np.where(rlzs_in_available)[0]]
        else:
            rlzs_to_run = []
    if damage_input['Type'] == 'R2DDamageSample':
        sample_size = damage_input['Parameters']['SampleSize']
        seed = damage_input['Parameters']['SampleSize']
        if sample_size < rlzs_num:
            np.random.seed(seed)
            rlzs_to_run = np.sort(np.random.choice(rlzs_available, sample_size,\
                                    replace = False)).tolist()
        else:
            rlzs_to_run = np.sort(rlzs_available).tolist()
    return rlzs_to_run

if __name__ == '__main__':

    #Defining the command line arguments

    workflowArgParser = argparse.ArgumentParser(
        "Run Pyrecodes from the NHERI SimCenter rWHALE workflow for a set of assets.",
        allow_abbrev=False)

    workflowArgParser.add_argument("-c", "--configJsonPath",
        help="Configuration file for running perycode")
    workflowArgParser.add_argument("-i", "--inputRWHALEPath",
        help="Configuration file specifying the rwhale applications and data "
             "used")
    workflowArgParser.add_argument("-p", "--parallelType",
        default='seqRUN',
        help="How parallel runs: options seqRUN, parSETUP, parRUN")
    workflowArgParser.add_argument("-m", "--mpiexec",
        default='mpiexec',
        help="How mpi runs, e.g. ibrun, mpirun, mpiexec")
    workflowArgParser.add_argument("-n", "--numP",
        default='8',
        help="If parallel, how many jobs to start with mpiexec option") 

    #Parsing the command line arguments
    wfArgs = workflowArgParser.parse_args()

    #Calling the main workflow method and passing the parsed arguments
    numPROC = int(wfArgs.numP)

    with open(Path(wfArgs.configJsonPath).resolve(), 'r') as f:
        rec_config = json.load(f)
    with open(Path(wfArgs.inputRWHALEPath).resolve(), 'r') as f:
        inputRWHALE = json.load(f)
    
    run_pyrecodes(rec_config=rec_config,\
         inputRWHALE=inputRWHALE,
         parallelType = wfArgs.parallelType,
         mpiExec = wfArgs.mpiexec,
         numPROC = numPROC)
    
    