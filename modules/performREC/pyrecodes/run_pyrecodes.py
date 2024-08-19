import json, os, shapely, argparse, sys, ujson, importlib  # noqa: INP001, I001, E401, D100
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path

# Delete below when pyrecodes can be installed as stand alone
import sys  # noqa: F811

sys.path.insert(0, '/Users/jinyanzhao/Desktop/SimCenterBuild/r2d_pyrecodes/')
from pyrecodes import main


def run_pyrecodes(rec_config, inputRWHALE, parallelType, mpiExec, numPROC):  # noqa: ARG001, C901, N803, D103
    # Initiate directory
    rec_ouput_dir = os.path.join(inputRWHALE['runDir'], 'Results', 'Recovery')  # noqa: PTH118
    if not os.path.exists(rec_ouput_dir):  # noqa: PTH110
        os.mkdir(rec_ouput_dir)  # noqa: PTH102

    # Find the realizations to run
    damage_input = rec_config.pop('DamageInput')
    realizations_to_run = select_realizations_to_run(damage_input, inputRWHALE)
    # Replace SimCenterDefault with correct path
    cmp_lib = rec_config['ComponentLibrary']
    if cmp_lib.startswith('SimCenterDefault'):
        cmp_lib_name = cmp_lib.split('/')[1]
        cmp_lib_dir = os.path.dirname(os.path.realpath(__file__))  # noqa: PTH120
        cmp_lib = os.path.join(cmp_lib_dir, cmp_lib_name)  # noqa: PTH118
        rec_config['ComponentLibrary'] = cmp_lib
    # loop through each realizations. Needs to be parallelized
    # Create the base of system configuration json
    system_configuration = create_system_configuration(rec_config)
    # Create the base of main json
    main_json = dict()  # noqa: C408
    main_json.update(
        {
            'ComponentLibrary': {
                'ComponentLibraryCreatorClass': 'JSONComponentLibraryCreator',
                'ComponentLibraryFile': rec_config['ComponentLibrary'],
            }
        }
    )

    # initialize a dict to accumulate recovery results stats
    result_det_path = os.path.join(  # noqa: PTH118
        inputRWHALE['runDir'],
        'Results',
        'Results_det.json',
    )
    with open(result_det_path, 'r') as f:  # noqa: PTH123, UP015
        results_det = json.load(f)
    result_agg = dict()  # noqa: C408
    resilience_results = dict()  # noqa: C408

    # Loop through realizations and run pyrecodes
    numP = 1  # noqa: N806
    procID = 0  # noqa: N806
    doParallel = False  # noqa: N806
    mpi_spec = importlib.util.find_spec('mpi4py')
    found = mpi_spec is not None
    if found and parallelType == 'parRUN':
        import mpi4py
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        numP = comm.Get_size()  # noqa: N806
        procID = comm.Get_rank()  # noqa: N806
        if numP < 2:  # noqa: PLR2004
            doParallel = False  # noqa: N806
            numP = 1  # noqa: N806
            procID = 0  # noqa: N806
        else:
            doParallel = True  # noqa: N806
    count = 0
    needsInitiation = True  # noqa: N806
    ind_in_rank = 0
    for ind, rlz_ind in enumerate(realizations_to_run):  # noqa: B007
        # Create a realization directory
        if count % numP == procID:
            rlz_dir = os.path.join(rec_ouput_dir, str(rlz_ind))  # noqa: PTH118
            if not os.path.exists(rlz_dir):  # noqa: PTH110
                os.mkdir(rlz_dir)  # noqa: PTH102

            # Update the system_configuration json
            damage_rlz_file = os.path.join(  # noqa: PTH118
                inputRWHALE['runDir'], 'Results', f'Results_{int(rlz_ind)}.json'
            )
            DamageInput = {  # noqa: N806
                'Type': 'R2DDamageInput',
                'Parameters': {'DamageFile': damage_rlz_file},
            }
            system_configuration.update({'DamageInput': DamageInput})

            # Write the system_configureation to a file
            system_configuration_file = os.path.join(  # noqa: PTH118
                rlz_dir, 'SystemConfiguration.json'
            )
            with open(system_configuration_file, 'w') as f:  # noqa: PTH123
                ujson.dump(system_configuration, f)
            # Update the main json
            main_json.update(
                {
                    'System': {
                        'SystemCreatorClass': 'ConcreteSystemCreator',
                        'SystemClass': 'BuiltEnvironmentSystem',
                        'SystemConfigurationFile': system_configuration_file,
                    }
                }
            )

            # Write the main json to a file
            main_file = os.path.join(rlz_dir, 'main.json')  # noqa: PTH118
            with open(main_file, 'w') as f:  # noqa: PTH123
                ujson.dump(main_json, f)

            system = main.run(main_file)

            system.calculate_resilience()

            # Append the recovery time to results_rlz
            if needsInitiation:
                needsInitiation = False  # noqa: N806
                num_of_rlz_per_rank = int(np.floor(len(realizations_to_run) / numP))
                if procID < len(realizations_to_run) % numP:
                    num_of_rlz_per_rank += 1
                # Initialize resilience_results
                resilience_results_buffer = dict()  # noqa: C408
                resilience_calculator_id = 0
                resilience_results.update(
                    {
                        'time_steps': list(range(0, system.MAX_TIME_STEP + 1))  # noqa: PIE808
                    }
                )
                resources_to_plot = system.resilience_calculators[
                    resilience_calculator_id
                ].system_supply.keys()
                for resource_name in resources_to_plot:
                    resilience_results_buffer.update(
                        {
                            resource_name: {
                                'Supply': np.zeros(
                                    [num_of_rlz_per_rank, system.MAX_TIME_STEP + 1]
                                ),
                                'Demand': np.zeros(
                                    [num_of_rlz_per_rank, system.MAX_TIME_STEP + 1]
                                ),
                                'Consumption': np.zeros(
                                    [num_of_rlz_per_rank, system.MAX_TIME_STEP + 1]
                                ),
                            }
                        }
                    )
                # Initialize result_agg
                result_agg_buffer = dict()  # noqa: C408
                for asset_type, item in results_det.items():
                    asset_type_result = dict()  # noqa: C408
                    for asset_subtype, asset_subtype_item in item.items():
                        asset_subtype_result = dict()  # noqa: C408
                        for aim_id, aim in asset_subtype_item.items():  # noqa: B007
                            asset_subtype_result.update(
                                {
                                    aim_id: {
                                        'RecoveryDuration': np.zeros(
                                            num_of_rlz_per_rank
                                        )
                                    }
                                }
                            )
                        asset_type_result.update(
                            {asset_subtype: asset_subtype_result}
                        )
                    result_agg_buffer.update({asset_type: asset_type_result})
                del results_det
            resilience_result_rlz_i = dict()  # noqa: C408
            for resource_name in resources_to_plot:
                resilience_result_rlz_i.update(
                    {
                        'time_steps': list(range(0, system.time_step + 1)),  # noqa: PIE808
                        resource_name: {
                            'Supply': system.resilience_calculators[
                                resilience_calculator_id
                            ].system_supply[resource_name][: system.time_step + 1],
                            'Demand': system.resilience_calculators[
                                resilience_calculator_id
                            ].system_demand[resource_name][: system.time_step + 1],
                            'Consumption': system.resilience_calculators[
                                resilience_calculator_id
                            ].system_consumption[resource_name][
                                : system.time_step + 1
                            ],
                        },
                    }
                )
                resilience_results_buffer[resource_name]['Supply'][
                    ind_in_rank, : system.time_step + 1
                ] = system.resilience_calculators[
                    resilience_calculator_id
                ].system_supply[resource_name][: system.time_step + 1]
                resilience_results_buffer[resource_name]['Demand'][
                    ind_in_rank, : system.time_step + 1
                ] = system.resilience_calculators[
                    resilience_calculator_id
                ].system_demand[resource_name][: system.time_step + 1]
                resilience_results_buffer[resource_name]['Consumption'][
                    ind_in_rank, : system.time_step + 1
                ] = system.resilience_calculators[
                    resilience_calculator_id
                ].system_consumption[resource_name][: system.time_step + 1]
            resilience_result_rlz_i_file = os.path.join(  # noqa: PTH118
                rlz_dir, 'ResilienceResult.json'
            )
            with open(resilience_result_rlz_i_file, 'w') as f:  # noqa: PTH123
                ujson.dump(resilience_result_rlz_i, f)
            result_file_name = os.path.join(  # noqa: PTH118
                inputRWHALE['runDir'],
                'Results',
                f'Results_{rlz_ind}.json',
            )
            with open(result_file_name, 'r') as f:  # noqa: PTH123, UP015
                results = json.load(f)
            for comp in system.components:
                if getattr(comp, 'r2d_comp', False) is True:
                    recovery_duration = (
                        getattr(comp, 'recoverd_time_step', system.MAX_TIME_STEP)
                        - system.DISASTER_TIME_STEP
                    )
                    recovery_duration = max(0, recovery_duration)
                    results[comp.asset_type][comp.asset_subtype][comp.aim_id].update(
                        {'Recovery': {'Duration': recovery_duration}}
                    )
                    result_agg_buffer[comp.asset_type][comp.asset_subtype][
                        comp.aim_id
                    ]['RecoveryDuration'][ind_in_rank] = recovery_duration
            with open(result_file_name, 'w') as f:  # noqa: PTH123
                ujson.dump(results, f)

            ind_in_rank += 1
        count = count + 1

    # wait for all to finish
    if doParallel:
        comm.Barrier()

    # if rank 0, gather result_agg and resilience_results, write to file
    # note that the gathered results doesn't follow the order in realization_to_run
    # but this order is not needed when calculating mean and std
    if doParallel:
        # gather results_agg
        for asset_type, item in result_agg_buffer.items():
            asset_type_result = dict()  # noqa: C408
            for asset_subtype, asset_subtype_item in item.items():
                asset_subtype_result = dict()  # noqa: C408
                for aim_id, aim in asset_subtype_item.items():  # noqa: B007
                    asset_subtype_result.update(
                        {
                            aim_id: {
                                'RecoveryDuration': comm.gather(
                                    result_agg_buffer[asset_type][asset_subtype],
                                    root=0,
                                )
                            }
                        }
                    )
                asset_type_result.update({asset_subtype: asset_subtype_result})
            result_agg.update({asset_type: asset_type_result})
            # gather resilience_resutls
        for resource_name in resources_to_plot:
            if procID == 0:
                resilience_results.update(
                    {
                        resource_name: {
                            'Supply': np.zeros(
                                [len(realizations_to_run), system.MAX_TIME_STEP + 1]
                            ),
                            'Demand': np.zeros(
                                [len(realizations_to_run), system.MAX_TIME_STEP + 1]
                            ),
                            'Consumption': np.zeros(
                                [len(realizations_to_run), system.MAX_TIME_STEP + 1]
                            ),
                        }
                    }
                )
            comm.gather(
                resilience_results_buffer[resource_name]['Supply'],
                resilience_results[resource_name]['Supply'],
                root=0,
            )
            comm.gather(
                resilience_results_buffer[resource_name]['Demand'],
                resilience_results[resource_name]['Demand'],
                root=0,
            )
            comm.gather(
                resilience_results_buffer[resource_name]['Consumption'],
                resilience_results[resource_name]['Consumption'],
                root=0,
            )
    else:
        for resource_name in resources_to_plot:
            resilience_results.update(
                {resource_name: resilience_results_buffer[resource_name]}
            )
        result_agg = result_agg_buffer

    if procID == 0:
        # Calculate stats of the results and add to results_det.json
        with open(result_det_path, 'r') as f:  # noqa: PTH123, UP015
            results_det = json.load(f)
        for asset_type, item in result_agg.items():
            for asset_subtype, asset_subtype_item in item.items():
                for aim_id, aim in asset_subtype_item.items():
                    if (
                        'R2Dres'  # noqa: SIM118
                        not in results_det[asset_type][asset_subtype][aim_id].keys()
                    ):
                        results_det[asset_type][asset_subtype][aim_id].update(
                            {'R2Dres': {}}
                        )
                    results_det[asset_type][asset_subtype][aim_id]['R2Dres'].update(
                        {
                            'R2Dres_mean_RecoveryDuration': aim[
                                'RecoveryDuration'
                            ].mean(),
                            'R2Dres_std_RecoveryDuration': aim[
                                'RecoveryDuration'
                            ].std(),
                        }
                    )
        with open(result_det_path, 'w') as f:  # noqa: PTH123
            ujson.dump(results_det, f)
        recovery_result_path = os.path.join(rec_ouput_dir, 'ResilienceResult.json')  # noqa: PTH118
        for resource_name in resources_to_plot:
            resilience_results[resource_name].update(
                {
                    'R2Dres_mean_Supply': resilience_results[resource_name]['Supply']
                    .mean(axis=0)
                    .tolist(),
                    'R2Dres_std_Supply': resilience_results[resource_name]['Supply']
                    .std(axis=0)
                    .tolist(),
                    'R2Dres_mean_Demand': resilience_results[resource_name]['Demand']
                    .mean(axis=0)
                    .tolist(),
                    'R2Dres_std_Demand': resilience_results[resource_name]['Demand']
                    .std(axis=0)
                    .tolist(),
                    'R2Dres_mean_Consumption': resilience_results[resource_name][
                        'Consumption'
                    ]
                    .mean(axis=0)
                    .tolist(),
                    'R2Dres_std_Consumption': resilience_results[resource_name][
                        'Consumption'
                    ]
                    .std(axis=0)
                    .tolist(),
                }
            )
            resilience_results[resource_name].pop('Supply')
            resilience_results[resource_name].pop('Demand')
            resilience_results[resource_name].pop('Consumption')

        with open(recovery_result_path, 'w') as f:  # noqa: PTH123
            ujson.dump(resilience_results, f)

    # Below are for development use
    from pyrecodes import GeoVisualizer as gvis  # noqa: N813

    geo_visualizer = gvis.R2D_GeoVisualizer(system.components)
    geo_visualizer.plot_component_localities()
    from pyrecodes import Plotter

    plotter_object = Plotter.Plotter()
    x_axis_label = 'Time step [day]'
    resources_to_plot = [
        'Shelter',
        'FunctionalHousing',
        'ElectricPower',
        'PotableWater',
    ]
    resource_units = ['[beds/day]', '[beds/day]', '[MWh/day]', '[RC/day]']
    # define which resilience calculator to use to plot the supply/demand/consumption of the resources
    # they are ordered as in the system configuration file
    resilience_calculator_id = 0
    for i, resource_name in enumerate(resources_to_plot):
        y_axis_label = f'{resource_name} {resource_units[i]} | {system.resilience_calculators[resilience_calculator_id].scope}'
        axis_object = plotter_object.setup_lor_plot_fig(x_axis_label, y_axis_label)
        time_range = system.time_step + 1
        time_steps_before_event = 10
        plotter_object.plot_single_resource(
            list(range(-time_steps_before_event, time_range)),
            resilience_results[resource_name]['R2Dres_mean_Supply'][:time_range],
            resilience_results[resource_name]['R2Dres_mean_Demand'][:time_range],
            resilience_results[resource_name]['R2Dres_mean_Consumption'][
                :time_range
            ],
            axis_object,
            warmup=time_steps_before_event,
        )
    print()  # noqa: T201


def create_system_configuration(rec_config):  # noqa: D103
    content_config = rec_config.pop('Content')
    system_configuration = rec_config.copy()
    if content_config['Creator'] == 'FromJsonFile':
        with open(content_config['FilePath'], 'r') as f:  # noqa: PTH123, UP015
            content = json.load(f)
        system_configuration.update({'Content': content})
    elif content_config['Creator'] == 'LocalityGeoJSON':
        # think how users can input RecoveryResourceSupplier and Resources
        pass
    return system_configuration


def select_realizations_to_run(damage_input, inputRWHALE):  # noqa: N803, D103
    rlzs_num = min(
        [
            item['ApplicationData']['Realizations']
            for _, item in inputRWHALE['Applications']['DL'].items()
        ]
    )
    rlzs_available = np.array(range(rlzs_num))
    if damage_input['Type'] == 'R2DDamageRealization':
        rlz_filter = damage_input['Parameters']['Filter']
        rlzs_requested = []
        for rlzs in rlz_filter.split(','):
            if '-' in rlzs:
                rlzs_low, rlzs_high = rlzs.split('-')
                rlzs_requested += list(range(int(rlzs_low), int(rlzs_high) + 1))
            else:
                rlzs_requested.append(int(rlzs))
        rlzs_requested = np.array(rlzs_requested)
        rlzs_in_available = np.in1d(rlzs_requested, rlzs_available)  # noqa: NPY201
        if rlzs_in_available.sum() != 0:
            rlzs_to_run = rlzs_requested[np.where(rlzs_in_available)[0]]
        else:
            rlzs_to_run = []
    if damage_input['Type'] == 'R2DDamageSample':
        sample_size = damage_input['Parameters']['SampleSize']
        seed = damage_input['Parameters']['SampleSize']
        if sample_size < rlzs_num:
            np.random.seed(seed)
            rlzs_to_run = np.sort(
                np.random.choice(rlzs_available, sample_size, replace=False)
            ).tolist()
        else:
            rlzs_to_run = np.sort(rlzs_available).tolist()
    return rlzs_to_run


if __name__ == '__main__':
    # Defining the command line arguments

    workflowArgParser = argparse.ArgumentParser(  # noqa: N816
        'Run Pyrecodes from the NHERI SimCenter rWHALE workflow for a set of assets.',
        allow_abbrev=False,
    )

    workflowArgParser.add_argument(
        '-c', '--configJsonPath', help='Configuration file for running perycode'
    )
    workflowArgParser.add_argument(
        '-i',
        '--inputRWHALEPath',
        help='Configuration file specifying the rwhale applications and data '
        'used',
    )
    workflowArgParser.add_argument(
        '-p',
        '--parallelType',
        default='seqRUN',
        help='How parallel runs: options seqRUN, parSETUP, parRUN',
    )
    workflowArgParser.add_argument(
        '-m',
        '--mpiexec',
        default='mpiexec',
        help='How mpi runs, e.g. ibrun, mpirun, mpiexec',
    )
    workflowArgParser.add_argument(
        '-n',
        '--numP',
        default='8',
        help='If parallel, how many jobs to start with mpiexec option',
    )

    # Parsing the command line arguments
    wfArgs = workflowArgParser.parse_args()  # noqa: N816

    # Calling the main workflow method and passing the parsed arguments
    numPROC = int(wfArgs.numP)  # noqa: N816

    with open(Path(wfArgs.configJsonPath).resolve(), 'r') as f:  # noqa: PTH123, UP015
        rec_config = json.load(f)

    with open(Path(wfArgs.inputRWHALEPath).resolve(), 'r') as f:  # noqa: PTH123, UP015
        inputRWHALE = json.load(f)  # noqa: N816
    run_pyrecodes(
        rec_config=rec_config,
        inputRWHALE=inputRWHALE,
        parallelType=wfArgs.parallelType,
        mpiExec=wfArgs.mpiexec,
        numPROC=numPROC,
    )

