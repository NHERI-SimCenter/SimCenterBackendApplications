import json, os, shapely, argparse, sys, ujson, importlib  # noqa: INP001, I001, E401, D100
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import shutil

# Add the path to the pyrecodes package
import pyrecodes
import pyrecodes.main
from pyrecodes.geovisualizer.r2d_geovisualizer import R2D_GeoVisualizer
from pyrecodes.plotter.concrete_plotter import ConcretePlotter

def select_realizations_to_run(damage_input, run_dir):
    """
    Select the realizations to run based on the damage input and available realizations.

    Parameters
    ----------
    damage_input : dict
        Dictionary containing the damage input parameters.
    run_dir : str
        Directory where the results are stored.

    Returns
    -------
    list
        List of realizations to run.
    """
    # Get the available realizations
    results_files = [f for f in os.listdir(run_dir) if f.startswith('Results_') and f.endswith('.json')]
    rlzs_available = sorted([
        int(file.split('_')[1].split('.')[0])
        for file in results_files
        if file.split('_')[1].split('.')[0].isnumeric()
    ])
    # Get the number of realizations
    if damage_input['Type'] == 'SpecificRealization':
        rlz_filter = damage_input['Parameters']['Filter']
        rlzs_requested = []

        if "," not in rlz_filter:
            rlz_filter = rlz_filter + ","

        for rlzs in rlz_filter.split(','):
            if rlzs == '':
                continue

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
    if damage_input['Type'] == 'SampleFromRealizations':
        sample_size = damage_input['Parameters']['SampleSize']
        seed = damage_input['Parameters']['SampleSeed']
        if sample_size <= len(rlzs_available):
            np.random.seed(seed)
            rlzs_to_run = np.sort(
                np.random.choice(rlzs_available, sample_size, replace=False)
            ).tolist()
        else:
            msg = 'Sample size is larger than the number of available realizations'
            raise ValueError(msg)

    return rlzs_to_run
def run_one_realization(main_file, rlz, rwhale_run_dir, system_config):
    """
    Run a single realization of the pyrecodes simulation.

    Parameters
    ----------
    main_file (str): Path to the main configuration file.
    rlz (int): Realization number.
    rwhale_run_dir (str): Directory where the results are stored.
    system_config (dict): System configuration dictionary.
    """
    # Run the pyrecodes
    system = pyrecodes.main.run(main_file)
    system.calculate_resilience(print_output=False)

    # Add the component recovery time to the Results_rlz.json file
    with Path(rwhale_run_dir / f'Results_{rlz}.json').open() as f:
        results_rlz = json.load(f)
    all_recovery_time = list(system.resilience_calculators[1].\
        component_recovery_times.values())
    for ind, comp in enumerate(system.components):
        if getattr(comp, 'r2d_comp', False) is True:
            asset_type = comp.asset_type
            asset_subtype = comp.asset_subtype
            asset_id = comp.general_information['AIM_id']
            results_rlz[asset_type][asset_subtype][asset_id]['Recovery'] = {
                'Time': all_recovery_time[ind]
            }
    # Write the results to a file in the current realization workdir
    with (Path(f'Results_{rlz}.json')).open('w') as f:
        json.dump(results_rlz, f)

    # Create a gif of the recovery process
    geo_visualizer = R2D_GeoVisualizer(system.components)
    # time_step_list = list(range(0, system.time_step, 1))
    time_step_list = list(np.linspace(0, system.time_step, min(system.time_step, 20)).astype(int))
    for time_step in time_step_list:
        fig, ax = plt.subplots(figsize=(10, 10))
        geo_visualizer.create_current_state_figure(time_step, ax=ax)
        legend = ax.get_legend()
        legend.set(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f'status_at_time_step_{time_step}.png', dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
    geo_visualizer.create_recovery_gif(time_step_list, file_name = \
                                'status_at_time_step_TIME_STEP.png', fps=2)

    # create a plot of the supply and demand recovery of the first resource
    plotter_object = ConcretePlotter()
    first_resource = next(iter(system_config['Resources'].keys()))
    first_unit = system_config['Resources'][first_resource].get('Unit', f'unit_{first_resource}')
    resources_to_plot = [first_resource]
    units_to_plot = [first_unit]
    if 'PotableWater' in system_config['Resources']:
        resources_to_plot.append('PotableWater')
        units_to_plot.append(system_config['Resources']['PotableWater'].get('Unit', 'unit_PotableWater'))
    for recource, unit in zip(resources_to_plot, units_to_plot):
        y_axis_label = f'{recource} {unit} | {system.resilience_calculators[0].scope}'
        x_axis_label = 'Time step [day]'
        axis_object = plotter_object.setup_lor_plot_fig(x_axis_label, y_axis_label)
        time_range = system.time_step+1
        time_steps_before_event = 10
        plotter_object.plot_single_resource(list(range(-time_steps_before_event, time_range)), system.resilience_calculators[0].system_supply[recource][:time_range],
                                        system.resilience_calculators[0].system_demand[recource][:time_range],
                                        system.resilience_calculators[0].system_consumption[recource][:time_range], axis_object, warmup=time_steps_before_event,
                                        show = False
                                        )
        plotter_object.save_current_figure(savename = f'{recource}_supply_demand_consumption.png')
    return True

def modify_system_config_conent(system_config, locality_geojson, rwhale_run_dir):
    """
    Modify the system configuration content with updated paths for locality geojson and R2DJSONFile_Info.

    Parameters
    ----------
    system_config : dict
        The system configuration dictionary.
    locality_geojson : str
        Path to the locality geojson file.
    rwhale_run_dir : str
        Directory where the results are stored.

    Returns
    -------
    dict
        The modified system configuration dictionary.
    """
    content_config = system_config['Content']
    for locality_value in content_config.values():
        # Change the locality geojson path to the GeoJSON in rwahle resource directory.
        # This is needed for the remote run to work
        coordinates = locality_value['Coordinates']
        if 'GeoJSON' in coordinates:
            coordinates['GeoJSON']['Filename'] = locality_geojson
        # Change the R2DJSONFile_Info path to the results_det file in rwahle resource directory.
        # This is needed for the remote run to work
        components = locality_value['Components']
        for infra_value in components.values():
            for subsystem in infra_value:
                subsystem_name = next(iter(subsystem.keys()))
                subsytesm_value = subsystem[subsystem_name]
                if 'R2DJSONFile_Info' in subsytesm_value['Parameters']:
                    subsytesm_value['Parameters']['R2DJSONFile_Info'] =str(
                    rwhale_run_dir / 'Results_det.json')
    return system_config

def modify_system_config_rewet_distribution(system_config, inp_file, rlz_run_dir):
    """
    Modify the system configuration to update the INP file path for REWETDistributionModel.

    Parameters
    ----------
    system_config : dict
        The system configuration dictionary.
    inp_file : str
        Path to the INP file.

    Returns
    -------
    dict
        The modified system configuration dictionary.
    """
    resources_config = system_config['Resources']
    for resouces in resources_config.values():
        distribution_model = resouces['DistributionModel']
        if distribution_model['ClassName'] == 'REWETDistributionModel':
            distribution_model['Parameters']['INPFile'] = inp_file
            if not (rlz_run_dir / 'rewet_results').exists():
                (rlz_run_dir / 'rewet_results').mkdir()
            if not (rlz_run_dir / 'rewet_temp').exists():
                (rlz_run_dir / 'rewet_temp').mkdir()
            distribution_model['Parameters']['Results_folder'] = str(rlz_run_dir / 'rewet_results')
            distribution_model['Parameters']['Temp_folder'] = str(rlz_run_dir / 'rewet_temp')
    return system_config


def modify_main_file(main_file_dict, component_library, run_dir):
    """
    Modify the main configuration file with updated paths for the component library and system configuration.

    Parameters
    ----------
    main_file_dict : dict
        The main configuration dictionary.
    component_library : str
        Path to the component library file.
    rwhale_run_dir : str
        Directory where the results are stored.

    Returns
    -------
    bool
        True if the modification is successful.
    """
    main_file_dict['System']['SystemConfigurationFile'] = str(
            run_dir / 'SystemConfiguration.json')
    main_file_dict['ComponentLibrary']['ComponentLibraryFile'] = component_library
    main_file_path = str(run_dir / 'main.json')
    with Path(main_file_path).open('w') as f:
        json.dump(main_file_dict, f)
    return main_file_path

def create_agg_results_dict(results_det_path):
    """
    Create an aggregated results dictionary from the detailed results file.

    Parameters
    ----------
    results_det_path : Path
        Path to the detailed results JSON file.

    Returns
    -------
    dict
        Aggregated results dictionary.
    """
    with results_det_path.open() as f:
        results_det = json.load(f)
    results_agg = {}
    for asset_type, asset_type_dict in results_det.items():
            results_agg[asset_type] = {}
            for asset_subtype, asset_subtype_dict in asset_type_dict.items():
                results_agg[asset_type][asset_subtype] = {}
                for asset_id in asset_subtype_dict:
                    results_agg[asset_type][asset_subtype][asset_id] = {}
                    results_agg[asset_type][asset_subtype][asset_id].update(
                        {'RecoveryDuration': []}
                    )
    return results_agg

def append_to_results_agg(results_agg, results_rlz_path):
    """
    Append recovery duration data from a realization results file to the aggregated results dictionary.

    Parameters
    ----------
    results_agg : dict
        Aggregated results dictionary.
    results_rlz_path : Path
        Path to the realization results JSON file.

    Returns
    -------
    dict
        Updated aggregated results dictionary.
    """
    with results_rlz_path.open() as f:
        results_rlz = json.load(f)
    for asset_type, asset_type_dict in results_rlz.items():
        for asset_subtype, asset_subtype_dict in asset_type_dict.items():
            for asset_id, asset_id_dict in asset_subtype_dict.items():
                if 'Recovery' in asset_id_dict:
                    results_agg[asset_type][asset_subtype][asset_id]['RecoveryDuration'].append(
                        asset_id_dict['Recovery']['Time']
                    )
                else:
                    results_agg[asset_type][asset_subtype][asset_id]['RecoveryDuration'].append(
                        np.inf
                    )
    return results_agg

def aggregate_results_to_det(results_agg, results_det_path):
    """
    Aggregate recovery duration data from multiple realizations into the detailed results file.

    Parameters
    ----------
    results_agg : dict
        Aggregated results dictionary.
    results_det_path : Path
        Path to the detailed results JSON file.
    """
    with results_det_path.open() as f:
        results_det = json.load(f)

    for asset_type, asset_type_dict in results_agg.items():
        for asset_subtype, asset_subtype_dict in asset_type_dict.items():
            for asset_id, asset_id_dict in asset_subtype_dict.items():
                mean_recovery_duration = np.mean(asset_id_dict['RecoveryDuration'])
                std_recovery_duration = np.std(asset_id_dict['RecoveryDuration'])
                results_det[asset_type][asset_subtype][asset_id]['R2Dres'].update(
                    {'R2Dres_mean_RecoveryDuration': mean_recovery_duration}
                )
                results_det[asset_type][asset_subtype][asset_id]['R2Dres'].update(
                    {'R2Dres_std_RecoveryDuration': std_recovery_duration}
                )
    with results_det_path.open('w') as f:
        json.dump(results_det, f)


def run_pyrecodes(  # noqa: C901
        main_file,
        system_config_file,
        component_library,
        locality_geojson,
        rewet_inp_file,
        r2d_run_dir,
        realization
):
    """
    Run pyrecodes simulation.

    Parameters
    ----------
    main_file (str): Path to the main configuration file.
    system_config_file (str): Path to the system file.
    component_library (str): Path to the component library file.
    locality_geojson (dict): Path to the locality geojson file.
    """
    # Assume Results_det.json and Results_rlz.json are in rwhale run dir
    # This script is call in rwhale run dir
    if r2d_run_dir is None:
        run_dir = Path.cwd()
    else:
        run_dir = Path(r2d_run_dir)

    # Make a dir for RecoverySimulation
    if (run_dir / 'RecoverySimulation').exists():
        msg = 'RecoverySimulation directory already exists'
        # Remove all the files and subfolders
        for filename in os.listdir(str(run_dir / 'RecoverySimulation')):
            file_path = run_dir / 'RecoverySimulation' / filename
            try:
                # If it's a file, remove it
                if file_path.is_file() or Path(file_path).is_symlink():
                    file_path.unlink()
                # If it's a folder, remove it and its contents
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            except (OSError, shutil.Error) as e:
                msg = f"Failed to delete {file_path}. Reason: {e}"
                raise RuntimeError(msg) from e
        # raise ValueError(msg)
    else:
        (run_dir / 'RecoverySimulation').mkdir()
    os.chdir(run_dir / 'RecoverySimulation')

    # Load the system configuration and modify the damage input part
    with Path(system_config_file).open() as f:
        system_config = json.load(f)

    # Modify the file pathes in the Content part of the system configuration
    system_config = modify_system_config_conent(system_config, locality_geojson,
                                                run_dir)

    # Check the realziation value
    if realization is None:
        raise RuntimeError("Realization is not provided")
    elif type(realization) is not str:
        raise RuntimeError(f"Realization text type must be string: {type(realization)}.")

    # Sina: Main_File is optional. If not provided, one is made by the code.
    # Required for the workflow app widget run.
    if main_file is not None:
        # Modify the DamageInput part of the system configuration
        with Path(main_file).open() as f:
            main_file_dict = json.load(f)
    else:
        main_file_dict = {
            "ComponentLibrary":{
                "ComponentLibraryCreatorClassName": "JSONComponentLibraryCreator",
                "ComponentLibraryCreatorFileName": "json_component_library_creator",
                "ComponentLibraryFile": f"{component_library}"
                },
            "DamageInput": {
                "Parameters": {
                    "Filter": realization
				},
                "Type": "SpecificRealization"
            },
            "System":{
                "SystemClassName": "BuiltEnvironment",
                "SystemConfigurationFile": f"{system_config_file}",
                "SystemCreatorClassName": "ConcreteSystemCreator",
                "SystemCreatorFileName": "concrete_system_creator",
                "SystemFileName": "built_environment"
                }
            }

    damage_input = main_file_dict['DamageInput']

    if damage_input['Type'] == 'MostlikelyDamageState':
        # Create a Results_rlz.json file for the most likely damage state
        rlz = 0
        with Path(run_dir / f'Results_{rlz}.json').open() as f:
            results_rlz = json.load(f)
        with Path(run_dir/'Results_det.json').open() as f:
            results_det = json.load(f)
        for asset_type, asset_type_dict in results_rlz.items():
            for asset_subtype, asset_subtype_dict in asset_type_dict.items():
                for asset_id, asset_id_dict in asset_subtype_dict.items():
                    damage_dict = asset_id_dict['Damage']
                    for comp in damage_dict:
                        damage_dict[comp] = int(results_det[asset_type][asset_subtype]\
                        [asset_id]['R2Dres']['R2Dres_MostLikelyCriticalDamageState'])
                    if 'Loss' in asset_id_dict:
                        loss_dist = asset_id_dict['Loss']
                        for comp in loss_dist['Repair']['Cost']:
                            mean_cost_key = next(x for x in results_det[asset_type][asset_subtype]\
                                [asset_id]['R2Dres'] if x.startswith('R2Dres_mean_RepairCost'))
                            # A minmum cost of 0.1 is set to avoid division by zero
                            loss_dist['Repair']['Cost'][comp] = max(results_det[asset_type][asset_subtype]\
                            [asset_id]['R2Dres'][mean_cost_key], 0.1)
                        for comp in loss_dist['Repair']['Time']:
                            mean_time_key = next(x for x in results_det[asset_type][asset_subtype]\
                                [asset_id]['R2Dres'] if x.startswith('R2Dres_mean_RepairTime'))
                            # A minmum time of 0.1 is set to avoid division by zero
                            loss_dist['Repair']['Time'][comp] = max(results_det[asset_type][asset_subtype]\
                            [asset_id]['R2Dres'][mean_time_key], 0.1)

        rlz = 'mostlikely'
        # Create a directory for the realization
        Path(f'workdir.{rlz}').mkdir()
        os.chdir(f'workdir.{rlz}')
        rlz_run_dir = Path.cwd()

        with (run_dir /'Results_mostlikely.json').open('w') as f:
            json.dump(results_rlz, f)

        pyrecodes_damage_input = system_config['DamageInput']
        pyrecodes_damage_input['Parameters']['DamageFile'] = str(
            run_dir / f'Results_{rlz}.json')

        # Modify the file pathes in the REWETDistributionModel part of the system configuration
        system_config = modify_system_config_rewet_distribution(system_config, rewet_inp_file, rlz_run_dir)
        # Write the modified system configuration to a file
        with Path('SystemConfiguration.json').open('w') as f:
            json.dump(system_config, f)

        # Modify the main file and write to a file
        main_file_path = modify_main_file(main_file_dict, component_library, rlz_run_dir)

        # Run the pyrecodes
        run_one_realization(main_file_path, rlz, run_dir, system_config)

        # Add the recovery time to the Results_det.json file
        with Path(run_dir / 'Results_det.json').open() as f:
            results_det = json.load(f)
        with Path(rlz_run_dir / f'Results_{rlz}.json').open() as f:
            results_rlz = json.load(f)
        for asset_type, asset_type_dict in results_rlz.items():
            for asset_subtype, asset_subtype_dict in asset_type_dict.items():
                for asset_id, asset_id_dict in asset_subtype_dict.items():
                    if 'Recovery' in asset_id_dict:
                        recovery_dict = asset_id_dict['Recovery']
                        results_det[asset_type][asset_subtype][asset_id]['R2Dres'].update(
                                {'R2Dres_mean_RecoveryDuration': recovery_dict['Time']}
                            )
                    else:
                        results_det[asset_type][asset_subtype][asset_id]['R2Dres'].update(
                                {'R2Dres_mean_RecoveryDuration': np.inf}
                            )
        with Path(run_dir / 'Results_det.json').open('w') as f:
            json.dump(results_det, f)

    elif damage_input['Type'] == 'SpecificRealization' or \
            damage_input['Type'] == 'SampleFromRealizations':
        rlz_to_run = select_realizations_to_run(damage_input, run_dir)
        results_agg = create_agg_results_dict(Path(run_dir/'Results_det.json'))
        for rlz in rlz_to_run:
            print(f'Running realization {rlz}')  # noqa: T201
            # Create a directory for the realization
            rlz_run_dir = run_dir / 'RecoverySimulation'/f'workdir.{rlz}'
            rlz_run_dir.mkdir()
            os.chdir(rlz_run_dir)
            # Create a Results_rlz.json file for the specific realization
            pyrecodes_damage_input = system_config['DamageInput']
            pyrecodes_damage_input['Parameters']['DamageFile'] = str(
                run_dir / f'Results_{rlz}.json')

            # Modify the loss values in the Results_rlz.json file so that the
            # loss values are a small number if the damage is nonzero but loss is zero
            # This needs to be removed once the pyrecodes is updated to handle this
            with Path(run_dir / f'Results_{rlz}.json').open() as f:
                results_rlz = json.load(f)
            for asset_type_dict in results_rlz.values():
                for asset_subtype_dict in asset_type_dict.values():
                    for asset_id_dict in asset_subtype_dict.values():
                        damage_dict = asset_id_dict['Damage']
                        if 'Loss' in asset_id_dict:
                            loss_dist = asset_id_dict['Loss']
                            for comp in loss_dist['Repair']['Cost']:
                                # A minmum cost of 0.1 is set to avoid division by zero
                                loss_dist['Repair']['Cost'][comp] = max(loss_dist['Repair']['Cost'][comp], 0.1)
                            for comp in loss_dist['Repair']['Time']:
                                # A minmum time of 0.1 is set to avoid division by zero
                                loss_dist['Repair']['Time'][comp] = max(loss_dist['Repair']['Time'][comp], 0.1)
            with Path(run_dir / f'Results_{rlz}.json').open('w') as f:
                json.dump(results_rlz, f)

            # Modify the file pathes in the REWETDistributionModel part of the system configuration
            system_config = modify_system_config_rewet_distribution(system_config, rewet_inp_file, rlz_run_dir)
            # Write the modified system configuration to a file
            with Path('SystemConfiguration.json').open('w') as f:
                json.dump(system_config, f)

            # Modify the main file and write to a file
            main_file_path = modify_main_file(main_file_dict, component_library, rlz_run_dir)

            # Run the pyrecodes
            run_one_realization(main_file_path, rlz, run_dir, system_config)

            # Append the results to the aggregated results dictionary
            results_agg = append_to_results_agg(results_agg, Path(rlz_run_dir / f'Results_{rlz}.json'))
            print(f'Rrealization {rlz} completed')  # noqa: T201
        # Write the aggregated results to the Results_det.json file
        aggregate_results_to_det(results_agg, Path(run_dir / 'Results_det.json'))
    else:
        msg = 'Damage input type not recognized'
        raise ValueError(msg)



    # with open(system_config_file, 'r') as f:
    #     system_config = json.load(f)
    # r2d_damage_input = system_config.pop('DamageInput')
    # realizations_to_run = select_realizations_to_run(damage_input, inputRWHALE)




# def run_pyrecodes_old(rec_config, inputRWHALE, parallelType, mpiExec, numPROC):  # noqa: ARG001, C901, N803, D103
#     # Initiate directory
#     rec_ouput_dir = os.path.join(inputRWHALE['runDir'], 'Results', 'Recovery')  # noqa: PTH118
#     if not os.path.exists(rec_ouput_dir):
#         os.mkdir(rec_ouput_dir)

#     # Find the realizations to run
#     damage_input = rec_config.pop('DamageInput')
#     realizations_to_run = select_realizations_to_run(damage_input, inputRWHALE)
#     # Replace SimCenterDefault with correct path
#     cmp_lib = rec_config['ComponentLibrary']
#     if cmp_lib.startswith('SimCenterDefault'):
#         cmp_lib_name = cmp_lib.split('/')[1]
#         cmp_lib_dir = os.path.dirname(os.path.realpath(__file__))
#         cmp_lib = os.path.join(cmp_lib_dir, cmp_lib_name)
#         rec_config['ComponentLibrary'] = cmp_lib
#     # loop through each realizations. Needs to be parallelized
#     # Create the base of system configuration json
#     system_configuration = create_system_configuration(rec_config)
#     # Create the base of main json
#     main_json = dict()
#     main_json.update(
#         {
#             'ComponentLibrary': {
#                 'ComponentLibraryCreatorClass': 'JSONComponentLibraryCreator',
#                 'ComponentLibraryFile': rec_config['ComponentLibrary'],
#             }
#         }
#     )

#     # initialize a dict to accumulate recovery results stats
#     result_det_path = os.path.join(  # noqa: PTH118
#         inputRWHALE['runDir'],
#         'Results',
#         'Results_det.json',
#     )
#     with open(result_det_path, 'r') as f:  # noqa: PTH123, UP015
#         results_det = json.load(f)
#     result_agg = dict()  # noqa: C408
#     resilience_results = dict()  # noqa: C408

#     # Loop through realizations and run pyrecodes
#     numP = 1  # noqa: N806
#     procID = 0  # noqa: N806
#     doParallel = False  # noqa: N806
#     mpi_spec = importlib.util.find_spec('mpi4py')
#     found = mpi_spec is not None
#     if found and parallelType == 'parRUN':
#         import mpi4py
#         from mpi4py import MPI

#         comm = MPI.COMM_WORLD
#         numP = comm.Get_size()  # noqa: N806
#         procID = comm.Get_rank()  # noqa: N806
#         if numP < 2:  # noqa: PLR2004
#             doParallel = False  # noqa: N806
#             numP = 1  # noqa: N806
#             procID = 0  # noqa: N806
#         else:
#             doParallel = True  # noqa: N806
#     count = 0
#     needsInitiation = True  # noqa: N806
#     ind_in_rank = 0
#     for ind, rlz_ind in enumerate(realizations_to_run):  # noqa: B007
#         # Create a realization directory
#         if count % numP == procID:
#             rlz_dir = os.path.join(rec_ouput_dir, str(rlz_ind))  # noqa: PTH118
#             if not os.path.exists(rlz_dir):  # noqa: PTH110
#                 os.mkdir(rlz_dir)  # noqa: PTH102

#             # Update the system_configuration json
#             damage_rlz_file = os.path.join(  # noqa: PTH118
#                 inputRWHALE['runDir'], 'Results', f'Results_{int(rlz_ind)}.json'
#             )
#             DamageInput = {  # noqa: N806
#                 'Type': 'R2DDamageInput',
#                 'Parameters': {'DamageFile': damage_rlz_file},
#             }
#             system_configuration.update({'DamageInput': DamageInput})

#             # Write the system_configureation to a file
#             system_configuration_file = os.path.join(  # noqa: PTH118
#                 rlz_dir, 'SystemConfiguration.json'
#             )
#             with open(system_configuration_file, 'w') as f:  # noqa: PTH123
#                 ujson.dump(system_configuration, f)
#             # Update the main json
#             main_json.update(
#                 {
#                     'System': {
#                         'SystemCreatorClass': 'ConcreteSystemCreator',
#                         'SystemClass': 'BuiltEnvironmentSystem',
#                         'SystemConfigurationFile': system_configuration_file,
#                     }
#                 }
#             )

#             # Write the main json to a file
#             main_file = os.path.join(rlz_dir, 'main.json')  # noqa: PTH118
#             with open(main_file, 'w') as f:  # noqa: PTH123
#                 ujson.dump(main_json, f)

#             system = main.run(main_file)

#             system.calculate_resilience()

#             # Append the recovery time to results_rlz
#             if needsInitiation:
#                 needsInitiation = False  # noqa: N806
#                 num_of_rlz_per_rank = int(np.floor(len(realizations_to_run) / numP))
#                 if procID < len(realizations_to_run) % numP:
#                     num_of_rlz_per_rank += 1
#                 # Initialize resilience_results
#                 resilience_results_buffer = dict()  # noqa: C408
#                 resilience_calculator_id = 0
#                 resilience_results.update(
#                     {
#                         'time_steps': list(range(0, system.MAX_TIME_STEP + 1))  # noqa: PIE808
#                     }
#                 )
#                 resources_to_plot = system.resilience_calculators[
#                     resilience_calculator_id
#                 ].system_supply.keys()
#                 for resource_name in resources_to_plot:
#                     resilience_results_buffer.update(
#                         {
#                             resource_name: {
#                                 'Supply': np.zeros(
#                                     [num_of_rlz_per_rank, system.MAX_TIME_STEP + 1]
#                                 ),
#                                 'Demand': np.zeros(
#                                     [num_of_rlz_per_rank, system.MAX_TIME_STEP + 1]
#                                 ),
#                                 'Consumption': np.zeros(
#                                     [num_of_rlz_per_rank, system.MAX_TIME_STEP + 1]
#                                 ),
#                             }
#                         }
#                     )
#                 # Initialize result_agg
#                 result_agg_buffer = dict()  # noqa: C408
#                 for asset_type, item in results_det.items():
#                     asset_type_result = dict()  # noqa: C408
#                     for asset_subtype, asset_subtype_item in item.items():
#                         asset_subtype_result = dict()  # noqa: C408
#                         for aim_id, aim in asset_subtype_item.items():  # noqa: B007
#                             asset_subtype_result.update(
#                                 {
#                                     aim_id: {
#                                         'RecoveryDuration': np.zeros(
#                                             num_of_rlz_per_rank
#                                         )
#                                     }
#                                 }
#                             )
#                         asset_type_result.update(
#                             {asset_subtype: asset_subtype_result}
#                         )
#                     result_agg_buffer.update({asset_type: asset_type_result})
#                 del results_det
#             resilience_result_rlz_i = dict()  # noqa: C408
#             for resource_name in resources_to_plot:
#                 resilience_result_rlz_i.update(
#                     {
#                         'time_steps': list(range(0, system.time_step + 1)),  # noqa: PIE808
#                         resource_name: {
#                             'Supply': system.resilience_calculators[
#                                 resilience_calculator_id
#                             ].system_supply[resource_name][: system.time_step + 1],
#                             'Demand': system.resilience_calculators[
#                                 resilience_calculator_id
#                             ].system_demand[resource_name][: system.time_step + 1],
#                             'Consumption': system.resilience_calculators[
#                                 resilience_calculator_id
#                             ].system_consumption[resource_name][
#                                 : system.time_step + 1
#                             ],
#                         },
#                     }
#                 )
#                 resilience_results_buffer[resource_name]['Supply'][
#                     ind_in_rank, : system.time_step + 1
#                 ] = system.resilience_calculators[
#                     resilience_calculator_id
#                 ].system_supply[resource_name][: system.time_step + 1]
#                 resilience_results_buffer[resource_name]['Demand'][
#                     ind_in_rank, : system.time_step + 1
#                 ] = system.resilience_calculators[
#                     resilience_calculator_id
#                 ].system_demand[resource_name][: system.time_step + 1]
#                 resilience_results_buffer[resource_name]['Consumption'][
#                     ind_in_rank, : system.time_step + 1
#                 ] = system.resilience_calculators[
#                     resilience_calculator_id
#                 ].system_consumption[resource_name][: system.time_step + 1]
#             resilience_result_rlz_i_file = os.path.join(  # noqa: PTH118
#                 rlz_dir, 'ResilienceResult.json'
#             )
#             with open(resilience_result_rlz_i_file, 'w') as f:  # noqa: PTH123
#                 ujson.dump(resilience_result_rlz_i, f)
#             result_file_name = os.path.join(  # noqa: PTH118
#                 inputRWHALE['runDir'],
#                 'Results',
#                 f'Results_{rlz_ind}.json',
#             )
#             with open(result_file_name, 'r') as f:  # noqa: PTH123, UP015
#                 results = json.load(f)
#             for comp in system.components:
#                 if getattr(comp, 'r2d_comp', False) is True:
#                     recovery_duration = (
#                         getattr(comp, 'recoverd_time_step', system.MAX_TIME_STEP)
#                         - system.DISASTER_TIME_STEP
#                     )
#                     recovery_duration = max(0, recovery_duration)
#                     results[comp.asset_type][comp.asset_subtype][comp.aim_id].update(
#                         {'Recovery': {'Duration': recovery_duration}}
#                     )
#                     result_agg_buffer[comp.asset_type][comp.asset_subtype][
#                         comp.aim_id
#                     ]['RecoveryDuration'][ind_in_rank] = recovery_duration
#             with open(result_file_name, 'w') as f:  # noqa: PTH123
#                 ujson.dump(results, f)

#             ind_in_rank += 1
#         count = count + 1  # noqa: PLR6104, RUF100

#     # wait for all to finish
#     if doParallel:
#         comm.Barrier()

#     # if rank 0, gather result_agg and resilience_results, write to file
#     # note that the gathered results doesn't follow the order in realization_to_run
#     # but this order is not needed when calculating mean and std
#     if doParallel:
#         # gather results_agg
#         for asset_type, item in result_agg_buffer.items():
#             asset_type_result = dict()  # noqa: C408
#             for asset_subtype, asset_subtype_item in item.items():
#                 asset_subtype_result = dict()  # noqa: C408
#                 for aim_id, aim in asset_subtype_item.items():  # noqa: B007
#                     asset_subtype_result.update(
#                         {
#                             aim_id: {
#                                 'RecoveryDuration': comm.gather(
#                                     result_agg_buffer[asset_type][asset_subtype],
#                                     root=0,
#                                 )
#                             }
#                         }
#                     )
#                 asset_type_result.update({asset_subtype: asset_subtype_result})
#             result_agg.update({asset_type: asset_type_result})
#             # gather resilience_resutls
#         for resource_name in resources_to_plot:
#             if procID == 0:
#                 resilience_results.update(
#                     {
#                         resource_name: {
#                             'Supply': np.zeros(
#                                 [len(realizations_to_run), system.MAX_TIME_STEP + 1]
#                             ),
#                             'Demand': np.zeros(
#                                 [len(realizations_to_run), system.MAX_TIME_STEP + 1]
#                             ),
#                             'Consumption': np.zeros(
#                                 [len(realizations_to_run), system.MAX_TIME_STEP + 1]
#                             ),
#                         }
#                     }
#                 )
#             comm.gather(
#                 resilience_results_buffer[resource_name]['Supply'],
#                 resilience_results[resource_name]['Supply'],
#                 root=0,
#             )
#             comm.gather(
#                 resilience_results_buffer[resource_name]['Demand'],
#                 resilience_results[resource_name]['Demand'],
#                 root=0,
#             )
#             comm.gather(
#                 resilience_results_buffer[resource_name]['Consumption'],
#                 resilience_results[resource_name]['Consumption'],
#                 root=0,
#             )
#     else:
#         for resource_name in resources_to_plot:
#             resilience_results.update(
#                 {resource_name: resilience_results_buffer[resource_name]}
#             )
#         result_agg = result_agg_buffer

#     if procID == 0:
#         # Calculate stats of the results and add to results_det.json
#         with open(result_det_path, 'r') as f:  # noqa: PTH123, UP015
#             results_det = json.load(f)
#         for asset_type, item in result_agg.items():
#             for asset_subtype, asset_subtype_item in item.items():
#                 for aim_id, aim in asset_subtype_item.items():
#                     if (
#                         'R2Dres'  # noqa: SIM118
#                         not in results_det[asset_type][asset_subtype][aim_id].keys()
#                     ):
#                         results_det[asset_type][asset_subtype][aim_id].update(
#                             {'R2Dres': {}}
#                         )
#                     results_det[asset_type][asset_subtype][aim_id]['R2Dres'].update(
#                         {
#                             'R2Dres_mean_RecoveryDuration': aim[
#                                 'RecoveryDuration'
#                             ].mean(),
#                             'R2Dres_std_RecoveryDuration': aim[
#                                 'RecoveryDuration'
#                             ].std(),
#                         }
#                     )
#         with open(result_det_path, 'w') as f:  # noqa: PTH123
#             ujson.dump(results_det, f)
#         recovery_result_path = os.path.join(rec_ouput_dir, 'ResilienceResult.json')  # noqa: PTH118
#         for resource_name in resources_to_plot:
#             resilience_results[resource_name].update(
#                 {
#                     'R2Dres_mean_Supply': resilience_results[resource_name]['Supply']
#                     .mean(axis=0)
#                     .tolist(),
#                     'R2Dres_std_Supply': resilience_results[resource_name]['Supply']
#                     .std(axis=0)
#                     .tolist(),
#                     'R2Dres_mean_Demand': resilience_results[resource_name]['Demand']
#                     .mean(axis=0)
#                     .tolist(),
#                     'R2Dres_std_Demand': resilience_results[resource_name]['Demand']
#                     .std(axis=0)
#                     .tolist(),
#                     'R2Dres_mean_Consumption': resilience_results[resource_name][
#                         'Consumption'
#                     ]
#                     .mean(axis=0)
#                     .tolist(),
#                     'R2Dres_std_Consumption': resilience_results[resource_name][
#                         'Consumption'
#                     ]
#                     .std(axis=0)
#                     .tolist(),
#                 }
#             )
#             resilience_results[resource_name].pop('Supply')
#             resilience_results[resource_name].pop('Demand')
#             resilience_results[resource_name].pop('Consumption')

#         with open(recovery_result_path, 'w') as f:  # noqa: PTH123
#             ujson.dump(resilience_results, f)

#     # Below are for development use
#     from pyrecodes import GeoVisualizer as gvis  # noqa: N813

#     geo_visualizer = gvis.R2D_GeoVisualizer(system.components)
#     geo_visualizer.plot_component_localities()
#     from pyrecodes import Plotter

#     plotter_object = Plotter.Plotter()
#     x_axis_label = 'Time step [day]'
#     resources_to_plot = [
#         'Shelter',
#         'FunctionalHousing',
#         'ElectricPower',
#         'PotableWater',
#     ]
#     resource_units = ['[beds/day]', '[beds/day]', '[MWh/day]', '[RC/day]']
#     # define which resilience calculator to use to plot the supply/demand/consumption of the resources
#     # they are ordered as in the system configuration file
#     resilience_calculator_id = 0
#     for i, resource_name in enumerate(resources_to_plot):
#         y_axis_label = f'{resource_name} {resource_units[i]} | {system.resilience_calculators[resilience_calculator_id].scope}'
#         axis_object = plotter_object.setup_lor_plot_fig(x_axis_label, y_axis_label)
#         time_range = system.time_step + 1
#         time_steps_before_event = 10
#         plotter_object.plot_single_resource(
#             list(range(-time_steps_before_event, time_range)),
#             resilience_results[resource_name]['R2Dres_mean_Supply'][:time_range],
#             resilience_results[resource_name]['R2Dres_mean_Demand'][:time_range],
#             resilience_results[resource_name]['R2Dres_mean_Consumption'][
#                 :time_range
#             ],
#             axis_object,
#             warmup=time_steps_before_event,
#         )
#     print()  # noqa: T201


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





if __name__ == '__main__':
    # Defining the command line arguments

    workflowArgParser = argparse.ArgumentParser(  # noqa: N816
        'Run Pyrecodes from the NHERI SimCenter rWHALE workflow for a set of assets.',
        allow_abbrev=False,
    )

    workflowArgParser.add_argument(
        '--input', required=True,
        help='R2D JSON input file.',
    )

    workflowArgParser.add_argument(
        '--mainFile', help='Pyrecodes main file', required=False
    )

    workflowArgParser.add_argument(
        '--systemConfigFile', help='Pyrecodes system configuration file', required=True
    )

    workflowArgParser.add_argument(
        '--componentLibraryFile', help='Pyrecodes component library file', required=True
    )

    workflowArgParser.add_argument(
        '--localityGeojsonFile',
        default=None,
        help='Geojson defining the locality of the assets',
    )

    workflowArgParser.add_argument(
        '--INPFile',
        default=None,
        help='Geojson defining the locality of the assets',
    )

    workflowArgParser.add_argument(
        '--r2dRunDir',
        default=None,
        help='R2D run directory containing the results',
    )
    # Below are for future parallelization
    workflowArgParser.add_argument(
        '--parallelType',
        default='seqRUN',
        help='How parallel runs: options seqRUN, parSETUP, parRUN',
    )
    workflowArgParser.add_argument(
        '--mpiexec',
        default='mpiexec',
        help='How mpi runs, e.g. ibrun, mpirun, mpiexec',
    )
    workflowArgParser.add_argument(
        '--numP',
        default='8',
        help='If parallel, how many jobs to start with mpiexec option',
    )

    # Parsing the command line arguments
    wfArgs = workflowArgParser.parse_args()  # noqa: N816

    # Calling the main workflow method and passing the parsed arguments
    # numPROC = int(wfArgs.numP)

    json_input_file_path = Path(wfArgs.input)
    json_input_file_path =json_input_file_path.resolve()

    with open(json_input_file_path, "rt") as f:
        json_input = json.load(f)

    dl_assets = json_input["Applications"]["DL"]

    realization = None

    realization_asset_types = ""
    for asset_types in dl_assets:
        cur_application = dl_assets[asset_types]
        if "Application" not in cur_application:
            continue
        if "ApplicationData" not in cur_application:
            continue
        cur_application_data = cur_application.get("ApplicationData")
        cur_realization = cur_application_data.get("Realizations")

        if cur_realization is None:
            continue
        elif type(cur_realization) is not int:
            try:
                cur_realization = int(cur_realization)
            except:
                try:
                    cur_realization = int(float(cur_realization))
                except:
                    raise RuntimeError(f"The realziation for {asset_types} is not an integer: {cur_realization}")

        realization_asset_types += asset_types + " "
        if realization is None:
            realization = cur_realization
        elif cur_realization < realization:
            realization = cur_realization

        print(f"The smallest realziation accross {realization_asset_types}is {realization}.")

        if realization < 0:
            raise ValueError(f"Realization should be more than 0: realziation = {realization}")

        realization_text = ""
        for i in range(realization):
            realization_text += f"{i+1},"

    run_pyrecodes(
        main_file=wfArgs.mainFile,
        system_config_file=wfArgs.systemConfigFile,
        component_library=wfArgs.componentLibraryFile,
        locality_geojson=wfArgs.localityGeojsonFile,
        rewet_inp_file=wfArgs.INPFile,
        r2d_run_dir=wfArgs.r2dRunDir,
        realization=realization_text
    )
