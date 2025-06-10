import ujson as json
import os
import sys
import psutil
import importlib
import subprocess
# Add the folder containing the script to the system path
script_path = os.path.dirname(os.path.realpath(__file__))
opensha_path = os.path.join(script_path, 'lib', 'OpenSHA-1.5.2.jar')
sys.path.append(script_path)
import geopandas as gpd
import pandas as pd
import numpy as np
import time
import copy
import argparse
import GlobalVariable
if GlobalVariable.JVM_started is False:
    GlobalVariable.JVM_started = True
    if importlib.util.find_spec('jpype') is None:
        subprocess.check_call(  # noqa: S603
            [sys.executable, '-m', 'pip', 'install', 'JPype1']
        )  # noqa: RUF100, S603
    import jpype

    # from jpype import imports
    import jpype.imports
    # from jpype.types import *  # noqa: F403

    memory_total = psutil.virtual_memory().total / (1024.0**3)
    memory_request = int(memory_total * 0.25)
    jpype.addClassPath(opensha_path)
    jpype.startJVM(f'-Xmx{memory_request}G', convertStrings=False)
print('JVM started')
from ComputeIntensityMeasure import compute_im  # noqa: F403
# print('debug 3')
from GMSimulators import simulate_ground_motion
# print('debug 1')
from LoadRupFile import load_earthquake_rup_scenario  # noqa: F403
# print('debug 1')
from SelectGroundMotion import select_ground_motion
from SelectGroundMotion import output_all_ground_motion_info
# print('debug 1')
from ComputeIntensityMeasure import export_im
# print('debug 1')

# # @profile
def run_scenario_i(sce_idx, hazard_info_orig, all_rups, procID = 0):

    
    hazard_info = copy.deepcopy(hazard_info_orig)
    # Select the range of scenarios to simulate
    hazard_info['Scenario']['Generator']['RuptureFilter'] = str(sce_idx+1) # The index in the r2d ground motion simulation starts from 1
    
    # Below are initializing HazardSimulationEQ
    # directory (back compatibility here)
    work_dir = hazard_info['Directory']
    output_dir = os.path.join(work_dir, f'Output_{sce_idx}')
    try:
        os.mkdir(f'{output_dir}')  # noqa: PTH102
    except:  # noqa: E722
        print('HazardSimulation: output folder already exists.')  # noqa: T201
    # Read Site .csv
    site_file = hazard_info['Site']['siteFile']
    try:
        stations = pd.read_csv(site_file).to_dict(orient='records')
        print(f'Proc {procID}: HazardSimulation: stations loaded.')  # noqa: T201
    except:  # noqa: E722
        print(f'Proc {procID}: HazardSimulation: please check the station file {site_file}')  # noqa: T201
        exit()  # noqa: PLR1722
    print(f'Proc {procID}: HazardSimulation: loading scenarios.')  # noqa: T201
    scenario_info = hazard_info['Scenario']
    scenarios = load_earthquake_rup_scenario(scenario_info, all_rups)   # noqa: F405
    print(f'Proc {procID}: HazardSimulation: scenarios loaded.')  # noqa: T201

    # current, peak = tracemalloc.get_traced_memory()
    # print(f"Current memory usage: {current / 10**6} MB")
    # print(f"Peak memory usage: {peak / 10**6} MB")

    selected_scen_ids = sorted(list(scenarios.keys()))  # noqa: C414
    # Computing intensity measures
    print(f'Proc {procID}: HazardSimulation: computing intensity measures.')  # noqa: T201
    # Computing uncorrelated Sa
    event_info = hazard_info['Event']
    # When vector IM is used. The PGA/SA needs to be computed before PGV
    im_info = event_info['IntensityMeasure']
    im_raw_path, im_list = compute_im(  # noqa: F405
                scenarios,
                stations,
                scenario_info,
                event_info.get('GMPE', None),
                event_info['IntensityMeasure'],
                scenario_info['Generator'],
                output_dir,
                mth_flag=False,
            )
    
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"Current memory usage: {current / 10**6} MB")
    # print(f"Peak memory usage: {peak / 10**6} MB")

    num_gm_per_site = event_info['NumberPerSite']
    # Computing correlated IMs
    ln_im_mr, mag_maf = simulate_ground_motion(
        stations,
        im_raw_path,
        im_list,
        scenarios,
        num_gm_per_site,
        event_info['CorrelationModel'],
        event_info['IntensityMeasure'],
        selected_scen_ids,
    )

    # Selecting ground motion records
    if scenario_info['Type'] == 'Earthquake':
        # Selecting records
        data_source = event_info.get('Database', 0)
        if data_source:
            print('HazardSimulation: selecting ground motion records.')  # noqa: T201
            sf_max = event_info['ScalingFactor']['Maximum']
            sf_min = event_info['ScalingFactor']['Minimum']
            start_time = time.time()
            gm_id, gm_file = select_ground_motion(  # noqa: F405
                im_list,
                ln_im_mr,
                data_source,
                sf_max,
                sf_min,
                output_dir,
                'EventGrid.csv',
                stations,
                selected_scen_ids,
            )
            print(  # noqa: T201
                f'HazardSimulation: ground motion records selected  ({time.time() - start_time} s).'
            )
            # print(gm_id)
            gm_id = [int(i) for i in np.unique(gm_id)]
            gm_file = [i for i in np.unique(gm_file)]  # noqa: C416
            runtag = output_all_ground_motion_info(  # noqa: F405
                gm_id, gm_file, output_dir, 'RecordsList.csv'
            )
            if runtag:
                print('HazardSimulation: the ground motion list saved.')  # noqa: T201
            else:
                sys.exit(
                    'HazardSimulation: warning - issues with saving the ground motion list.'
                )
        else:
            print('HazardSimulation: ground motion selection is not requested.')  # noqa: T201

    gf_im_list = []
    if 'GroundFailure' in hazard_info['Event'].keys():  # noqa: SIM118
        ground_failure_info = hazard_info['Event']['GroundFailure']
        if 'Liquefaction' in ground_failure_info.keys():  # noqa: SIM118
            import liquefaction

            trigging_info = ground_failure_info['Liquefaction']['Triggering']
            trigging_model = getattr(liquefaction, trigging_info['Model'])(
                trigging_info['Parameters'], stations
            )
            trigging_output_keys = ['liq_prob', 'liq_susc']
            additional_output_required_keys = (
                liquefaction.find_additional_output_req(
                    ground_failure_info['Liquefaction'], 'Triggering'
                )
            )
            ln_im_mr, mag_maf, im_list, addtional_output = trigging_model.run(
                ln_im_mr,
                mag_maf,
                im_list,
                trigging_output_keys,
                additional_output_required_keys,
            )
            del trigging_model
            gf_im_list += trigging_info['Output']
            if 'LateralSpreading' in ground_failure_info['Liquefaction'].keys():  # noqa: SIM118
                lat_spread_info = ground_failure_info['Liquefaction'][
                    'LateralSpreading'
                ]
                lat_spread_para = lat_spread_info['Parameters']
                if (
                    lat_spread_info['Model'] == 'Hazus2020Lateral'
                ) and addtional_output.get('dist_to_water', None) is not None:
                    lat_spread_para.update(
                        'DistWater', addtional_output['dist_to_water']
                    )
                lat_spread_model = getattr(liquefaction, lat_spread_info['Model'])(
                    stations, lat_spread_para
                )
                ln_im_mr, mag_maf, im_list = lat_spread_model.run(
                    ln_im_mr, mag_maf, im_list
                )
                gf_im_list += lat_spread_info['Output']
            if 'Settlement' in ground_failure_info['Liquefaction'].keys():  # noqa: SIM118
                settlement_info = ground_failure_info['Liquefaction']['Settlement']
                settlement_model = getattr(liquefaction, settlement_info['Model'])()
                ln_im_mr, mag_maf, im_list = settlement_model.run(
                    ln_im_mr, mag_maf, im_list
                )
                gf_im_list += settlement_info['Output']
        if 'Landslide' in ground_failure_info.keys():  # noqa: SIM118
            import landslide  # noqa: PLC0415, RUF100

            if 'Landslide' in ground_failure_info['Landslide'].keys():  # noqa: SIM118
                lsld_info = ground_failure_info['Landslide']['Landslide']
                lsld_model = getattr(landslide, lsld_info['Model'])(
                    lsld_info['Parameters'], stations
                )
                ln_im_mr, mag_maf, im_list = lsld_model.run(
                    ln_im_mr, mag_maf, im_list
                )
                gf_im_list += lsld_info['Output']
    if event_info['SaveIM'] and ln_im_mr:
        print('HazardSimulation: saving simulated intensity measures.')  # noqa: T201
        _ = export_im(  # noqa: F405
            stations,
            im_list,
            ln_im_mr,
            mag_maf,
            output_dir,
            'SiteIM.json',
            1,
            gf_im_list,
            selected_scen_ids,
        )
        print('HazardSimulation: simulated intensity measures saved.')  # noqa: T201
    else:
        print('HazardSimulation: IM is not required to saved or no IM is found.')  # noqa: T201



if __name__ == '__main__':
    ## input:
    parser = argparse.ArgumentParser(description="Pass arguments through command line")
    parser.add_argument('--input_dir')
    parser.add_argument('--sce_idx')
    parser.add_argument('--procID')
    args = parser.parse_args()  # Parse the arguments

    input_dir = args.input_dir
    sce_idx = int(args.sce_idx)
    procID = int(args.procID)

    hazard_config_file = os.path.join(input_dir, 'EQHazardConfiguration.json')
    with open(hazard_config_file, 'r') as f:
        hazard_info = json.load(f)

    rup_file = hazard_info['Scenario']['sourceFile']
    all_rups = gpd.read_file(rup_file)

    run_scenario_i(sce_idx, hazard_info, all_rups, procID)


