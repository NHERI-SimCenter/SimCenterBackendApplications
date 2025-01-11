#  # noqa: INP001, D100
# Copyright (c) 2022 Leland Stanford Junior University
# Copyright (c) 2022 The Regents of the University of California
#
# This file is part of the SimCenter Backend Applications
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# this file. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Kuanshi Zhong
#

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil

R2D = True


def hazard_job(hazard_info):  # noqa: C901, D103, PLR0915
    from CreateScenario import load_ruptures_openquake
    from GMSimulators import simulate_ground_motion

    try:
        # oq_flag = hazard_info['Scenario']['EqRupture']['Type'] in ['oqSourceXML']
        oq_flag = 'OpenQuake' in hazard_info['Scenario']['EqRupture']['Type']
    except:  # noqa: E722
        oq_flag = False
    # Read Site .csv
    site_file = hazard_info['Site']['siteFile']
    try:
        stations = pd.read_csv(site_file).to_dict(orient='records')
        print('HazardSimulation: stations loaded.')  # noqa: T201
    except:  # noqa: E722
        print(f'HazardSimulation: please check the station file {site_file}')  # noqa: T201
        exit()  # noqa: PLR1722
    # print(stations)

    # Scenarios
    print('HazardSimulation: loading scenarios.')  # noqa: T201
    scenario_info = hazard_info['Scenario']
    if scenario_info['Type'] == 'Earthquake':
        # KZ-10/31/2022: checking user-provided scenarios
        if scenario_info['EqRupture']['Type'] == 'oqSourceXML':
            # The rup file is not enough for oq erf, so the rupture needs to be recalculated
            rupFile = scenario_info['sourceFile']  # noqa: N806
            scenarios = load_ruptures_openquake(
                scenario_info, stations, work_dir, site_file, rupFile
            )
        else:
            rupFile = scenario_info['sourceFile']  # noqa: N806
            scenarios = load_earthquake_rupFile(scenario_info, rupFile)  # noqa: F405
    else:
        # TODO: extending this to other hazards  # noqa: TD002
        print('HazardSimulation: currently only supports EQ and Wind simulations.')  # noqa: T201
    # print(scenarios)
    print('HazardSimulation: scenarios loaded.')  # noqa: T201
    selected_scen_ids = sorted(list(scenarios.keys()))  # noqa: C414
    # Computing intensity measures
    print('HazardSimulation: computing intensity measures.')  # noqa: T201
    if scenario_info['Type'] == 'Earthquake':
        # Computing uncorrelated Sa
        event_info = hazard_info['Event']
        # When vector IM is used. The PGA/SA needs to be computed before PGV
        im_info = event_info['IntensityMeasure']
        # To make the SA format consistent with R2D requirement
        if im_info['Type'] == 'Vector' and 'SA' in im_info.keys():  # noqa: SIM118
            periods = im_info['SA']['Periods']
            periods = [float(i) for i in periods]
            im_info['SA']['Periods'] = periods
        if im_info['Type'] == 'SA':
            periods = im_info['Periods']
            periods = [float(i) for i in periods]
            im_info['Periods'] = periods
        # Reorder the IMs
        if im_info['Type'] == 'Vector':
            new_im_info = {}
            desired_order = ['PGA', 'SA', 'PGV', 'DS575H', 'DS595H']
            for key in desired_order:
                if key in im_info:
                    new_im_info.update({key: im_info[key]})
            new_im_info.update({"Type": "Vector"})
            event_info['IntensityMeasure'] = new_im_info

        if (
            opensha_flag
            or hazard_info['Scenario']['EqRupture']['Type'] == 'oqSourceXML'
        ):
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
            # update the im_info
            event_info['IntensityMeasure'] = im_info
        elif oq_flag:
            # Preparing config ini for OpenQuake
            filePath_ini, oq_ver_loaded, event_info = openquake_config(  # noqa: N806, F405
                hazard_info['Site'],
                scenario_info,
                event_info,
                hazard_info['Directory'],
            )
            if not filePath_ini:
                # Error in ini file
                sys.exit(
                    'HazardSimulation: errors in preparing the OpenQuake configuration file.'
                )
            if scenario_info['EqRupture']['Type'] in [
                'OpenQuakeClassicalPSHA',
                'OpenQuakeUserConfig',
                'OpenQuakeClassicalPSHA-User',
            ]:
                # Calling openquake to run classical PSHA
                # oq_version = scenario_info['EqRupture'].get('OQVersion',default_oq_version)
                oq_run_flag = oq_run_classical_psha(  # noqa: F405
                    filePath_ini,
                    exports='csv',
                    oq_version=oq_ver_loaded,
                    dir_info=dir_info,  # noqa: F405
                )
                if oq_run_flag:
                    err_msg = 'HazardSimulation: OpenQuake Classical PSHA failed.'
                    if not new_db_sqlite3:
                        err_msg = (
                            err_msg
                            + ' Please see if there is leaked python threads in background still occupying {}.'.format(
                                os.path.expanduser('~/oqdata/db.sqlite3')  # noqa: PTH111
                            )
                        )
                    print(err_msg)  # noqa: T201
                    sys.exit(err_msg)
                else:
                    print('HazardSimulation: OpenQuake Classical PSHA completed.')  # noqa: T201
                if scenario_info['EqRupture'].get('UHS', False):
                    ln_im_mr, mag_maf, im_list = oq_read_uhs_classical_psha(  # noqa: F405
                        scenario_info,
                        event_info,
                        dir_info,  # noqa: F405
                    )
                else:
                    ln_im_mr = []
                    mag_maf = []
                    im_list = []
                # stn_new = stations['Stations']

            elif scenario_info['EqRupture']['Type'] == 'oqSourceXML':
                # Creating and conducting OpenQuake calculations
                oq_calc = OpenQuakeHazardCalc(  # noqa: F405
                    filePath_ini,
                    event_info,
                    oq_ver_loaded,
                    dir_info=hazard_info['Directory'],
                )
                oq_calc.run_calc()
                im_raw = [oq_calc.eval_calc()]  # noqa: F841
                # stn_new = stations['Stations']
                print('HazardSimulation: OpenQuake Scenario calculation completed.')  # noqa: T201

            else:
                sys.exit(
                    'HazardSimulation: OpenQuakeClassicalPSHA, OpenQuakeUserConfig and OpenQuakeScenario are supported.'
                )

        # KZ-08/23/22: adding method to do hazard occurrence model
        # im_type = 'SA'
        # period = 1.0
        # im_level = 0.2*np.ones((len(im_raw[0].get('GroundMotions')),1))
        hc_curves = None
        occurrence_sampling = scenario_info['Generator']['method'] == 'Subsampling'
        if occurrence_sampling:
            # read all configurations
            occurrence_info = scenario_info['Generator']['Parameters']
            reweight_only = occurrence_info.get('ReweightOnly', False)
            # KZ-10/31/22: adding a flag for whether to re-sample ground motion maps or just monte-carlo
            sampling_gmms = occurrence_info.get('SamplingGMMs', True)
            occ_dict = configure_hazard_occurrence(  # noqa: F405
                input_dir,
                output_dir,
                im_raw_path,
                im_list,
                scenarios,
                hzo_config=occurrence_info,
                site_config=stations,
            )
            model_type = occ_dict.get('Model')
            num_target_eqs = occ_dict.get('NumTargetEQs')
            num_target_gmms = occ_dict.get('NumTargetGMMs')
            num_per_eq_avg = int(np.ceil(num_target_gmms / num_target_eqs))
            return_periods = occ_dict.get('ReturnPeriods')
            im_type = occ_dict.get('IntensityMeasure')
            period = occ_dict.get('Period')
            hc_curves = occ_dict.get('HazardCurves')
            # get im exceedance probabilities
            im_exceedance_prob = get_im_exceedance_probility(  # noqa: F405
                im_raw_path, im_list, im_type, period, hc_curves, selected_scen_ids
            )
            # sample the earthquake scenario occurrence
            # if reweight_only:
            #     occurrence_rate_origin = [scenarios[i].get('MeanAnnualRate') for i in range(len(scenarios))]
            # else:
            #     occurrence_rate_origin = None
            occurrence_rate_origin = [
                scenarios[i].get('MeanAnnualRate') for i in selected_scen_ids
            ]
            occurrence_model = sample_earthquake_occurrence(  # noqa: F405
                model_type,
                num_target_eqs,
                return_periods,
                im_exceedance_prob,
                reweight_only,
                occurrence_rate_origin,
                occurrence_info,
            )
            # print(occurrence_model)
            P, Z = occurrence_model.get_selected_earthquake()  # noqa: N806
            # now update the im_raw with selected eqs with Z > 0
            id_selected_eqs = []
            for i in range(len(Z)):
                if P[i] > 0:
                    id_selected_eqs.append(selected_scen_ids[i])  # noqa: PERF401
            selected_scen_ids = id_selected_eqs
            num_per_eq_avg = int(np.ceil(num_target_gmms / len(selected_scen_ids)))
            # compute error from optimization residual
            error = occurrence_model.get_error_vector()
            # export sampled earthquakes
            _ = export_sampled_earthquakes(  # noqa: F405
                error, selected_scen_ids, scenarios, P, output_dir
            )

        # Updating station information
        # stations['Stations'] = stn_new
        print('HazardSimulation: uncorrelated response spectra computed.')  # noqa: T201
        # print(im_raw)
        # KZ-08/23/22: adding method to do hazard occurrence model
        if occurrence_sampling and sampling_gmms:
            num_gm_per_site = num_per_eq_avg
        else:
            num_gm_per_site = event_info['NumberPerSite']
        print('num_gm_per_site = ', num_gm_per_site)  # noqa: T201
        if scenario_info['EqRupture']['Type'] not in [
            'OpenQuakeClassicalPSHA',
            'OpenQuakeUserConfig',
            'OpenQuakeClassicalPSHA-User',
        ]:
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
            print('HazardSimulation: correlated response spectra computed.')  # noqa: T201
        # KZ-08/23/22: adding method to do hazard occurrence model
        if occurrence_sampling and sampling_gmms:
            # get im exceedance probabilities for individual ground motions
            # print('im_list = ',im_list)
            im_exceedance_prob_gmm, occur_rate_origin = (
                get_im_exceedance_probability_gm(  # noqa: F405
                    np.exp(ln_im_mr),
                    im_list,
                    im_type,
                    period,
                    hc_curves,
                    np.array(mag_maf)[:, 1],
                )
            )
            # sample the earthquake scenario occurrence
            # if reweight_only:
            #     occurrence_rate_origin = [scenarios[i].get('MeanAnnualRate') for i in range(len(scenarios))]
            # else:
            #     occurrence_rate_origin = None
            occurrence_model_gmm = sample_earthquake_occurrence(  # noqa: F405
                model_type,
                num_target_gmms,
                return_periods,
                im_exceedance_prob_gmm,
                reweight_only,
                occur_rate_origin,
                occurrence_info,
            )
            # print(occurrence_model)
            P_gmm, Z_gmm = occurrence_model_gmm.get_selected_earthquake()  # noqa: N806
            # now update the im_raw with selected eqs with Z > 0
            id_selected_gmms = []
            for i in range(len(Z_gmm)):
                if P_gmm[i] > 0:
                    id_selected_gmms.append(i)  # noqa: PERF401
            id_selected_scens = np.array(
                [
                    selected_scen_ids[int(x / num_gm_per_site)]
                    for x in id_selected_gmms
                ]
            )
            id_selected_simus = np.array(
                [x % num_gm_per_site for x in id_selected_gmms]
            )
            # export sampled earthquakes
            occurrence_model_gmm.export_sampled_gmms(
                id_selected_gmms, id_selected_scens, P_gmm, output_dir
            )

            selected_scen_ids_step2 = sorted(list(set(id_selected_scens)))  # noqa: C414
            sampled_ln_im_mr = [None] * len(selected_scen_ids_step2)
            sampled_mag_maf = [None] * len(selected_scen_ids_step2)

            for i, selected_scen in enumerate(selected_scen_ids_step2):
                scen_ind = selected_scen_ids.index(selected_scen)
                selected_simus_in_scen_i = sorted(  # noqa: C414
                    list(set(id_selected_simus[id_selected_scens == selected_scen]))
                )
                sampled_ln_im_mr[i] = ln_im_mr[scen_ind][
                    :, :, selected_simus_in_scen_i
                ]
                sampled_mag_maf[i] = mag_maf[scen_ind]
            ln_im_mr = sampled_ln_im_mr
            mag_maf = sampled_mag_maf

        # if event_info['SaveIM'] and ln_im_mr:
        #     print('HazardSimulation: saving simulated intensity measures.')
        #     _ = export_im(stations, im_list,
        #                   ln_im_mr, mag_maf, output_dir, 'SiteIM.json', 1)
        #     print('HazardSimulation: simulated intensity measures saved.')
        # else:
        #     print('HazardSimulation: IM is not required to saved or no IM is found.')
        # print(np.exp(ln_im_mr[0][0, :, 1]))
        # print(np.exp(ln_im_mr[0][1, :, 1]))
    else:
        # TODO: extending this to other hazards  # noqa: TD002
        print('HazardSimulation currently only supports earthquake simulations.')  # noqa: T201
    print('HazardSimulation: intensity measures computed.')  # noqa: T201
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
            # Downloading records
            user_name = event_info.get('UserName', None)
            user_password = event_info.get('UserPassword', None)
            if (user_name is not None) and (user_password is not None) and (not R2D):
                print('HazardSimulation: downloading ground motion records.')  # noqa: T201
                raw_dir = download_ground_motion(  # noqa: F405
                    gm_id, user_name, user_password, output_dir
                )
                if raw_dir:
                    print('HazardSimulation: ground motion records downloaded.')  # noqa: T201
                    # Parsing records
                    print('HazardSimulation: parsing records.')  # noqa: T201
                    record_dir = parse_record(  # noqa: F405, F841
                        gm_file,
                        raw_dir,
                        output_dir,
                        event_info['Database'],
                        event_info['OutputFormat'],
                    )
                    print('HazardSimulation: records parsed.')  # noqa: T201
                else:
                    print('HazardSimulation: No records to be parsed.')  # noqa: T201
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

    # If hazard downsampling algorithm is used. Save the errors.


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--hazard_config')
    args = parser.parse_args()

    # read the hazard configuration file
    with open(args.hazard_config) as f:  # noqa: PTH123
        hazard_info = json.load(f)

    # directory (back compatibility here)
    work_dir = hazard_info['Directory']
    input_dir = os.path.join(work_dir, 'Input')  # noqa: PTH118
    output_dir = os.path.join(work_dir, 'Output')  # noqa: PTH118
    try:
        os.mkdir(f'{output_dir}')  # noqa: PTH102
    except:  # noqa: E722
        print('HazardSimulation: output folder already exists.')  # noqa: T201

    # parse job type for set up environment and constants
    try:
        opensha_flag = hazard_info['Scenario']['EqRupture']['Type'] in [
            'PointSource',
            'ERF',
        ]
    except:  # noqa: E722
        opensha_flag = False
    try:
        oq_flag = hazard_info['Scenario']['EqRupture']['Type'] == 'oqSourceXML'
    except:  # noqa: E722
        oq_flag = False

    # dependencies
    if R2D:
        packages = ['tqdm', 'psutil', 'PuLP', 'requests']
    else:
        packages = ['selenium', 'tqdm', 'psutil', 'PuLP', 'requests']
    for p in packages:
        if importlib.util.find_spec(p) is None:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', p])  # noqa: S603

    # set up environment
    import socket

    if 'stampede2' not in socket.gethostname():
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
            from jpype.types import *  # noqa: F403

            memory_total = psutil.virtual_memory().total / (1024.0**3)
            memory_request = int(memory_total * 0.75)
            jpype.addClassPath('./lib/OpenSHA-1.5.2.jar')
            jpype.startJVM(f'-Xmx{memory_request}G', convertStrings=False)
    if oq_flag:
        # clear up old db.sqlite3 if any
        if os.path.isfile(os.path.expanduser('~/oqdata/db.sqlite3')):  # noqa: PTH111, PTH113
            new_db_sqlite3 = True
            try:
                os.remove(os.path.expanduser('~/oqdata/db.sqlite3'))  # noqa: PTH107, PTH111
            except:  # noqa: E722
                new_db_sqlite3 = False
        # data dir
        os.environ['OQ_DATADIR'] = os.path.join(  # noqa: PTH118
            os.path.abspath(output_dir),  # noqa: PTH100
            'oqdata',
        )
        print('HazardSimulation: local OQ_DATADIR = ' + os.environ.get('OQ_DATADIR'))  # noqa: T201
        if os.path.exists(os.environ.get('OQ_DATADIR')):  # noqa: PTH110
            print(  # noqa: T201
                'HazardSimulation: local OQ folder already exists, overwriting it now...'
            )
            shutil.rmtree(os.environ.get('OQ_DATADIR'))
        os.makedirs(f"{os.environ.get('OQ_DATADIR')}")  # noqa: PTH103

    # import modules
    from ComputeIntensityMeasure import *  # noqa: F403
    from CreateScenario import *  # noqa: F403
    from CreateStation import *  # noqa: F403

    # KZ-08/23/22: adding hazard occurrence model
    from HazardOccurrence import *  # noqa: F403
    from SelectGroundMotion import *  # noqa: F403

    if oq_flag:
        # import FetchOpenQuake
        from FetchOpenQuake import *  # noqa: F403

    # untar site databases
    # site_database = ['global_vs30_4km.tar.gz','global_zTR_4km.tar.gz','thompson_vs30_4km.tar.gz']
    # print('HazardSimulation: Extracting site databases.')
    # cwd = os.path.dirname(os.path.realpath(__file__))
    # for cur_database in site_database:
    #     subprocess.run(["tar","-xvzf",cwd+"/database/site/"+cur_database,"-C",cwd+"/database/site/"])

    # Initial process list
    import psutil

    proc_list_init = [
        p.info
        for p in psutil.process_iter(attrs=['pid', 'name'])
        if 'python' in p.info['name']
    ]

    hazard_job(hazard_info)

    # Closing the current process
    sys.exit(0)
