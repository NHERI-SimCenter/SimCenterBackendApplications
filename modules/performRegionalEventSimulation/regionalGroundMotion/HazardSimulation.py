# -*- coding: utf-8 -*-
#
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

import os, shutil, psutil
import sys
import subprocess
import argparse, posixpath, json
import numpy as np
import pandas as pd
import time
import importlib

R2D = True

def site_job(hazard_info):

    # Sites and stations
    print('HazardSimulation: creating stations.')
    site_info = hazard_info['Site']
    if site_info['Type'] == 'From_CSV':
        input_file = os.path.join(input_dir,site_info['input_file'])
        output_file = site_info.get('output_file',False)
        if output_file:
            output_file = os.path.join(output_dir, output_file)
        min_ID = site_info['min_ID']
        max_ID = site_info['max_ID']
        # forward compatibility
        if minID:
            min_ID = minID
            site_info['min_ID'] = minID
        if maxID:
            max_ID = maxID
            site_info['max_ID'] = maxID
        # Creating stations from the csv input file
        z1_tag = 0
        z25_tag = 0
        # Vs30
        if 'Global Vs30' in site_info['Vs30']['Type']:
            vs30_tag = 1
        elif 'Thompson' in site_info['Vs30']['Type']:
            vs30_tag = 2
        elif 'National Crustal Model' in site_info['Vs30']['Type']:
            vs30_tag = 3
        else:
            vs30_tag = 0
        # Bedrock depth
        zTR_tag = 0
        if 'SoilGrid250' in site_info['BedrockDepth']['Type']:
            zTR_tag = 0
        elif 'National Crustal Model' in site_info['BedrockDepth']['Type']:
            zTR_tag = 1
        # soil model if any
        if site_info.get('SoilModel', None) is not None:
            soil_model_type = site_info['SoilModel'].get('Type', 'EI')
        else:
            soil_model_type = None
        # user soil model function file path
        soil_user_fun = None
        if soil_model_type == 'User':
            soil_user_fun = site_info['SoilModel'].get('Parameters', None)
            if soil_user_fun is not None:
                soil_user_fun = os.path.join(input_dir, soil_user_fun)
        # Creating stations from the csv input file
        stations = create_stations(input_file, output_file, min_ID, max_ID, vs30_tag, z1_tag, z25_tag, zTR_tag=zTR_tag, 
                                   soil_flag=True, soil_model_type=soil_model_type, soil_user_fun=soil_user_fun)
    if stations:
        print('HazardSimulation: site data are fetched and saved in {}.'.format(output_file))
    else:
        print('HazardSimulation: please check the "Input" directory in the configuration json file.')
        exit()


def hazard_job(hazard_info):

    # Sites and stations
    print('HazardSimulation: creating stations.')
    site_info = hazard_info['Site']
    if site_info['Type'] == 'From_CSV':
        input_file = os.path.join(input_dir,site_info['input_file'])
        output_file = site_info.get('output_file',False)
        if output_file:
            output_file = os.path.join(input_dir, output_file)
        min_ID = site_info['min_ID']
        max_ID = site_info['max_ID']
        # forward compatibility
        if minID:
            min_ID = minID
            site_info['min_ID'] = minID
        if maxID:
            max_ID = maxID
            site_info['max_ID'] = maxID
        # Creating stations from the csv input file
        z1_tag = 0
        z25_tag = 0
        if 'OpenQuake' in hazard_info['Scenario']['EqRupture']['Type']:
            z1_tag = 1
            z25_tag = 1
        if 'Global Vs30' in site_info['Vs30']['Type']:
            vs30_tag = 1
        elif 'Thompson' in site_info['Vs30']['Type']:
            vs30_tag = 2
        elif 'NCM' in site_info['Vs30']['Type']:
            vs30_tag = 3
        else:
            vs30_tag = 0
        # Creating stations from the csv input file
        stations = create_stations(input_file, output_file, min_ID, max_ID, vs30_tag, z1_tag, z25_tag)
    if stations:
        print('HazardSimulation: stations created.')
    else:
        print('HazardSimulation: please check the "Input" directory in the configuration json file.')
        exit()
    #print(stations)

    # Scenarios
    print('HazardSimulation: creating scenarios.')
    scenario_info = hazard_info['Scenario']
    if scenario_info['Type'] == 'Earthquake':
        # Creating earthquake scenarios
        if scenario_info['EqRupture']['Type'] in ['PointSource', 'ERF']:
            scenarios = create_earthquake_scenarios(scenario_info, stations, dir_info)
    elif scenario_info['Type'] == 'Wind':
        # Creating wind scenarios
        scenarios = create_wind_scenarios(scenario_info, stations, input_dir)
    else:
        # TODO: extending this to other hazards
        print('HazardSimulation: currently only supports EQ and Wind simulations.')
    #print(scenarios)
    print('HazardSimulation: scenarios created.')

    # Computing intensity measures
    print('HazardSimulation: computing intensity measures.')
    if scenario_info['Type'] == 'Earthquake':
        # Computing uncorrelated Sa
        event_info = hazard_info['Event']
        if opensha_flag:
            im_raw, im_info = compute_im(scenarios, stations['Stations'],
                                event_info['GMPE'], event_info['IntensityMeasure'],
                                scenario_info.get('EqRupture').get('HazardOccurrence',None), output_dir)
            # update the im_info
            event_info['IntensityMeasure'] = im_info
        elif oq_flag:
            # Preparing config ini for OpenQuake
            filePath_ini, oq_ver_loaded, event_info = openquake_config(site_info, scenario_info, event_info, dir_info)
            if not filePath_ini:
                # Error in ini file
                sys.exit('HazardSimulation: errors in preparing the OpenQuake configuration file.') 
            if scenario_info['EqRupture']['Type'] in ['OpenQuakeClassicalPSHA','OpenQuakeUserConfig', 'OpenQuakeClassicalPSHA-User']:
                # Calling openquake to run classical PSHA
                #oq_version = scenario_info['EqRupture'].get('OQVersion',default_oq_version)
                oq_run_flag = oq_run_classical_psha(filePath_ini, exports='csv', oq_version=oq_ver_loaded, dir_info=dir_info)
                if oq_run_flag:
                    err_msg = 'HazardSimulation: OpenQuake Classical PSHA failed.'
                    if not new_db_sqlite3:
                        err_msg = err_msg + ' Please see if there is leaked python threads in background still occupying {}.'.format(os.path.expanduser('~/oqdata/db.sqlite3'))
                    print(err_msg)
                    sys.exit(err_msg)
                else:
                    print('HazardSimulation: OpenQuake Classical PSHA completed.')
                if scenario_info['EqRupture'].get('UHS', False):
                    ln_im_mr, mag_maf, im_list = oq_read_uhs_classical_psha(scenario_info, event_info, dir_info)
                else:
                    ln_im_mr = []
                    mag_maf = []
                    im_list = []
                #stn_new = stations['Stations']

            elif scenario_info['EqRupture']['Type'] == 'OpenQuakeScenario':
                # Creating and conducting OpenQuake calculations
                oq_calc = OpenQuakeHazardCalc(filePath_ini, event_info, oq_ver_loaded, dir_info=dir_info)
                oq_calc.run_calc()
                im_raw = [oq_calc.eval_calc()]
                #stn_new = stations['Stations']
                print('HazardSimulation: OpenQuake Scenario calculation completed.')

            else:                
                sys.exit('HazardSimulation: OpenQuakeClassicalPSHA, OpenQuakeUserConfig and OpenQuakeScenario are supported.')
            
        # KZ-08/23/22: adding method to do hazard occurrence model
        #im_type = 'SA'
        #period = 1.0
        #im_level = 0.2*np.ones((len(im_raw[0].get('GroundMotions')),1))
        occurrence_sampling = scenario_info.get('EqRupture').get('OccurrenceSampling',False)
        if occurrence_sampling:
            # read all configurations
            occurrence_info = scenario_info.get('EqRupture').get('HazardOccurrence')
            occ_dict = configure_hazard_occurrence(input_dir, output_dir, hzo_config=occurrence_info, site_config=stations['Stations'])
            model_type = occ_dict.get('Model')
            num_target_eqs = occ_dict.get('NumTargetEQs')
            num_target_gmms = occ_dict.get('NumTargetGMMs')
            num_per_eq_avg = int(np.ceil(num_target_gmms/num_target_eqs))
            return_periods = occ_dict.get('ReturnPeriods')
            im_type = occ_dict.get('IntensityMeasure')
            period = occ_dict.get('Period')
            hc_curves = occ_dict.get('HazardCurves')
            # get im exceedance probabilities
            im_exceedance_prob = get_im_exceedance_probility(im_raw, im_type, period, hc_curves)
            # sample the earthquake scenario occurrence
            occurrence_model = sample_earthquake_occurrence(model_type,num_target_eqs,return_periods,im_exceedance_prob)
            #print(occurrence_model)
            P, Z = occurrence_model.get_selected_earthquake()
            # now update the im_raw with selected eqs with Z > 0
            id_selected_eqs = []
            for i in range(len(Z)):
                if P[i] > 0:
                    id_selected_eqs.append(i)
            im_raw_sampled = [im_raw[i] for i in id_selected_eqs]
            im_raw = im_raw_sampled
            num_per_eq_avg = int(np.ceil(num_target_gmms/len(id_selected_eqs)))
            # export sampled earthquakes
            _ = export_sampled_earthquakes(id_selected_eqs, scenarios, P, output_dir)
        
        # Updating station information
        #stations['Stations'] = stn_new
        print('HazardSimulation: uncorrelated response spectra computed.')
        #print(im_raw)
        # KZ-08/23/22: adding method to do hazard occurrence model
        if occurrence_sampling:
            num_gm_per_site = num_per_eq_avg
        else:
            num_gm_per_site = event_info['NumberPerSite']
        print('num_gm_per_site = ',num_gm_per_site)
        if not scenario_info['EqRupture']['Type'] in ['OpenQuakeClassicalPSHA','OpenQuakeUserConfig','OpenQuakeClassicalPSHA-User']:
            # Computing correlated IMs
            ln_im_mr, mag_maf, im_list = simulate_ground_motion(stations['Stations'], im_raw,
                                                                num_gm_per_site,
                                                                event_info['CorrelationModel'],
                                                                event_info['IntensityMeasure'])
            print('HazardSimulation: correlated response spectra computed.')
        # KZ-08/23/22: adding method to do hazard occurrence model
        if occurrence_sampling:
            # get im exceedance probabilities for individual ground motions
            #print('im_list = ',im_list)
            im_exceedance_prob_gmm = get_im_exceedance_probability_gm(np.exp(ln_im_mr), im_list, im_type, period, hc_curves)
            # sample the earthquake scenario occurrence
            occurrence_model_gmm = sample_earthquake_occurrence(model_type,num_target_gmms,return_periods,im_exceedance_prob_gmm)
            #print(occurrence_model)
            P_gmm, Z_gmm = occurrence_model_gmm.get_selected_earthquake()
            # now update the im_raw with selected eqs with Z > 0
            id_selected_gmms = []
            for i in range(len(Z_gmm)):
                if P_gmm[i] > 0:
                    id_selected_gmms.append(i)
            id_selected_scens = [int(x/num_gm_per_site) for x in id_selected_gmms]
            id_selected_simus = [x%num_gm_per_site for x in id_selected_gmms]
            # export sampled earthquakes
            _ = export_sampled_gmms(id_selected_gmms, id_selected_scens, P_gmm, output_dir)
            num_site = ln_im_mr[0].shape[0]
            num_im = ln_im_mr[0].shape[1]
            sampled_im_gmms = np.zeros((num_site,num_im,len(id_selected_gmms)))
            count = 0
            for i in range(len(id_selected_gmms)):
                sampled_im_gmms[:,:,count]=ln_im_mr[id_selected_scens[i]][:,:,id_selected_simus[i]].tolist()
                count = count+1
            ln_im_mr_sampled = [sampled_im_gmms]
            ln_im_mr = ln_im_mr_sampled
            mag_maf = [[0,0,0,0]]
            
        if event_info['SaveIM'] and ln_im_mr:
            print('HazardSimulation: saving simulated intensity measures.')
            _ = export_im(stations['Stations'], im_list,
                          ln_im_mr, mag_maf, output_dir, 'SiteIM.json', 1)
            print('HazardSimulation: simulated intensity measures saved.')
        else:
            print('HazardSimulation: IM is not required to saved or no IM is found.')
        #print(np.exp(ln_im_mr[0][0, :, 1]))
        #print(np.exp(ln_im_mr[0][1, :, 1]))
    else:
        # TODO: extending this to other hazards
        print('HazardSimulation currently only supports earthquake simulations.')
    print('HazardSimulation: intensity measures computed.')
    # Selecting ground motion records
    if scenario_info['Type'] == 'Earthquake':
        # Selecting records
        data_source = event_info.get('Database',0)
        if data_source:
            print('HazardSimulation: selecting ground motion records.')
            sf_max = event_info['ScalingFactor']['Maximum']
            sf_min = event_info['ScalingFactor']['Minimum']
            start_time = time.time()
            gm_id, gm_file = select_ground_motion(im_list, ln_im_mr, data_source,
                                                  sf_max, sf_min, output_dir, 'EventGrid.csv',
                                                  stations['Stations'])
            print('HazardSimulation: ground motion records selected  ({0} s).'.format(time.time() - start_time))
            #print(gm_id)
            gm_id = [int(i) for i in np.unique(gm_id)]
            gm_file = [i for i in np.unique(gm_file)]
            runtag = output_all_ground_motion_info(gm_id, gm_file, output_dir, 'RecordsList.csv')
            if runtag:
                print('HazardSimulation: the ground motion list saved.')
            else:
                sys.exit('HazardSimulation: warning - issues with saving the ground motion list.')
            # Downloading records
            user_name = event_info.get('UserName', None)
            user_password = event_info.get('UserPassword', None)
            if (user_name is not None) and (user_password is not None) and (not R2D):
                print('HazardSimulation: downloading ground motion records.')
                raw_dir = download_ground_motion(gm_id, user_name,
                                                 user_password, output_dir)
                if raw_dir:
                    print('HazardSimulation: ground motion records downloaded.')
                    # Parsing records
                    print('HazardSimulation: parsing records.')
                    record_dir = parse_record(gm_file, raw_dir, output_dir,
                                              event_info['Database'],
                                              event_info['OutputFormat'])
                    print('HazardSimulation: records parsed.')
                else:
                    print('HazardSimulation: No records to be parsed.')
        else:
            print('HazardSimulation: ground motion selection is not requested.')


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--hazard_config')
    parser.add_argument('--filter', default=None)
    parser.add_argument('-d', '--referenceDir', default=None)
    parser.add_argument('-w', '--workDir', default=None)
    parser.add_argument('--hcid', default=None)
    parser.add_argument('-j', '--job_type', default='Hazard')
    args = parser.parse_args()

    # read the hazard configuration file
    with open(args.hazard_config) as f:
        hazard_info = json.load(f)

    # directory (back compatibility here)
    dir_info = hazard_info['Directory']
    work_dir = dir_info['Work']
    input_dir = dir_info['Input']
    output_dir = dir_info['Output']
    if args.referenceDir:
        input_dir = args.referenceDir
        dir_info['Input'] = input_dir
    if args.workDir:
        output_dir = args.workDir
        dir_info['Output'] = output_dir
        dir_info['Work'] = output_dir
    try:
        os.mkdir(f"{output_dir}")
    except:
        print('HazardSimulation: output folder already exists.')

    # site filter (if explicitly defined)
    minID = None
    maxID = None
    if args.filter:
        tmp = [int(x) for x in args.filter.split('-')]
        if len(tmp) == 1:
            minID = tmp[0]
            maxID = minID
        else:
            [minID, maxID] = tmp

    # parse job type for set up environment and constants
    try:
        opensha_flag = hazard_info['Scenario']['EqRupture']['Type'] in ['PointSource', 'ERF']
    except:
        opensha_flag = False
    try:
        oq_flag = 'OpenQuake' in hazard_info['Scenario']['EqRupture']['Type']
    except:
        oq_flag = False

    # dependencies
    if R2D:
        packages = ['tqdm', 'psutil', 'PuLP', 'requests']
    else:
        packages = ['selenium', 'tqdm', 'psutil', 'PuLP', 'requests']
    for p in packages:
        if importlib.util.find_spec(p) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

    # set up environment
    import socket
    if 'stampede2' not in socket.gethostname():
        if importlib.util.find_spec('jpype') is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "JPype1"])
        import jpype
        from jpype import imports
        from jpype.types import *
        memory_total = psutil.virtual_memory().total/(1024.**3)
        memory_request = int(memory_total*0.75)
        jpype.addClassPath('./lib/OpenSHA-1.5.2.jar')
        jpype.startJVM("-Xmx{}G".format(memory_request), convertStrings=False)
    if oq_flag:
        # clear up old db.sqlite3 if any
        if os.path.isfile(os.path.expanduser('~/oqdata/db.sqlite3')):
            new_db_sqlite3 = True
            try:
                os.remove(os.path.expanduser('~/oqdata/db.sqlite3'))
            except:
                new_db_sqlite3 = False
        # data dir
        os.environ['OQ_DATADIR'] = os.path.join(os.path.abspath(output_dir), 'oqdata')
        print('HazardSimulation: local OQ_DATADIR = '+os.environ.get('OQ_DATADIR'))
        if os.path.exists(os.environ.get('OQ_DATADIR')):
            print('HazardSimulation: local OQ folder already exists, overwiting it now...')
            shutil.rmtree(os.environ.get('OQ_DATADIR'))
        os.makedirs(f"{os.environ.get('OQ_DATADIR')}")

    # import modules
    from CreateStation import *
    from CreateScenario import *
    from ComputeIntensityMeasure import *
    from SelectGroundMotion import *
    # KZ-08/23/22: adding hazard occurrence model
    from HazardOccurrence import *
    if oq_flag:
        # import FetchOpenQuake
        from FetchOpenQuake import *

    # untar site databases
    site_database = ['global_vs30_4km.tar.gz','global_zTR_4km.tar.gz','thompson_vs30_4km.tar.gz']
    print('HazardSimulation: Extracting site databases.')
    cwd = os.path.dirname(os.path.realpath(__file__))
    for cur_database in site_database:
        subprocess.run(["tar","-xvzf",cwd+"/database/site/"+cur_database,"-C",cwd+"/database/site/"])

    # Initial process list
    import psutil
    proc_list_init = [p.info for p in psutil.process_iter(attrs=['pid', 'name']) if 'python' in p.info['name']]

    # run the job
    if args.job_type == 'Hazard':
        hazard_job(hazard_info)
    elif args.job_type == 'Site':
        site_job(hazard_info)
    else:
        print('HazardSimulation: --job_type = Hazard or Site (please check).')

    # Closing the current process
    sys.exit(0)