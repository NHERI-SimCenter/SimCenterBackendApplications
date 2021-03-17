# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
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

import os
import sys
import subprocess
R2D = True
if R2D:
    packages = ['JPype1', 'tqdm']
else:
    packages = ['JPype1', 'selenium', 'tqdm']
for p in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", p])
import argparse, posixpath, json
import numpy as np
import pandas as pd
import time
import jpype
from jpype import imports
from jpype.types import *
jpype.addClassPath('./lib/OpenSHA-1.5.2.jar')
jpype.startJVM("-Xmx8G", convertStrings=False)

from CreateStation import *
from CreateScenario import *
from ComputeIntensityMeasure import *
from SelectGroundMotion import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hazard_config')
    args = parser.parse_args()
    with open(args.hazard_config) as f:
        hazard_info = json.load(f)

    # Directory
    dir_info = hazard_info['Directory']
    work_dir = dir_info['Work']
    input_dir = dir_info['Input']
    output_dir = dir_info['Output']
    try:
        os.mkdir(f"{output_dir}")
    except:
        print('HazardSimulation: output folder already exists.')

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
        # Creating stations from the csv input file
        stations = create_stations(input_file, output_file, min_ID, max_ID)
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
        scenarios = create_earthquake_scenarios(scenario_info, stations)
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
        psa_raw, stn_new = compute_spectra(scenarios, stations['Stations'],
                                           event_info['GMPE'],
                                           event_info['IntensityMeasure'])
        # Updating station information
        stations['Stations'] = stn_new
        print('HazardSimulation: uncorrelated response spectra computed.')
        #print(psa_raw)
        # Computing log mean Sa
        ln_psa_mr, mag_maf = simulate_ground_motion(stations['Stations'], psa_raw,
                                                    event_info['NumberPerSite'],
                                                    event_info['CorrelationModel'],
                                                    event_info['IntensityMeasure'])
        print('HazardSimulation: correlated response spectra computed.')
        if event_info['SaveIM']:
            print('HazardSimulation: saving simulated intensity measures.')
            _ = export_im(stations['Stations'], event_info['IntensityMeasure']['Periods'],
                          ln_psa_mr, mag_maf, output_dir, 'SiteIM.json')
            print('HazardSimulation: simulated intensity measures saved.')
        #print(np.exp(ln_psa_mr[0][0, :, 1]))
        #print(np.exp(ln_psa_mr[0][1, :, 1]))
    elif scenario_info['Type'] == 'Wind':
        if scenario_info['Generator'] == 'Simulation':
            storm_dir = simulate_storm(scenario_info['AppDir'], input_dir, output_dir)
        else:
            print('HazardSimulation: currently supporting Wind-Simulation')
    else:
        # TODO: extending this to other hazards
        print('HazardSimulation currently only supports earthquake simulations.')
    print('HazardSimulation: intensity measures computed.')
    # Selecting ground motion records
    if scenario_info['Type'] == 'Earthquake':
        # Selecting records
        target_T = event_info['IntensityMeasure']['Periods']
        if event_info['IntensityMeasure']['Type'] =='PGA':
            # PGA only
            target_T = [0.0]
        data_source = event_info.get('Database',0)
        if data_source:
            print('HazardSimulation: selecting ground motion records.')
            sf_max = event_info['ScalingFactor']['Maximum']
            sf_min = event_info['ScalingFactor']['Minimum']
            start_time = time.time()
            gm_id, gm_file = select_ground_motion(target_T, ln_psa_mr, data_source,
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
                print('HazardSimulation: warning - issues with saving the ground motion list.')
            print(gm_id)
            print(gm_file)
            # Downloading records
            user_name = event_info.get('UserName', None)
            user_password = event_info.get('UserPassword', None)
            if (user_name is not None) and (user_password is not None) and (not RDT):
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
