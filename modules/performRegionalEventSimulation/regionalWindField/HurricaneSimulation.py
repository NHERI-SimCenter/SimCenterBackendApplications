# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Leland Stanford Junior University
# Copyright (c) 2021 The Regents of the University of California
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
import argparse, posixpath, json
import numpy as np
import pandas as pd
from CreateStation import *
from CreateScenario import *
from ComputeIntensityMeasure import *

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
        print('HurricaneSimulation: output folder already exists.')

    # Sites and stations
    print('HurricaneSimulation: creating stations.')
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
        print('HurricaneSimulation: stations created.')
    else:
        print('HurricaneSimulation: please check the "Input" directory in the configuration json file.')
        exit()

    # Scenarios
    print('HurricaneSimulation: creating scenarios.')
    scenario_info = hazard_info['Scenario']
    if scenario_info['Type'] == 'Wind':
        # Creating wind scenarios
        event_info = hazard_info['Event']
        scenarios = create_wind_scenarios(scenario_info, event_info, stations, input_dir)
    else:
        print('HurricaneSimulation: currently only supports wind simulations.')
    print('HurricaneSimulation: scenarios created.')

    # Computing intensity measures
    print('HurricaneSimulation: computing intensity measures.')
    if scenario_info['Type'] == 'Wind':
        if scenario_info['Generator'] == 'Simulation':
            if scenario_info['ModelType'] == 'LinearAnalyticalPy':
                # simulating storm
                storm_simu = simulate_storm(scenarios, event_info, 'LinearAnalytical')
            elif scenario_info['ModelType'] == 'LinearAnalytical':
                # simulation storm (c++ binary)
                storm_simu = simulate_storm_cpp(site_info, scenario_info, event_info, 'LinearAnalytical', dir_info)
            else:
                print('HurricaneSimulation: currently supporting LinearAnalytical model type.')
            # converting peak wind speed
            pws = convert_wind_speed(event_info, storm_simu)
            # saving results
            export_pws(stations, pws, output_dir, filename = 'EventGrid.csv')
        else:
            print('HurricaneSimulation: currently only supporting wind simulations.')
    else:
        print('HurricaneSimulation currently only supports earthquake and wind simulations.')
    print('HurricaneSimulation: intensity measures computed.')
    
