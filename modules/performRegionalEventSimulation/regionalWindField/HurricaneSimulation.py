#  # noqa: INP001, D100
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

import argparse
import json
import logging
import os
import sys

from ComputeIntensityMeasure import *  # noqa: F403
from CreateScenario import *  # noqa: F403
from CreateStation import *  # noqa: F403

if __name__ == '__main__':
    logger = logging.getLogger()
    handlerStream = logging.StreamHandler(sys.stdout)  # noqa: N816
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handlerStream.setFormatter(formatter)
    logger.addHandler(handlerStream)

    parser = argparse.ArgumentParser()
    parser.add_argument('--hazard_config')
    args = parser.parse_args()
    with open(args.hazard_config) as f:  # noqa: PTH123
        hazard_info = json.load(f)

    # Directory
    dir_info = hazard_info['Directory']
    work_dir = dir_info['Work']
    input_dir = dir_info['Input']
    output_dir = dir_info['Output']
    try:
        os.mkdir(f'{output_dir}')  # noqa: PTH102
    except:  # noqa: E722
        print('HurricaneSimulation: output folder already exists.')  # noqa: T201

    # Sites and stations
    print('HurricaneSimulation: creating stations.')  # noqa: T201
    site_info = hazard_info['Site']
    if site_info['Type'] == 'From_CSV':
        input_file = os.path.join(input_dir, site_info['input_file'])  # noqa: PTH118
        output_file = site_info.get('output_file', False)
        if output_file:
            output_file = os.path.join(output_dir, output_file)  # noqa: PTH118
        min_ID = site_info['min_ID']  # noqa: N816
        max_ID = site_info['max_ID']  # noqa: N816
        # Creating stations from the csv input file
        stations = create_stations(input_file, output_file, min_ID, max_ID)  # noqa: F405
    if stations:
        print('HurricaneSimulation: stations created.')  # noqa: T201
    else:
        print(  # noqa: T201
            'HurricaneSimulation: please check the "Input" directory in the configuration json file.'
        )
        exit()  # noqa: PLR1722

    # Scenarios
    print('HurricaneSimulation: creating scenarios.')  # noqa: T201
    scenario_info = hazard_info['Scenario']
    if scenario_info['Type'] == 'Wind':
        # Creating wind scenarios
        event_info = hazard_info['Event']
        scenarios = create_wind_scenarios(  # noqa: F405
            scenario_info, event_info, stations, input_dir
        )
    else:
        print('HurricaneSimulation: currently only supports wind simulations.')  # noqa: T201
    print('HurricaneSimulation: scenarios created.')  # noqa: T201

    # Computing intensity measures
    print('HurricaneSimulation: computing intensity measures.')  # noqa: T201
    if scenario_info['Type'] == 'Wind':
        if 'Simulation' in scenario_info['Generator']:
            if scenario_info['ModelType'] == 'LinearAnalyticalPy':
                # simulating storm
                storm_simu = simulate_storm(  # noqa: F405
                    scenarios, event_info, 'LinearAnalytical'
                )
            elif scenario_info['ModelType'] == 'LinearAnalytical':
                # simulation storm (c++ binary)
                storm_simu = simulate_storm_cpp(  # noqa: F405
                    site_info,
                    scenario_info,
                    scenarios,
                    event_info,
                    'LinearAnalytical',
                    dir_info,
                )
            else:
                print(  # noqa: T201
                    'HurricaneSimulation: currently supporting LinearAnalytical model type.'
                )
            # converting peak wind speed
            pws = convert_wind_speed(event_info, storm_simu)  # noqa: F405
            # saving results
            export_pws(stations, pws, output_dir, filename='EventGrid.csv')  # noqa: F405
        else:
            print('HurricaneSimulation: currently only supporting wind simulations.')  # noqa: T201
    else:
        print(  # noqa: T201
            'HurricaneSimulation currently only supports earthquake and wind simulations.'
        )
    print('HurricaneSimulation: intensity measures computed.')  # noqa: T201
