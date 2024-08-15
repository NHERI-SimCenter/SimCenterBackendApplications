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
# Jinyan Zhao

import argparse
import importlib
import json
import os
import subprocess
import sys
import tarfile

import psutil

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--hazard_config')
    args = parser.parse_args()

    # read the hazard configuration file
    with open(args.hazard_config) as f:
        hazard_info = json.load(f)

    # directory (back compatibility here)
    work_dir = hazard_info['Directory']
    input_dir = os.path.join(work_dir, 'Input')
    output_dir = os.path.join(work_dir, 'Output')
    try:
        os.mkdir(f'{output_dir}')
    except:
        print('HazardSimulation: output folder already exists.')

    # parse job type for set up environment and constants
    try:
        opensha_flag = hazard_info['Scenario']['EqRupture']['Type'] in [
            'PointSource',
            'ERF',
        ]
    except:
        opensha_flag = False
    try:
        oq_flag = (
            'OpenQuake' in hazard_info['Scenario']['EqRupture']['Type']
            or 'oqSourceXML' in hazard_info['Scenario']['EqRupture']['Type']
        )
    except:
        oq_flag = False

    # dependencies
    packages = ['tqdm', 'psutil', 'pulp', 'requests']
    for p in packages:
        if importlib.util.find_spec(p) is None:
            # print(f"""The Python package {p} is required but not found.
            #        Please install it by running
            #       "{sys.executable} -m pip install -q {p}"
            #        in your terminal or command prompt""")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', p])

    # set up environment
    import socket

    if 'stampede2' not in socket.gethostname():
        if importlib.util.find_spec('jpype') is None:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'JPype1'])
        import jpype
        from jpype.types import *

        memory_total = psutil.virtual_memory().total / (1024.0**3)
        memory_request = int(memory_total * 0.75)
        jpype.addClassPath('./lib/OpenSHA-1.5.2.jar')
        jpype.startJVM(f'-Xmx{memory_request}G', convertStrings=False)
    from CreateScenario import (
        create_earthquake_scenarios,
        create_wind_scenarios,
        load_earthquake_scenarios,
    )
    from CreateStation import create_stations
    # if oq_flag:
    #     # clear up old db.sqlite3 if any
    #     if os.path.isfile(os.path.expanduser('~/oqdata/db.sqlite3')):
    #         new_db_sqlite3 = True
    #         try:
    #             os.remove(os.path.expanduser('~/oqdata/db.sqlite3'))
    #         except:
    #             new_db_sqlite3 = False
    #     # data dir
    #     os.environ['OQ_DATADIR'] = os.path.join(os.path.abspath(output_dir), 'oqdata')
    #     print('HazardSimulation: local OQ_DATADIR = '+os.environ.get('OQ_DATADIR'))
    #     if os.path.exists(os.environ.get('OQ_DATADIR')):
    #         print('HazardSimulation: local OQ folder already exists, overwriting it now...')
    #         shutil.rmtree(os.environ.get('OQ_DATADIR'))
    #     os.makedirs(f"{os.environ.get('OQ_DATADIR')}")

    if oq_flag:
        # import FetchOpenQuake
        from FetchOpenQuake import *

    # untar site databases
    site_database = [
        'global_vs30_4km.tar.gz',
        'global_zTR_4km.tar.gz',
        'thompson_vs30_4km.tar.gz',
    ]
    print('HazardSimulation: Extracting site databases.')
    cwd = os.path.dirname(os.path.realpath(__file__))
    for cur_database in site_database:
        # subprocess.run(["tar","-xvzf",cwd+"/database/site/"+cur_database,"-C",cwd+"/database/site/"])
        tar = tarfile.open(cwd + '/database/site/' + cur_database, 'r:gz')
        tar.extractall(cwd + '/database/site/')
        tar.close()

    # # Initial process list
    # import psutil
    # proc_list_init = [p.info for p in psutil.process_iter(attrs=['pid', 'name']) if 'python' in p.info['name']]

    # Sites and stations
    print('HazardSimulation: creating stations.')
    site_info = hazard_info['Site']
    z1_tag = 0
    z25_tag = 0
    if 'OpenQuake' in hazard_info['Scenario']['EqRupture']['Type']:
        z1_tag = 1
        z25_tag = 1
    if opensha_flag:
        z1_tag = 2  # interpolate from openSHA default database
        z25_tag = 2  # interpolate from openSHA default database
        # openSHA database: https://github.com/opensha/opensha/blob/16aaf6892fe2a31b5e497270429b8d899098361a/src/main/java/org/opensha/commons/data/siteData/OrderedSiteDataProviderList.java
    site_info['Z1pt0'].update({'z1_tag': z1_tag})
    site_info['Z2pt5'].update({'z25_tag': z25_tag})
    if site_info['Type'] == 'From_CSV':
        input_file = os.path.join(input_dir, site_info['input_file'])
        output_file = site_info.get('output_file', False)
        if output_file:
            output_file = os.path.join(input_dir, output_file)
        filter = site_info['filter']
        # Creating stations from the csv input file
        stations = create_stations(
            input_file,
            output_file,
            filter,
            site_info['Vs30'],
            site_info['Z1pt0'],
            site_info['Z2pt5'],
        )
    else:
        print("""Only From_CSV site_info['Type'] is supported now""")
    if stations:
        print('ScenarioForecast: stations created.')
    else:
        print(
            'HazardSimulation: please check the "Input" directory in the configuration json file.'
        )
        exit()

    # Scenarios
    print('HazardSimulation: creating scenarios.')
    scenario_info = hazard_info['Scenario']
    if scenario_info['Type'] == 'Earthquake':
        # KZ-10/31/2022: checking user-provided scenarios
        user_scenarios = scenario_info.get('EqRupture').get(
            'UserScenarioFile', False
        )
        if user_scenarios:
            load_earthquake_scenarios(scenario_info, stations, input_dir)
        # Creating earthquake scenarios
        elif scenario_info['EqRupture']['Type'] in [
            'PointSource',
            'ERF',
            'oqSourceXML',
        ]:
            create_earthquake_scenarios(
                scenario_info, stations, work_dir, hazard_info['Site']['output_file']
            )
    elif scenario_info['Type'] == 'Wind':
        # Creating wind scenarios
        create_wind_scenarios(scenario_info, stations, input_dir)
    else:
        # TODO: extending this to other hazards
        print('HazardSimulation: currently only supports EQ and Wind simulations.')
    # print(scenarios)
    print('HazardSimulation: scenarios created.')

    # Closing the current process
    sys.exit(0)
