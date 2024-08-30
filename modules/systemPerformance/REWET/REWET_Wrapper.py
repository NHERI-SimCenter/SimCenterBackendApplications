#  # noqa: INP001, D100
# Copyright (c) 2024 The Regents of the University of California
# Copyright (c) 2024 Leland Stanford Junior University
#
# This file is part of whale.
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
# whale. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Sina Naeimi
# Jinyan Zhao

import argparse
import importlib
import json
import os
import random
import string
import sys
from pathlib import Path

import damage_convertor
import pandas as pd
import preprocessorIO
from shapely import geometry

# TODO (SINA): please resolve this one later
this_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()  # noqa: PTH100, PTH120

# TODO (SINA): Import REWET intead and check these
sys.path.insert(0, str(this_dir / 'REWET'))
from initial import Starter  # noqa: E402
from Result_Project import Project_Result  # noqa: E402


def createScnearioList(run_directory, scn_number):  # noqa: N802, D103
    damage_input_dir = os.path.join(  # noqa: PTH118
        run_directory, 'Results', 'WaterDistributionNetwork', 'damage_input'
    )

    if not os.path.exists(damage_input_dir):  # noqa: PTH110
        os.makedirs(damage_input_dir)  # noqa: PTH103

    # REWET_input_data["damage_input_dir"] = damage_input_dir

    prefix = chooseARandomPreefix(damage_input_dir)

    scenario_name = f'{prefix}_scn_{scn_number}'
    pipe_file_name = f'{prefix}_pipe_{scn_number}'
    node_file_name = f'{prefix}_node_{scn_number}'
    pump_file_name = f'{prefix}_pump_{scn_number}'
    tank_file_name = f'{prefix}_tank_{scn_number}'

    scenario = {
        'Scenario Name': scenario_name,
        'Pipe Damage': pipe_file_name,
        'Nodal Damage': node_file_name,
        'Pump Damage': pump_file_name,
        'Tank Damage': tank_file_name,
        'Probability': 1,
    }

    return scenario, prefix


def chooseARandomPreefix(damage_input_dir):  # noqa: N802
    """Choses a random prefix for sceranio and pipe, node, pump and tank damage
    file. The is important to find and unused prefix so if this script is being
    ran in parallel, then files are not being overwritten.

    Parameters
    ----------
    damage_input_dir : path
        The path to the damage input files directory.

    Returns
    -------
    random_prefix : str
        The Chosen random prefix string.

    """  # noqa: D205
    number_of_prefix = 4
    dir_list = os.listdir(damage_input_dir)

    prefix_dir_list = [dir_name for dir_name in dir_list if dir_name.find('_') > 0]
    prefix_list = [dir_name.split('_')[0] for dir_name in prefix_dir_list]

    random_prefix = random.choices(string.ascii_letters, k=number_of_prefix)
    s = ''
    for letter in random_prefix:
        s = s + letter
    random_prefix = s

    while random_prefix in prefix_list:
        random_prefix = random.choices(string.ascii_letters, k=number_of_prefix)
        s = ''
        for letter in random_prefix:
            s = s + letter
        random_prefix = s

    return random_prefix

    # def setSettingsData(rwhale_data, REWET_input_data):
    """
    Sets the settings (future project file) for REWET. REWET input data
    dictionary is both used as a source and destination for settinsg data. The
    data is stored in REWET_input_data object with key='settings'. The value
    for 'settinsg' is an object of REWET's 'Settings' class.

    Parameters
    ----------
    rwhale_data : json dict
        rwhale input file.
    REWET_input_data : dict
        REWET input data.

    Returns
    -------
    None.

    """  # noqa: RET503, W291


def getDLFileName(run_dir, dl_file_path, scn_number):  # noqa: N802
    """If dl_file_path is not given, the path is acquired from rwhale input data.

    Parameters
    ----------
    REWET_input_data : TYPE
        DESCRIPTION.
    damage_file_pa : TYPE
        DESCRIPTION.
    scn_number : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if dl_file_path == None:  # noqa: E711
        file_name = f'WaterDistributionNetwork_{scn_number}.json'
        run_dir = run_dir  # noqa: PLW0127
        file_dir = os.path.join(run_dir, 'Results', 'WaterDistributionNetwork')  # noqa: PTH118
        file_path = os.path.join(file_dir, file_name)  # noqa: PTH118
    else:
        file_path = dl_file_path
        file_dir = Path(dl_file_path).parent

    return file_path, file_dir


def setSettingsData(input_json, REWET_input_data):  # noqa: ARG001, N802, N803, D103
    policy_file_name = rwhale_input_data['SystemPerformance'][
        'WaterDistributionNetwork'
    ]['Policy Definition']
    policy_file_path = rwhale_input_data['SystemPerformance'][
        'WaterDistributionNetwork'
    ]['Policy DefinitionPath']

    policy_config_file = os.path.join(Path(policy_file_path), Path(policy_file_name))  # noqa: PTH118

    REWET_input_data['settings']['RUN_TIME'] = rwhale_input_data[
        'SystemPerformance'
    ]['WaterDistributionNetwork']['simulationTime']
    REWET_input_data['settings']['simulation_time_step'] = rwhale_input_data[
        'SystemPerformance'
    ]['WaterDistributionNetwork']['simulationTimeStep']

    REWET_input_data['settings']['last_sequence_termination'] = rwhale_input_data[
        'SystemPerformance'
    ]['WaterDistributionNetwork']['last_sequence_termination']
    REWET_input_data['settings']['node_demand_temination'] = rwhale_input_data[
        'SystemPerformance'
    ]['WaterDistributionNetwork']['node_demand_temination']
    REWET_input_data['settings']['node_demand_termination_time'] = rwhale_input_data[
        'SystemPerformance'
    ]['WaterDistributionNetwork']['node_demand_termination_time']
    REWET_input_data['settings']['node_demand_termination_ratio'] = (
        rwhale_input_data[
            'SystemPerformance'
        ]['WaterDistributionNetwork']['node_demand_termination_ratio']
    )
    REWET_input_data['settings']['solver'] = rwhale_input_data['SystemPerformance'][
        'WaterDistributionNetwork'
    ]['Solver']
    REWET_input_data['settings']['Restoration_on'] = rwhale_input_data[
        'SystemPerformance'
    ]['WaterDistributionNetwork']['Restoration_on']
    REWET_input_data['settings']['minimum_job_time'] = rwhale_input_data[
        'SystemPerformance'
    ]['WaterDistributionNetwork']['minimum_job_time']
    REWET_input_data['settings']['Restortion_config_file'] = (
        policy_config_file  # TODO: SINA unmark it  # noqa: TD002
    )

    p = rwhale_input_data['SystemPerformance']['WaterDistributionNetwork'][
        'pipe_damage_model'
    ]
    REWET_input_data['settings']['pipe_damage_model'] = {}
    for mat_data in p:
        REWET_input_data['settings']['pipe_damage_model'][mat_data[0]] = {
            'alpha': mat_data[1],
            'beta': mat_data[2],
            'gamma': mat_data[3],
            'a': mat_data[4],
            'b': mat_data[5],
        }

    n = rwhale_input_data['SystemPerformance']['WaterDistributionNetwork'][
        'node_damage_model'
    ]
    n = n[0]
    REWET_input_data['settings']['node_damage_model'] = {
        'x': 0.9012,
        'a': n[0],
        'aa': n[1],
        'b': n[2],
        'bb': n[3],
        'c': n[4],
        'cc': n[5],
        'd': n[6],
        'dd': n[7],
        'e': n[8],
        'ee1': n[9],
        'ee2': n[10],
        'f': n[11],
        'ff1': n[12],
        'ff2': n[13],
        'damage_node_model': 'equal_diameter_emitter',
    }

    if rwhale_input_data['SystemPerformance']['WaterDistributionNetwork'][
        'Pipe_Leak_Based'
    ]:
        pipe_leak_amount = rwhale_input_data['SystemPerformance'][
            'WaterDistributionNetwork'
        ]['pipe_leak_amount']
        pipe_leak_time = rwhale_input_data['SystemPerformance'][
            'WaterDistributionNetwork'
        ]['pipe_leak_time']
        pipe_damage_discovery_mode = {
            'method': 'leak_based',
            'leak_amount': pipe_leak_amount,
            'leak_time': pipe_leak_time,
        }
    else:
        pipe_time_discovery_ratio = rwhale_input_data['SystemPerformance'][
            'WaterDistributionNetwork'
        ]['pipe_time_discovery_ratio']
        pipe_damage_discovery_mode = {
            'method': 'time_based',
            'time_discovery_ratio': pipe_time_discovery_ratio,
        }  # pd.Series([line[0] for line in pipe_time_discovery_ratio], index = [line[1] for line in pipe_time_discovery_ratio])}

    if rwhale_input_data['SystemPerformance']['WaterDistributionNetwork'][
        'Node_Leak_Based'
    ]:
        node_leak_amount = rwhale_input_data['SystemPerformance'][
            'WaterDistributionNetwork'
        ]['node_leak_amount']
        node_leak_time = rwhale_input_data['SystemPerformance'][
            'WaterDistributionNetwork'
        ]['node_leak_time']
        node_damage_discovery_mode = {
            'method': 'leak_based',
            'leak_amount': node_leak_amount,
            'leak_time': node_leak_time,
        }
    else:
        node_time_discovery_ratio = rwhale_input_data['SystemPerformance'][
            'WaterDistributionNetwork'
        ]['node_time_discovery_ratio']
        node_damage_discovery_mode = {
            'method': 'time_based',
            'time_discovery_ratio': node_time_discovery_ratio,
        }  # pd.Series([line[0] for line in node_time_discovery_ratio], index = [line[1] for line in node_time_discovery_ratio])}

    pump_time_discovery_ratio = rwhale_input_data['SystemPerformance'][
        'WaterDistributionNetwork'
    ]['pump_time_discovery_ratio']
    tank_time_discovery_ratio = rwhale_input_data['SystemPerformance'][
        'WaterDistributionNetwork'
    ]['tank_time_discovery_ratio']
    pump_damage_discovery_model = {
        'method': 'time_based',
        'time_discovery_ratio': pump_time_discovery_ratio,
    }  # pd.Series([line[0] for line in pump_time_discovery_ratio], index = [line[1] for line in pump_time_discovery_ratio])}
    tank_damage_discovery_model = {
        'method': 'time_based',
        'time_discovery_ratio': tank_time_discovery_ratio,
    }  # pd.Series([line[0] for line in tank_time_discovery_ratio], index = [line[1] for line in tank_time_discovery_ratio])}

    REWET_input_data['settings']['pipe_damage_discovery_model'] = (
        pipe_damage_discovery_mode
    )
    REWET_input_data['settings']['node_damage_discovery_model'] = (
        node_damage_discovery_mode
    )
    REWET_input_data['settings']['pump_damage_discovery_model'] = (
        pump_damage_discovery_model
    )
    REWET_input_data['settings']['tank_damage_discovery_model'] = (
        tank_damage_discovery_model
    )
    REWET_input_data['settings']['minimum_pressure'] = rwhale_input_data[
        'SystemPerformance'
    ]['WaterDistributionNetwork']['minimum_pressure']
    REWET_input_data['settings']['required_pressure'] = rwhale_input_data[
        'SystemPerformance'
    ]['WaterDistributionNetwork']['required_pressure']

    # Not Supposed to be in R2DTool GUI ############
    REWET_input_data['settings']['minimum_simulation_time'] = (
        0  # TODO : HERE #REWET_input_data["event_time"] + REWET_input_data["settings"]["simulation_time_step"]  # noqa: TD002
    )
    REWET_input_data['settings']['save_time_step'] = True
    REWET_input_data['settings']['record_restoration_agent_logs'] = True
    REWET_input_data['settings']['record_damage_table_logs'] = True
    REWET_input_data['settings']['simulation_time_step'] = 3600
    REWET_input_data['settings']['number_of_proccessor'] = 1
    REWET_input_data['settings']['demand_ratio'] = 1
    REWET_input_data['settings']['dmg_rst_data_save'] = True
    REWET_input_data['settings']['Parameter_override'] = True
    REWET_input_data['settings']['mpi_resume'] = (
        True  # ignores the scenarios that are done
    )
    REWET_input_data['settings']['ignore_empty_damage'] = False
    REWET_input_data['settings']['result_details'] = 'extended'
    REWET_input_data['settings']['negative_node_elmination'] = True
    REWET_input_data['settings']['nne_flow_limit'] = 0.5
    REWET_input_data['settings']['nne_pressure_limit'] = -5
    REWET_input_data['settings']['Virtual_node'] = True
    REWET_input_data['settings']['damage_node_model'] = (
        'equal_diameter_emitter'  # "equal_diameter_reservoir"
    )
    REWET_input_data['settings']['default_pipe_damage_model'] = {
        'alpha': -0.0038,
        'beta': 0.1096,
        'gamma': 0.0196,
        'a': 2,
        'b': 1,
    }

    REWET_input_data['settings'][
        'limit_result_file_size'
    ] = -1  # in Mb. 0 means no limit
    REWET_input_data['settings']['Pipe_damage_input_method'] = 'pickle'


def create_path(path):  # noqa: D103
    if isinstance(path, str):
        path = Path(path)
    not_existing_hir = []
    while os.path.exists(path) == False:  # noqa: PTH110, E712
        not_existing_hir.append(path.name)
        path = path.parent

    while len(not_existing_hir):
        new_path = path / not_existing_hir[-1]
        new_path.mkdir()
        not_existing_hir.pop(-1)
        path = new_path


if __name__ == '__main__':
    # Setting arg parser
    arg_parser = argparse.ArgumentParser('Preprocess rwhale workflow to REWET input.')

    arg_parser.add_argument(
        '--input',
        '-i',
        default='inputRWHALE.json',
        help='rwhale input file json file',
    )

    arg_parser.add_argument('--dir', '-d', help='WDN damage result directory')

    arg_parser.add_argument(
        '--number',
        '-n',
        default=None,
        help='If specified, indicates realization number, otherwise, all scenarios are run on all CPUS.',
    )

    arg_parser.add_argument(
        '--par',
        '-p',
        default=False,
        action='store_true',
        help='if specified, uses all CPUS. 2 or more CPUs are not available, it will revert back to serial run.',
    )

    parser_data = arg_parser.parse_args()

    # Setting parallel or serial settings

    proc_size = 1
    proc_ID = 0
    do_parallel = False

    mpi_spec = importlib.util.find_spec('mpi4py')
    found = mpi_spec is not None
    if found and arg_parser.par:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        proc_size = comm.Get_size()
        proc_ID = comm.Get_rank()
        if proc_size < 2:
            do_parallel = False
            proc_size = 1
            proc_ID = 0
            print(
                'Parallel running is not possible. Number of CPUS are are not enough.'
            )
        else:
            do_parallel = True

    # Setting up run settings
    REWET_input_data = {}
    REWET_input_data['settings'] = {}

    rwhale_input_data = preprocessorIO.readJSONFile(parser_data.input)  # noqa: N816
    setSettingsData(rwhale_input_data, REWET_input_data)
    event_time = rwhale_input_data['SystemPerformance']['WaterDistributionNetwork'][
        'eventTime'
    ]

    # Set R2D environment and parameters
    water_asset_data = rwhale_input_data['Applications']['Assets'][
        'WaterDistributionNetwork'
    ]
    inp_file_addr = water_asset_data['ApplicationData']['inpFile']
    if '{Current_Dir}' in inp_file_addr:
        inp_file_addr = inp_file_addr.replace('{Current_Dir}', '.')
    sc_geojson = rwhale_input_data['Applications']['Assets'][
        'WaterDistributionNetwork'
    ]['ApplicationData']['assetSourceFile']

    run_directory = rwhale_input_data['runDir']
    number_of_realization = rwhale_input_data['Applications']['DL'][
        'WaterDistributionNetwork'
    ]['ApplicationData']['Realizations']

    REWET_input_data['settings']['result_directory'] = os.path.join(  # noqa: PTH118
        run_directory, 'Results', 'WaterDistributionNetwork', 'REWET_Result'
    )

    REWET_input_data['settings']['temp_directory'] = os.path.join(  # noqa: PTH118
        run_directory, 'Results', 'WaterDistributionNetwork', 'REWET_RunFiles'
    )

    REWET_input_data['settings']['WN_INP'] = inp_file_addr

    damage_save_path = (
        Path(run_directory) / 'Results' / 'WaterDistributionNetwork' / 'damage_input'
    )
    damage_save_path_hir = damage_save_path
    create_path(damage_save_path_hir)

    if parser_data.number is None:
        scneario_list_path = damage_save_path / 'scenario_table.xlsx'
    else:
        scneario_list_path = (
            damage_save_path / f'scenario_table_{parser_data.number}.xlsx'
        )

    REWET_input_data['settings']['pipe_damage_file_list'] = str(scneario_list_path)
    REWET_input_data['settings']['pipe_damage_file_directory'] = str(
        damage_save_path
    )

    # Add Single Scenario or multiple scenario
    Damage_file_name = []

    if do_parallel and proc_ID > 0:
        pass
    else:
        settings_json_file_path = preprocessorIO.saveSettingsFile(
            REWET_input_data, damage_save_path, parser_data.number
        )

        scenario_table = preprocessorIO.create_scneario_table()

        if parser_data.number is None:
            Damage_file_name = list(range(number_of_realization))

        else:
            Damage_file_name.append(parser_data.number)

        damage_save_path = scneario_list_path.parent

        for scn_number in Damage_file_name:
            dl_file_path, dl_file_dir = getDLFileName(
                run_directory, parser_data.dir, scn_number
            )

            damage_data = damage_convertor.readDamagefile(
                dl_file_path, run_directory, event_time, sc_geojson
            )

            cur_damage_file_name_list = preprocessorIO.save_damage_data(
                damage_save_path, damage_data, scn_number
            )

            scenario_table = preprocessorIO.update_scenario_table(
                scenario_table, cur_damage_file_name_list, scn_number
            )

        preprocessorIO.save_scenario_table(
            scenario_table, REWET_input_data['settings']['pipe_damage_file_list']
        )

    command = (
        'python '
        f'C:\\Users\\naeim\\Desktop\\REWET\\main.py -j {settings_json_file_path}'
    )

    # try:
    # result = subprocess.check_output(command, shell=True, text=True)
    # returncode = 0
    # except subprocess.CalledProcessError as e:
    # result = e.output
    # returncode = e.returncode

    # if returncode != 0:
    # print('return code: {}'.format(returncode))
    # if returncode == 0:
    # print("REWET ran Successfully")

    create_path(REWET_input_data['settings']['result_directory'])
    create_path(REWET_input_data['settings']['temp_directory'])

    rewet_log_path = (
        Path(run_directory)
        / 'Results'
        / 'WaterDistributionNetwork'
        / 'rewet_log.txt'
    )

    system_std_out = sys.stdout
    with open(rewet_log_path, 'w') as log_file:  # noqa: PTH123
        sys.stdout = log_file
        REWET_starter = Starter()
        REWET_starter.run(settings_json_file_path)

        p = Project_Result(
            Path(REWET_input_data['settings']['result_directory']) / 'project.prj'
        )

        # these are the input for result section. They are not include in the
        requested_result = ['DL', 'QN']
        substitute_ft = {'DL': 'Delivery', 'QN': 'Quantity'}
        consistency_time_window = 0  # 7200
        iConsider_leak = False  # True  # noqa: N816
        # the following does not matter if iConsider_leak is false
        leak_ratio = {'DL': 0.75, 'QN': 0}

        sub_asset_list = ['Junction', 'Pipe', 'Reservoir']
        sub_asset_name_to_id = dict()  # noqa: C408
        sub_asset_id_to_name = dict()  # noqa: C408
        for sub_asset in sub_asset_list:
            sc_geojson_file = preprocessorIO.readJSONFile(sc_geojson)
            sub_asset_data = [
                ss
                for ss in sc_geojson_file['features']
                if ss['properties']['type'] == sub_asset
            ]
            sub_asset_id = [str(ss['id']) for ss in sub_asset_data]
            sub_asset_name = [ss['properties']['InpID'] for ss in sub_asset_data]
            sub_asset_name_to_id.update(
                {sub_asset: dict(zip(sub_asset_name, sub_asset_id))}
            )
            sub_asset_id_to_name.update(
                {sub_asset: dict(zip(sub_asset_id, sub_asset_name))}
            )

        res = {}
        res_agg = {}
        scneario_size = len(p.project.scenario_list.index)
        # create a dictionary with keys for each scenario number (in int) and keys for a BSC metric (None)
        temp_realization_in_each_time_series = dict(
            zip(range(scneario_size), [None] * scneario_size)
        )
        # create a dictionary for storing times series, each BSC metric (requested_result) is a key and each key has the dictionary created in line before
        time_series_result = dict(
            zip(
                requested_result,
                [temp_realization_in_each_time_series] * len(requested_result),
            )
        )

        for scn_name, row in p.project.scenario_list.iterrows():  # noqa: B007
            realization_number = int(scn_name.strip('SCN_'))
            for single_requested_result in requested_result:

                if single_requested_result in {'DL', 'QN'}:

                    # Running Output module's method to get DL time series status
                    time_series_result[single_requested_result][
                        realization_number
                    ] = p.getBSCIndexPopulation_4(
                        scn_name,
                        bsc=single_requested_result,
                        iPopulation=False,
                        ratio=True,
                        consider_leak=False,
                        leak_ratio=1,
                    )
                    time_series_result[single_requested_result][
                        realization_number
                    ].index = (
                        time_series_result[single_requested_result][
                            realization_number
                        ].index / 3600
                    )

                    # Run Output module's method to get BSC data for
                    # each junction (sum of outage)
                    res[single_requested_result] = p.getOutageTimeGeoPandas_5(
                        scn_name,
                        bsc=single_requested_result,
                        iConsider_leak=False,
                        leak_ratio=leak_ratio,
                        consistency_time_window=consistency_time_window,
                        sum_time=True,
                    )
                    if res_agg.get(single_requested_result) is None:
                        res_agg[single_requested_result] = res[
                            single_requested_result
                        ].to_dict()
                        for key in res_agg[single_requested_result]:
                            res_agg[single_requested_result][key] = [
                                res_agg[single_requested_result][key]
                            ]
                    else:
                        for key in res_agg[single_requested_result]:
                            res_agg[single_requested_result][key].append(
                                res[single_requested_result][key]
                            )

            cur_json_file_name = (
                f'WaterDistributionNetwork_{realization_number}.json'
            )
            cur_json_file_path = (
                Path(run_directory)
                / 'Results'
                / 'WaterDistributionNetwork'
                / cur_json_file_name
            )

            with open(cur_json_file_path) as f:  # noqa: PTH123
                json_data = json.load(f)

            for single_requested_result in requested_result:
                req_result = res[single_requested_result]
                result_key = f'{substitute_ft[single_requested_result]}Outage'

                # Only Junction needs to be added to rlz json
                junction_json_data = json_data['WaterDistributionNetwork'].get(
                    'Junction', {}
                )

                for junction_name in req_result.keys():  # noqa: SIM118
                    junction_id = sub_asset_name_to_id['Junction'][junction_name]
                    cur_junction = junction_json_data.get(junction_id, {})
                    cur_junction_SP = cur_junction.get('SystemPerformance', {})  # noqa: N816
                    cur_junction_SP[result_key] = float(req_result[junction_name])

                    cur_junction['SystemPerformance'] = cur_junction_SP
                    junction_json_data[junction_id] = cur_junction

                json_data['WaterDistributionNetwork']['Junction'] = (
                    junction_json_data
                )

            with open(cur_json_file_path, 'w') as f:  # noqa: PTH123
                json_data = json.dump(json_data, f, indent=2)

        res_agg_mean = dict()  # noqa: C408
        res_agg_std = dict()  # noqa: C408
        for single_requested_result in requested_result:
            res_agg[single_requested_result] = pd.DataFrame(
                res_agg[single_requested_result]
            )
            res_agg_mean[single_requested_result] = res_agg[
                single_requested_result
            ].mean()
            res_agg_std[single_requested_result] = res_agg[
                single_requested_result
            ].std()
    sys.stdout = system_std_out

    # Append junction and reservior general information to WaterDistributionNetwork_det
    det_json_path = cur_json_file_path = (
        Path(run_directory)
        / 'Results'
        / 'WaterDistributionNetwork'
        / 'WaterDistributionNetwork_det.json'
    )
    det_json = preprocessorIO.readJSONFile(det_json_path)
    inp_json = preprocessorIO.readJSONFile(sc_geojson)
    inp_json = inp_json['features']
    for WDNtype in ['Reservoir', 'Junction']:
        json_to_attach = dict()  # noqa: C408
        for ft in inp_json:
            prop = ft['properties']
            if prop['type'] == WDNtype:
                id = str(ft['id'])  # noqa: A001
                generalInfo = dict()  # noqa: C408, N816
                json_geometry = ft['geometry']
                shapely_geometry = geometry.shape(json_geometry)
                wkt_geometry = shapely_geometry.wkt
                generalInfo.update({'geometry': wkt_geometry})
                asset_name = sub_asset_id_to_name[WDNtype][id]
                generalInfo.update({'REWET_id': asset_name})
                generalInfo.update({'AIM_id': id})
                for key, item in prop.items():
                    if key == 'id':
                        continue
                    generalInfo.update({key: item})
                R2Dres = dict()  # noqa: C408
                asset_name = sub_asset_id_to_name[WDNtype][id]
                for single_requested_result in requested_result:
                    if asset_name not in res_agg_mean[single_requested_result].index:
                        continue
                    R2Dres_key_mean = f'R2Dres_mean_{single_requested_result}'
                    R2Dres_key_std = f'R2Dres_std_{single_requested_result}'
                    R2Dres.update(
                        {
                            R2Dres_key_mean: res_agg_mean[single_requested_result][
                                asset_name
                            ],
                            R2Dres_key_std: res_agg_std[single_requested_result][
                                asset_name
                            ],
                        }
                    )
                # location = dict()
                # location.update({'latitude':ft['geometry']['coordinates'][1],\
                #                 'longitude':ft['geometry']['coordinates'][0]})
                # generalInfo.update({'location':location})
                json_to_attach.update(
                    {id: {'GeneralInformation': generalInfo, 'R2Dres': R2Dres}}
                )
        det_json['WaterDistributionNetwork'].update({WDNtype: json_to_attach})
    with open(det_json_path, 'w') as f:  # noqa: PTH123
        json.dump(det_json, f, indent=2)

    ts_result_json_path = cur_json_file_path = (
        Path(run_directory)
        / 'Results'
        / 'WaterDistributionNetwork'
        / 'WaterDistributionNetwork_timeseries.json'
    )
    time_series_result_struc = {
        'Type': 'TimeSeries',
        'Asset': 'WaterDistributionNetwork',
        'Result': {
            'QN': {'Name': 'Water Quantity', 'Data': {}},
            'DL': {'Name': 'Water Delivery', 'Data': {}},
        },
    }

    for single_requested_result in requested_result:
        for i in range(scneario_size):
            time_series_result_struc['Result'][single_requested_result]['Data'][
                i
            ] = time_series_result[single_requested_result][i].to_dict()

    with open(ts_result_json_path, 'w') as f:  # noqa: PTH123
        json.dump(time_series_result_struc, f, indent=2)
    print('here')  # noqa: T201
