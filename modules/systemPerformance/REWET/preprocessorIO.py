#
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

import json
import os

import pandas as pd


def readJSONFile(file_addr):
    """Reads a json file.

    Parameters
    ----------
    file_addr : Path
        JSON file address.

    Raises
    ------
    ValueError
        If the file is not found on the address.

    Returns
    -------
    data : dict
        JSON File data as a dict.

    """
    if not os.path.exists(file_addr):
        raise ValueError('INPUT WHALE FILE is not found.', repr(file_addr))

    with open(file_addr) as f:
        data = json.load(f)

    return data


# =============================================================================
# def readRWHALEFileForREWET(file_addr, REWET_input_data):
#     """
#     Reads rwhile input file and returns the data as a dict and updates REWET
#     input file.
#
#     Parameters
#     ----------
#     file_addr : Path
#         rwhale input file path.
#     REWET_input_data : dict
#         REWET input data.
#
#     Returns
#     -------
#     rwhale_data : dict
#         rwhale inoput data as a dict.
#
#     """
#
#
#     water_asset_data = rwhale_data["Applications"]\
#         ["Assets"]["WaterDistributionNetwork"]
#     inp_file_addr = water_asset_data["ApplicationData"]["inpFile"]
#     run_directory = rwhale_data["runDir"]
#     number_of_realization = rwhale_data["Applications"]\
#         ["DL"]["WaterDistributionNetwork"]["ApplicationData"]["Realizations"]
#
#     REWET_input_data["inp_file" ] = inp_file_addr
#     REWET_input_data["run_dir"] = run_directory
#     REWET_input_data["number_of_realizations"] = number_of_realization
#
#     return rwhale_data
# =============================================================================


def save_damage_data(damage_save_path, damage_data, scn_number):
    pipe_damage_data = damage_data['Pipe']
    node_damage_data = damage_data['Node']
    pump_damage_data = damage_data['Pump']
    tank_damage_data = damage_data['Tank']

    pipe_damage_file_name = f'pipe_damage_{scn_number}'
    node_damage_file_name = f'node_damage_{scn_number}'
    pump_damage_file_name = f'pump_damage_{scn_number}'
    tank_damage_file_name = f'tank_damage_{scn_number}'

    pipe_damage_file_path = os.path.join(damage_save_path, pipe_damage_file_name)
    node_damage_file_path = os.path.join(damage_save_path, node_damage_file_name)
    pump_damage_file_path = os.path.join(damage_save_path, pump_damage_file_name)
    tank_damage_file_path = os.path.join(damage_save_path, tank_damage_file_name)

    pipe_damage_data.to_pickle(pipe_damage_file_path)
    node_damage_data.to_pickle(node_damage_file_path)
    pump_damage_data.to_pickle(pump_damage_file_path)
    tank_damage_data.to_pickle(tank_damage_file_path)

    damage_file_name_list = {
        'Pipe': pipe_damage_file_name,
        'Node': node_damage_file_name,
        'Pump': pump_damage_file_name,
        'Tank': tank_damage_file_name,
    }

    return damage_file_name_list


def create_scneario_table():
    scenario_table = pd.DataFrame(
        dtype='O',
        columns=[
            'Scenario Name',
            'Pipe Damage',
            'Nodal Damage',
            'Pump Damage',
            'Tank Damage',
            'Probability',
        ],
    )
    return scenario_table


def update_scenario_table(scenario_table, cur_damage_file_name_list, scn_number):
    if isinstance(scenario_table, pd.core.frame.DataFrame):
        scenario_table = scenario_table.to_dict('records')
    elif isinstance(scenario_table, list):
        pass
    else:
        raise ValueError('This is an unknown behavior.')

    new_row = {
        'Scenario Name': f'SCN_{scn_number}',
        'Pipe Damage': cur_damage_file_name_list['Pipe'],
        'Nodal Damage': cur_damage_file_name_list['Node'],
        'Pump Damage': cur_damage_file_name_list['Pump'],
        'Tank Damage': cur_damage_file_name_list['Tank'],
        'Probability': 1,
    }

    scenario_table.append(new_row)

    return scenario_table


def save_scenario_table(scenario_table, scenario_table_file_path):
    """Saves the scneario data including scneario table and damaghe data acording
    to the table data/

    Parameters
    ----------
    REWET_input_data : Dict
        REWET input data.

    Returns
    -------
    None.

    """
    if isinstance(scenario_table, pd.core.frame.DataFrame):
        pass
    elif isinstance(scenario_table, list):
        scenario_table = pd.DataFrame(scenario_table)
    else:
        raise ValueError('This is an unknown behavior.')

    scenario_table = scenario_table.set_index('Scenario Name')

    # scenario_list_file_path = os.path.join(damage_save_path, scenario_list_file_name)

    scenario_table.to_excel(scenario_table_file_path)


def saveSettingsFile(REWET_input_data, save_directory, prefix):
    """Saves seetings data that REWET NEEDs.

    Parameters
    ----------
    REWET_input_data : Dict
        REWET input data.

    Returns
    -------
    None.

    """
    settings = REWET_input_data['settings']
    if prefix == None:
        settings_file_name = 'settings.json'
    else:
        settings_file_name = prefix + '_' + 'settings.json'
    damage_save_path = save_directory / settings_file_name
    with open(damage_save_path, 'w') as f:
        json.dump(settings, f, indent=4)

    return damage_save_path
