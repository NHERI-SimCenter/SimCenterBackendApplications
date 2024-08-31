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

import pandas as pd
import preprocessorIO

CBIG_int = int(1e9)
LEAK_VALUE = 1
BREAK_VALUE = 2


def create_pipe_damage_input_for_rewet(
    pipe_damage_data, run_dir, event_time, sc_geojson
):
    """Create REWET-style pipe damage file.

    Parameters
    ----------
    pipe_damage_data : dict
        Pipe damage data from PELICUN.
    rewet_input_data : dict
        REWET input data.

    Raises
    ------
    ValueError
        If damage type is not what it should be.

    Returns
    -------
    pipe_damage_list : Pandas Series
        REWET-style pipe damage file.

    """
    pipe_id_list = list(pipe_damage_data.keys())

    damage_list = []
    damage_time = event_time
    sc_geojson_file = preprocessorIO.read_json_file(sc_geojson)
    pipe_data = [
        ss
        for ss in sc_geojson_file['features']
        if ss['properties']['type'] == 'Pipe'
    ]
    pipe_index = [str(ss['id']) for ss in pipe_data]
    pipe_id = [ss['properties']['InpID'] for ss in pipe_data]
    pipe_index_to_id = dict(zip(pipe_index, pipe_id))

    for pipe_id in pipe_id_list:
        cur_data = pipe_damage_data[pipe_id]

        cur_damage = cur_data['Damage']

        aim_data = find_read_aim_file(
            pipe_id,
            'WaterDistributionNetwork',
            'Pipe',
            run_dir,
        )

        material = aim_data['GeneralInformation'].get('Material', None)

        # If material is not ptovided, then the material is CI as the default
        if material is None:
            material = 'CI'

        aggregates_list = [
            cur_agg for cur_agg in list(cur_damage.keys()) if 'aggregate' in cur_agg
        ]
        segment_sizes = len(aggregates_list)
        segment_step = 1 / segment_sizes
        c = 0

        for cur_agg in aggregates_list:
            damage_val = cur_damage[cur_agg]
            if damage_val > 0:
                if damage_val == LEAK_VALUE:
                    damage_type = 'leak'
                elif damage_val == BREAK_VALUE:
                    damage_type = 'break'
                else:
                    raise ValueError('The damage type must be either 1 or 2')  # noqa: EM101, TRY003
            else:
                continue

            cur_loc = c * segment_step + segment_step / 2
            c += 1
            damage_list.append(
                {
                    'pipe_id': pipe_index_to_id[pipe_id],
                    'damage_loc': cur_loc,
                    'type': damage_type,
                    'Material': material,
                }
            )
    damage_list.reverse()
    pipe_damage_list = pd.Series(
        data=damage_list, index=[damage_time for val in damage_list], dtype='O'
    )

    return pipe_damage_list  # noqa: RET504


def create_node_damage_input_for_rewet(node_damage_data, run_dir, event_time):
    """Create REWET-style node damage file.

    Parameters
    ----------
    node_damage_data : dict
        Node damage data from PELICUN.
    rewet_input_data : dict
        REWET input data.

    Returns
    -------
    node_damage_list : Pandas Series
        REWET-style node damage file.

    """
    node_id_list = node_damage_data.keys()

    damage_list = []
    damage_time = event_time

    for node_id in node_id_list:
        cur_damage = node_damage_data[node_id]
        aggregates_list = [
            cur_agg for cur_agg in list(cur_damage.keys()) if 'aggregate' in cur_agg
        ]

        if len(aggregates_list) == 0:
            continue

        cur_data = node_damage_data[node_id]

        cur_damage = cur_data['Damage']

        aim_data = find_read_aim_file(
            node_id,
            'WaterDistributionNetwork',
            'Node',
            run_dir,
        )

        total_length = aim_data['GeneralInformation'].get('Total_length', None)
        total_number_of_damages = cur_damage['aggregate']

        damage_list.append(
            {
                'node_name': node_id,
                'number_of_damages': total_number_of_damages,
                'node_Pipe_Length': total_length,
            }
        )

    node_damage_list = pd.Series(
        data=damage_list, index=[damage_time for val in damage_list], dtype='O'
    )

    return node_damage_list  # noqa: RET504


def create_pump_damage_input_for_rewet(pump_damage_data, rewet_input_data):
    """Create REWET-style pump damage file.

    Parameters
    ----------
    pump_damage_data : dict
        Pump damage data from PELICUN.
    rewet_input_data : dict
        REWET input data.

    Returns
    -------
    pump_damage_list : Pandas Series
        REWET-style pump damage file.

    """
    pump_id_list = list(pump_damage_data.keys())

    damage_list = []
    damage_time = rewet_input_data['event_time']

    for pump_id in pump_id_list:
        cur_data = pump_damage_data[pump_id]

        cur_damage = cur_data['Damage']
        cur_repair_time = cur_data['Repair']

        if cur_damage == 0:
            continue  # cur_damage_state = 0 means undamaged pump

        # (SINA) I'm not sure if we need any data about the pump at this point

        # aim_data = findAndReadAIMFile(tank_id, os.path.join(
        # "Results", "WaterDistributionNetwork", "Pump"),
        # rewet_input_data["run_dir"])

        # We are getting this data from PELICUN
        # restore_time = getPumpRetsoreTime(cur_damage)
        damage_list.append(
            {
                'pump_id': pump_id,
                'time': damage_time,
                'Restore_time': cur_repair_time,
            }
        )
    pump_damage_list = pd.Series(
        index=[damage_time for val in damage_list], data=damage_list
    )

    return pump_damage_list  # noqa: RET504


def create_tank_damage_input_for_rewet(tank_damage_data, rewet_input_data):
    """Create REWET-style Tank damage file.

    Parameters
    ----------
    tank_damage_data : dict
        Tank damage data from PELICUN.
    rewet_input_data : dict
        REWET input data.

    Returns
    -------
    tank_damage_list : Pandas Series
        REWET-style tank damage file.

    """
    tank_id_list = tank_damage_data.keys()

    damage_list = []
    damage_time = rewet_input_data['event_time']

    for tank_id in tank_id_list:
        cur_data = tank_damage_data[tank_id]

        cur_damage = cur_data['Damage']
        cur_repair_time = cur_data['Repair']

        if cur_damage == 0:
            continue  # cur_damage_state = 0 meeans undamged tank

        # =============================================================================
        #         # We are getting his data from REWET
        #
        #         aim_data = findAndReadAIMFile(tank_id, os.path.join(
        #             "Results", "WaterDistributionNetwork", "Tank"),
        #                                            rewet_input_data["run_dir"])
        #         tank_type = aim_data["GeneralInformation"].get("Type", None)
        #         restore_time = getTankRetsoreTime(tank_type, cur_damage)
        # =============================================================================

        damage_list.append(
            {
                'tank_id': tank_id,
                'time': damage_time,
                'Restore_time': cur_repair_time,
            }
        )

    tank_damage_list = pd.Series(
        index=[damage_time for val in damage_list], data=damage_list
    )

    return tank_damage_list  # noqa: RET504


def find_read_aim_file(asset_id, asset_type, asset_sub_type, run_dir):
    """Find and read the AIM file for an asset.

    Parameters
    ----------
    asset_id : int
        The asset ID.
    asset_type : str
        Asset Type (e.g., Building, WaterDistributionNetwork).
    run_dir : path
        The directory where data is stored (aka the R2dTool directory)

    Returns
    -------
    aim_file_data : dict
        AIM file data as a dict.

    """
    file_path = (
        run_dir
        / asset_type
        / asset_sub_type
        / str(asset_id)
        / 'templatedir'
        / f'{asset_id}-AIM.json'
    )
    aim_file_data = preprocessorIO.read_json_file(str(file_path))
    return aim_file_data  # noqa: RET504


def read_damage_file(file_addr, run_dir, event_time, sc_geojson):
    """Read PELICUN damage files.

    Read PELICUN damage files and create REWET-Style damage for all
    WaterDistributionNetwork elements.

    Parameters
    ----------
    file_addr : path
        PELICUN damage file in JSON format.
    rewet_input_data : dict
        REWET input data, which is updated in the function.
    scn_number : dict
        JSON FILE.

    Returns
    -------
    damage_data : dict
        Damage data in PELICUN dict format.

    """
    # TODO(SINA): Make reading once for each scenario

    damage_data = preprocessorIO.read_json_file(file_addr)

    wn_damage_data = damage_data['WaterDistributionNetwork']

    if 'Pipe' in wn_damage_data:
        pipe_damage_data = create_pipe_damage_input_for_rewet(
            wn_damage_data['Pipe'], run_dir, event_time, sc_geojson
        )
    else:
        pipe_damage_data = pd.Series(dtype='O')

    if 'Tank' in wn_damage_data:
        tank_damage_data = create_tank_damage_input_for_rewet(
            wn_damage_data['Tank'], run_dir, event_time
        )
    else:
        tank_damage_data = pd.Series(dtype='O')

    if 'Pump' in wn_damage_data:
        pump_damage_data = create_pump_damage_input_for_rewet(
            wn_damage_data['Pump'], run_dir, event_time
        )
    else:
        pump_damage_data = pd.Series(dtype='O')

    if 'Junction' in wn_damage_data:
        node_damage_data = create_node_damage_input_for_rewet(
            wn_damage_data['Junction'], run_dir, event_time
        )
    else:
        node_damage_data = pd.Series(dtype='O')

    damage_data = {}
    damage_data['Pipe'] = pipe_damage_data
    damage_data['Tank'] = tank_damage_data
    damage_data['Pump'] = pump_damage_data
    damage_data['Node'] = node_damage_data

    return damage_data
