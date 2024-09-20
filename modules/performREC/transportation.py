"""Methods for performance simulations of transportation networks."""

# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 The Regents of the University of California
#
# This file is part of SimCenter Backend Applications.
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
# BRAILS. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Barbaros Cetiner
# Tianyu Han
#
# Last updated:
# 08-14-2024

from __future__ import annotations

import gc
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandana.network as pdna
import pandas as pd
from brails.workflow.TransportationElementHandler import TransportationElementHandler
from scipy.spatial.distance import cdist
from shapely.wkt import loads


class TransportationPerformance(ABC):
    """
    An abstract base class for simulating transportation networks.

    This class provides an interface for implementing methods that process
    transportation data (such as system state and origin-destination files) and
    compute network performance metrics.

    Subclasses must define how to process these data inputs and update system
    performance in concrete terms.

    Attributes__
        assets (list): A list of asset types (e.g., 'Bridge', 'Roadway',
                      'Tunnel') to be analyzed in the transportation network.
        csv_filenames (list): A list of filenames (e.g., 'edges.csv', '
                              nodes.csv', 'closed_edges.csv') that are required
                              for network simulations.
        capacity_map (dict): A mapping that relates damage states to capacity
                             ratios. For example, damage states of 0-2 may
                             represent full capacity (1), while higher damage
                             states reduce capacity (e.g., 0.5 or 0).
        no_identifier (dict): A mapping of asset types to their unique
                              identifiers (e.g., 'StructureNumber' for Bridges,
                              'TigerOID' for Roadways). These are used to track
                              and manage assets within the system.

    Methods__
        system_state(detfile: str, csv_file_dir: str) -> None:
            Abstract method to process a given det (damage state) file and
            update the system state with capacity ratios. Also checks for
            missing or necessary CSV files for network simulation.

        update_od_file(old_nodes: str, old_det: str, new_nodes: str, new_det:
                       str, od: str, origin_ids: list) -> pd.DataFrame:
            Abstract method to update the origin-destination (OD) file based
            on population changes between time steps. The method tracks
            population growth in nodes and generates new trips in the OD file
            accordingly.

        system_performance(detfile: str, csv_file_dir: str) -> None:
            Abstract method to compute or update the performance of the
            transportation system based on current state and available data.

    Notes__
        This is an abstract class. To use it, create a subclass that implements
        the abstract methods for specific behavior related to transportation
        network performance analysis.
    """

    def __init__(self,
                 assets: list[str] | None = None,
                 capacity_map: dict[int, float] | None = None,
                 csv_files: dict[str, str] | None = None,
                 no_identifier: dict[str, str] | None = None):
        """
        Initialize the TransportationPerformance class with essential data.

        Args__
            assets (list): A list of asset types such as 'Bridge', 'Roadway',
                           'Tunnel'.
            capacity_map (dict): A mapping of damage states to capacity ratios.
            csv_files (dict): A dictionary of CSV filenames for network data,
                              including 'network_edges', 'network_nodes',
                              'edge_closures', and 'od_pairs'.
            no_identifier (dict): A mapping of asset types to their unique
                                  identifiers.
        """
        if assets is None:
            assets = ['Bridge', 'Roadway', 'Tunnel']
        if capacity_map is None:
            capacity_map = {0: 1, 1: 1, 2: 1, 3: 0.5, 4: 0}
        if csv_files is None:
            csv_files = {'network_edges': 'edges.csv',
                         'network_nodes': 'nodes.csv',
                         'edge_closures': 'closed_edges.csv',
                         'od_pairs': 'od.csv'}
        if no_identifier is None:
            no_identifier = {'Bridge': 'StructureNumber',
                             'Roadway': 'TigerOID',
                             'Tunnel': 'TunnelNumber'}

        self.assets = assets
        self.csv_files = csv_files
        self.capacity_map = capacity_map
        self.no_identifier = no_identifier

    @abstractmethod
    def system_state(self,
                     initial_state: str,
                     damage_states: str) -> dict:  # updated_state
        """
        Process given det and damage results file to get updated system state.

        This function reads a JSON file containing undamaged network attributes
        and JSON file containing damage states and updates the state of the
        network (i.e., determines edges experiencing capacity reductions)
        using  capacity ratios defined for each damage state. It also checks
        for the existence of required CSV files, created the if missing, and
        generates a file listing closed edges.

        Args__
            detfile (str): Path to the JSON file containing the asset data.
            csv_file_dir (str): Directory containing the CSV files needed for
                                    running network simulations.

        Returns__
            The function does not return any value. It creates updated det file
            and CSV file necessary to run network simulations

        Raises__
            FileNotFoundError: If the `detfile` or any required CSV files are
                               not found.
            json.JSONDecodeError: If the `detfile` cannot be parsed as JSON.
            KeyError: If expected keys are missing in the JSON data.
            ValueError: If there are issues with data conversions, e.g., JSON
                        to integer.

        Examples__
            >>> system_state('damage_states.json', '/path/to/csv/files')
            Missing files: nodes.csv
            All required files are present.
            # This will process 'damage_states.json', update it, and use CSV
              files in '/path/to/csv/files'

            >>> system_state('damage_states.json', '/path/to/nonexistent/dir')
            Missing files: edges.csv, nodes.csv
            # This will attempt to process 'damage_states.json' and handle
              missing files in a non-existent directory
        """

        def files_exist(directory, filenames):
            # Convert directory path to a Path object
            dir_path = Path(directory)

            # Get a set of files in the directory
            files_in_directory = {
                f.name for f in dir_path.iterdir() if f.is_file()}

            # Check if each file exists in the directory
            missing_files = [filename for filename in filenames if
                             filename not in files_in_directory]

            if missing_files:
                print(f"Missing files: {', '.join(missing_files)}")  # noqa: T201
                out = False
            else:
                print("All required files are present.")  # noqa: T201
                out = True

            return out

        # Read damage states for det file and determine element capacity ratios
        # 1 denotes fully open and 0 denotes fully closed:
        capacity_dict = {}
        with Path.open(initial_state, encoding="utf-8") as file:
            temp = json.load(file)
            data = temp['TransportationNetwork']
            for asset_type in self.assets:
                datadict = data[asset_type]
                for key in datadict:
                    item_id = datadict[key]['GeneralInformation'][
                        self.no_identifier[asset_type]]
                    damage_state = int(datadict[key]['R2Dres']
                                       ['R2Dres_MostLikelyCriticalDamageState'
                                        ])
                    capacity_ratio = self.capacity_map[damage_state]
                    capacity_dict[item_id] = capacity_ratio
                    datadict[key]['GeneralInformation']['Open'] = \
                        capacity_ratio

        # Update det file with closure information:
        temp = initial_state.split('.')
        detfile_updated = temp[0] + '_updated.' + temp[1]

        with Path.open(detfile_updated, 'w', encoding="utf-8") as file:
            json.dump(data, file, indent=2)

        # Create link closures for network simulations:
        fexists = files_exist(self.csv_files['network_edges'])

        if fexists:
            graph_edge_file = csv_file_dir + '/' + self.csv_files[0]
        else:
            element_handler = TransportationElementHandler(self.assets)
            element_handler.get_graph_network(initial_state, csv_file_dir)
            graph_edge_file = element_handler['output_files'][
                'graph_network'][0]

        graph_edge_df = pd.read_csv(graph_edge_file)
        closed_edges = []
        for key in datadict:
            matches = graph_edge_df[graph_edge_df['name'] == key]

        if not matches.empty and datadict[key] == 1:
            uniqueids = matches['uniqueid'].tolist()
            closed_edges.append(uniqueids)

        # Write closed edges:
        edge_closure_file = csv_file_dir + '/' + self.csv_files[-1]
        with Path.open(edge_closure_file, 'w', encoding="utf-8") as file:
            # Write each item on a new line
            for item in closed_edges:
                file.write(item + '\n')

    @abstractmethod
    def system_performance(self, state) -> None:  # Move the CSV creation here
        def substep_assignment(nodes_df=None,
                               weighted_edges_df=None,
                               od_ss=None,
                               quarter_demand=None,
                               assigned_demand=None,
                               quarter_counts=4,
                               trip_info=None,
                               agent_time_limit=0,
                               sample_interval=1,
                               agents_path=None,
                               hour=None,
                               quarter=None,
                               ss_id=None,
                               alpha_f=0.3,
                               beta_f=3):

            open_edges_df = weighted_edges_df.loc[weighted_edges_df['fft'] <
                                                  36000]

            net = pdna.Network(nodes_df["x"], nodes_df["y"],
                               open_edges_df["start_nid"],
                               open_edges_df["end_nid"],
                               open_edges_df[["weight"]],
                               twoway=False)

            print('network')  # noqa: T201
            net.set(pd.Series(net.node_ids))
            print('net')  # noqa: T201

            nodes_origin = od_ss['origin_nid'].to_numpy
            nodes_destin = od_ss['destin_nid'].to_numpy
            nodes_current = od_ss['current_nid'].to_numpy
            agent_ids = od_ss['agent_id'].to_numpy()
            agent_current_links = od_ss['current_link'].to_numpy()
            agent_current_link_times = od_ss['current_link_time'].to_numpy()
            paths = net.shortest_paths(nodes_current, nodes_destin)

            # check agent time limit
            path_lengths = net.shortest_path_lengths(
                nodes_current, nodes_destin)
            remove_agent_list = []
            if agent_time_limit is None:
                pass
            else:
                for agent_idx, agent_id in enumerate(agent_ids):
                    planned_trip_length = path_lengths[agent_idx]
                    # agent_time_limit[agent_id]
                    trip_length_limit = agent_time_limit
                    if planned_trip_length > trip_length_limit+0:
                        remove_agent_list.append(agent_id)

            edge_travel_time_dict = weighted_edges_df['t_avg'].T.to_dict()
            edge_current_vehicles = weighted_edges_df['veh_current'].T.to_dict(
            )
            edge_quarter_vol = weighted_edges_df['vol_true'].T.to_dict()
            # edge_length_dict = weighted_edges_df['length'].T.to_dict()
            od_residual_ss_list = []
            # all_paths = []
            path_i = 0
            for path in paths:
                trip_origin = nodes_origin[path_i]
                trip_destin = nodes_destin[path_i]
                agent_id = agent_ids[path_i]
                # remove some agent (path too long)
                if agent_id in remove_agent_list:
                    path_i += 1
                    # no need to update trip info
                    continue
                remaining_time = 3600/quarter_counts + \
                    agent_current_link_times[path_i]
                used_time = 0
                for edge_s, edge_e in zip(path, path[1:]):
                    edge_str = "{edge_s}-{edge_e}"
                    edge_travel_time = edge_travel_time_dict[edge_str]

                    if (remaining_time > edge_travel_time) and \
                       (edge_travel_time < 36000):
                        # all_paths.append(edge_str)
                        # p_dist += edge_travel_time
                        remaining_time -= edge_travel_time
                        used_time += edge_travel_time
                        edge_quarter_vol[edge_str] += (1 * sample_interval)
                        trip_stop = edge_e

                        if edge_str == agent_current_links[path_i]:
                            edge_current_vehicles[edge_str] -= (
                                1 * sample_interval)
                    else:
                        if edge_str != agent_current_links[path_i]:
                            edge_current_vehicles[edge_str] += (
                                1 * sample_interval)
                        new_current_link = edge_str
                        new_current_link_time = remaining_time
                        trip_stop = edge_s
                        od_residual_ss_list.append(
                            [agent_id,
                             trip_origin,
                             trip_destin,
                             trip_stop,
                             new_current_link,
                             new_current_link_time])
                        break
                trip_info[(agent_id, trip_origin, trip_destin)
                          ][0] += 3600/quarter_counts
                trip_info[(agent_id, trip_origin, trip_destin)][1] += used_time
                trip_info[(agent_id, trip_origin, trip_destin)][2] = trip_stop
                trip_info[(agent_id, trip_origin, trip_destin)][3] = hour
                trip_info[(agent_id, trip_origin, trip_destin)][4] = quarter
                trip_info[(agent_id, trip_origin, trip_destin)][5] = ss_id
                path_i += 1

            new_edges_df = weighted_edges_df[['uniqueid',
                                              'u',
                                              'v',
                                              'start_nid',
                                              'end_nid',
                                              'fft',
                                              'capacity',
                                              'normal_fft',
                                              'normal_capacity',
                                              'length',
                                              'vol_true',
                                              'vol_tot',
                                              'veh_current',
                                              'geometry']].copy()
            # new_edges_df = new_edges_df.join(edge_volume, how='left')
            # new_edges_df['vol_ss'] = new_edges_df['vol_ss'].fillna(0)
            # new_edges_df['vol_true'] += new_edges_df['vol_ss']
            new_edges_df['vol_true'] = new_edges_df.index.map(edge_quarter_vol)
            new_edges_df['veh_current'] = new_edges_df.index.map(
                edge_current_vehicles)
            # new_edges_df['vol_tot'] += new_edges_df['vol_ss']
            new_edges_df['flow'] = (
                new_edges_df['vol_true']*quarter_demand/assigned_demand) *\
                quarter_counts
            new_edges_df['t_avg'] = new_edges_df['fft'] * \
                (1 + alpha_f *
                 (new_edges_df['flow']/new_edges_df['capacity'])**beta_f)
            new_edges_df['t_avg'] = np.where(
                new_edges_df['t_avg'] > 36000, 36000, new_edges_df['t_avg'])
            new_edges_df['t_avg'] = new_edges_df['t_avg'].round(2)

            return new_edges_df, od_residual_ss_list, trip_info, agents_path

        def write_edge_vol(edges_df=None,
                           simulation_outputs=None,
                           quarter=None,
                           hour=None,
                           scen_nm=None):
            if 'flow' in edges_df.columns:
                if edges_df.shape[0] < 10:
                    edges_df[['uniqueid',
                              'start_nid',
                              'end_nid',
                              'capacity',
                              'veh_current',
                              'vol_true',
                              'vol_tot',
                              'flow',
                              't_avg',
                              'geometry']].to_csv(
                        f'{simulation_outputs}/edge_vol/edge_vol_hr{hour}_'
                        f'qt{quarter}_{scen_nm}.csv',
                        index=False
                    )

                else:
                    edges_df.loc[edges_df['vol_true'] > 0, [
                        'uniqueid',
                        'start_nid',
                        'end_nid',
                        'capacity',
                        'veh_current',
                        'vol_true',
                        'vol_tot',
                        'flow',
                        't_avg',
                        'geometry']
                    ].to_csv(
                        f'{simulation_outputs}/edge_vol/edge_vol_hr{hour}_'
                        f'qt{quarter}_{scen_nm}.csv',
                        index=False
                    )

        def write_final_vol(edges_df=None,
                            simulation_outputs=None,
                            quarter=None,
                            hour=None,
                            scen_nm=None):
            edges_df.loc[edges_df['vol_tot'] > 0, [
                'uniqueid',
                'start_nid',
                'end_nid',
                'vol_tot',
                'geometry']
            ].to_csv(
                f'{simulation_outputs}/edge_vol/final_edge_vol_hr{hour}_qt'
                f'{quarter}_{scen_nm}.csv',
                index=False
            )

        def assignment(quarter_counts=6,
                       substep_counts=15,
                       substep_size=30000,
                       edges_df=None,
                       nodes_df=None,
                       od_all=None,
                       simulation_outputs=None,
                       scen_nm=None,
                       hour_list=None,
                       quarter_list=None,
                       cost_factor=None,
                       closure_hours=None,
                       closed_links=None,
                       agent_time_limit=None,
                       sample_interval=1,
                       agents_path=None,
                       alpha_f=0.3,
                       beta_f=4):
            if closure_hours is None:
                closure_hours = []

            od_all['current_nid'] = od_all['origin_nid']
            trip_info = {(od.agent_id,
                          od.origin_nid,
                          od.destin_nid): [0,
                                           0,
                                           od.origin_nid,
                                           0,
                                           od.hour,
                                           od.quarter,
                                           0,
                                           0] for od in
                         od_all.itertuples()}

            # Quarters and substeps
            # probability of being in each division of hour
            if quarter_list is None:
                quarter_counts = 4
            else:
                quarter_counts = len(quarter_list)
            quarter_ps = [1/quarter_counts for i in range(quarter_counts)]
            quarter_ids = list(range(quarter_counts))

            # initial setup
            od_residual_list = []
            # accumulator
            edges_df['vol_tot'] = 0
            edges_df['veh_current'] = 0

            # Loop through days and hours
            for _day in ['na']:
                for hour in hour_list:
                    gc.collect()
                    if hour in closure_hours:
                        for row in closed_links.itertuples():
                            edges_df.loc[(edges_df['u'] == row.u)
                                         & (edges_df['v'] == row.v
                                            ), 'capacity'] = 1
                            edges_df.loc[(edges_df['u'] == row.u) &
                                         (edges_df['v'] == row.v
                                          ), 'fft'] = 36000
                    else:
                        edges_df['capacity'] = edges_df['normal_capacity']
                        edges_df['fft'] = edges_df['normal_fft']

                    # Read OD
                    od_hour = od_all[od_all['hour'] == hour].copy()
                    if od_hour.shape[0] == 0:
                        od_hour = pd.DataFrame([], columns=od_all.columns)
                    od_hour['current_link'] = None
                    od_hour['current_link_time'] = 0

                    # Divide into quarters
                    if 'quarter' in od_all.columns:
                        pass
                    else:
                        od_quarter_msk = np.random.choice(
                            quarter_ids, size=od_hour.shape[0], p=quarter_ps)
                        od_hour['quarter'] = od_quarter_msk

                    if quarter_list is None:
                        quarter_list = quarter_ids
                    for quarter in quarter_list:
                        # New OD in assignment period
                        od_quarter = od_hour.loc[od_hour['quarter'] == quarter,
                                                 ['agent_id',
                                                  'origin_nid',
                                                  'destin_nid',
                                                  'current_nid',
                                                  'current_link',
                                                  'current_link_time']]
                        # Add resudal OD
                        od_residual = pd.DataFrame(od_residual_list, columns=[
                                                   'agent_id',
                                                   'origin_nid',
                                                   'destin_nid',
                                                   'current_nid',
                                                   'current_link',
                                                   'current_link_time'])
                        od_residual['quarter'] = quarter
                        # Total OD in each assignment period is the combined
                        # of new and residual OD:
                        od_quarter = pd.concat(
                            [od_quarter, od_residual],
                            sort=False,
                            ignore_index=True)
                        # Residual OD is no longer residual after it has been
                        # merged to the quarterly OD:
                        od_residual_list = []
                        od_quarter = od_quarter[od_quarter['current_nid']
                                                != od_quarter['destin_nid']]

                        # total demand for this quarter, including total and
                        # residual demand:
                        quarter_demand = od_quarter.shape[0]
                        # how many among the OD pairs to be assigned in this
                        # quarter are actually residual from previous quarters
                        residual_demand = od_residual.shape[0]
                        assigned_demand = 0

                        substep_counts = max(
                            1, (quarter_demand // substep_size) + 1)
                        logging.info(f'HR {hour} QT {quarter} has '
                                     f'{quarter_demand}/{residual_demand} od'
                                     f's/residuals {substep_counts} substeps')
                        substep_ps = [
                            1/substep_counts for i in range(substep_counts)]
                        substep_ids = list(range(substep_counts))
                        od_substep_msk = np.random.choice(
                            substep_ids, size=quarter_demand, p=substep_ps)
                        od_quarter['ss_id'] = od_substep_msk

                        # reset volume at each quarter
                        edges_df['vol_true'] = 0

                        for ss_id in substep_ids:
                            gc.collect()

                            time_ss_0 = time.time()
                            logging.info(f'Hour: {hour}, Quarter: {quarter}, '
                                         'SS ID: {ss_id}')
                            od_ss = od_quarter[od_quarter['ss_id'] == ss_id]
                            assigned_demand += od_ss.shape[0]
                            if assigned_demand == 0:
                                continue
                            # calculate weight
                            weighted_edges_df = edges_df.copy()
                            # weight by travel distance
                            # weighted_edges_df['weight'] = edges_df['length']
                            # weight by travel time
                            # weighted_edges_df['weight'] = edges_df['t_avg']
                            # weight by travel time
                            # weighted_edges_df['weight'] = (edges_df['t_avg']
                            # - edges_df['fft']) * 0.5 + edges_df['length']*0.1
                            # + cost_factor*edges_df['length']*0.1*(
                            # edges_df['is_highway'])
                            # 10 yen per 100 m --> 0.1 yen per m
                            weighted_edges_df['weight'] = edges_df['t_avg']
                            # weighted_edges_df['weight'] = np.where(
                            # weighted_edges_df['weight']<0.1, 0.1,
                            # weighted_edges_df['weight'])

                            # traffic assignment with truncated path
                            (edges_df,
                             od_residual_ss_list,
                             trip_info,
                             agents_path) = \
                                substep_assignment(nodes_df=nodes_df,
                                                   weighted_edges_df=weighted_edges_df,
                                                   od_ss=od_ss,
                                                   quarter_demand=quarter_demand,
                                                   assigned_demand=assigned_demand,
                                                   quarter_counts=quarter_counts,
                                                   trip_info=trip_info,
                                                   agent_time_limit=agent_time_limit,
                                                   sample_interval=sample_interval,
                                                   agents_path=agents_path,
                                                   hour=hour,
                                                   quarter=quarter,
                                                   ss_id=ss_id,
                                                   alpha_f=alpha_f,
                                                   beta_f=beta_f)

                            od_residual_list += od_residual_ss_list
                            # write_edge_vol(edges_df=edges_df,
                            #            simulation_outputs=simulation_outputs,
                            #               quarter=quarter,
                            #               hour=hour,
                            #         scen_nm='ss{}_{}'.format(ss_id, scen_nm))
                            logging.info(f'HR {hour} QT {quarter} SS {ss_id}'
                                         ' finished, max vol '
                                         f'{np.max(edges_df["vol_true"])}, '
                                         f'time {time.time() - time_ss_0}')

                        # write quarterly results
                        edges_df['vol_tot'] += edges_df['vol_true']
                        if True:  # hour >=16 or (hour==15 and quarter==3):
                            write_edge_vol(edges_df=edges_df,
                                           simulation_outputs=simulation_outputs,
                                           quarter=quarter,
                                           hour=hour,
                                           scen_nm=scen_nm)

                    if hour % 3 == 0:
                        trip_info_df = pd.DataFrame([[trip_key[0],
                                                      trip_key[1],
                                                      trip_key[2],
                                                      trip_value[0],
                                                      trip_value[1],
                                                      trip_value[2],
                                                      trip_value[3],
                                                      trip_value[4],
                                                      trip_value[5]] for
                                                     trip_key, trip_value in
                                                     trip_info.items()],
                                                    columns=['agent_id',
                                                             'origin_nid',
                                                             'destin_nid',
                                                             'travel_time',
                                                             'travel_time_used',
                                                             'stop_nid',
                                                             'stop_hour',
                                                             'stop_quarter',
                                                             'stop_ssid'])
                        trip_info_df.to_csv(simulation_outputs + '/trip_info'
                                            f'/trip_info_{scen_nm}_hr{hour}'
                                            '.csv', index=False)

            # output individual trip travel time and stop location

            trip_info_df = pd.DataFrame([[trip_key[0],
                                          trip_key[1],
                                          trip_key[2],
                                          trip_value[0],
                                          trip_value[1],
                                          trip_value[2],
                                          trip_value[3],
                                          trip_value[4],
                                          trip_value[5]] for trip_key,
                                         trip_value in trip_info.items()],
                                        columns=['agent_id',
                                                 'origin_nid',
                                                 'destin_nid',
                                                 'travel_time',
                                                 'travel_time_used',
                                                 'stop_nid',
                                                 'stop_hour',
                                                 'stop_quarter',
                                                 'stop_ssid'])
            trip_info_df.to_csv(simulation_outputs +
                                f'/trip_info/trip_info_{scen_nm}.csv',
                                index=False)

            write_final_vol(edges_df=edges_df,
                            simulation_outputs=simulation_outputs,
                            quarter=quarter,
                            hour=hour,
                            scen_nm=scen_nm)

        network_edges = self.csv_filenames[0]
        network_nodes = self.csv_filenames[1]
        closed_edges_file = self.csv_filenames[2]
        demand_file = self.csv_filenames[3]
        simulation_outputs = 'simulation_outputs'
        scen_nm = 'simulation_out'

        hour_list = list(range(6, 9))
        quarter_list = [0, 1, 2, 3, 4, 5]
        closure_hours = []

        edges_df = pd.read_csv(network_edges)
        edges_df = edges_df[["uniqueid", "geometry", "osmid", "length", "type",
                             "lanes", "maxspeed", "fft", "capacity",
                             "start_nid", "end_nid"]]
        edges_df = gpd.GeoDataFrame(
            edges_df, crs='epsg:4326', geometry=edges_df['geometry'].map(
                loads))
        edges_df = edges_df.sort_values(by='fft', ascending=False).\
            drop_duplicates(subset=['start_nid', 'end_nid'], keep='first')
        # pay attention to the unit conversion
        edges_df['fft'] = edges_df['length']/edges_df['maxspeed']*2.23694
        edges_df['edge_str'] = edges_df['start_nid'].astype(
            'str') + '-' + edges_df['end_nid'].astype('str')
        edges_df['capacity'] = np.where(
            edges_df['capacity'] < 1, 950, edges_df['capacity'])
        edges_df['normal_capacity'] = edges_df['capacity']
        edges_df['normal_fft'] = edges_df['fft']
        edges_df['t_avg'] = edges_df['fft']
        edges_df['u'] = edges_df['start_nid']
        edges_df['v'] = edges_df['end_nid']
        edges_df = edges_df.set_index('edge_str')
        # closure locations
        closed_links = pd.read_csv(closed_edges_file)
        for row in closed_links.itertuples():
            edges_df.loc[(edges_df['uniqueid'] == row.uniqueid),
                         'capacity'] = 1
            edges_df.loc[(edges_df['uniqueid'] == row.uniqueid), 'fft'] = 36000
        # output closed file for visualization
        edges_df.loc[edges_df['fft'] == 36000, ['uniqueid',
                                                'start_nid',
                                                'end_nid',
                                                'capacity',
                                                'fft',
                                                'geometry']].to_csv(
                                                    simulation_outputs +
                                                    '/closed_links_'
                                                    f'{scen_nm}.csv')

        # nodes processing
        nodes_df = pd.read_csv(network_nodes)

        nodes_df['x'] = nodes_df['lon']
        nodes_df['y'] = nodes_df['lat']
        nodes_df = nodes_df.set_index('node_id')

        # demand processing
        t_od_0 = time.time()
        od_all = pd.read_csv(demand_file)
        t_od_1 = time.time()
        logging.info('%d sec to read %d OD pairs',
                     t_od_1 - t_od_0, od_all.shape[0])

        # run residual_demand_assignment
        assignment(edges_df=edges_df,
                   nodes_df=nodes_df,
                   od_all=od_all,
                   simulation_outputs=simulation_outputs,
                   scen_nm=scen_nm,
                   hour_list=hour_list,
                   quarter_list=quarter_list,
                   closure_hours=closure_hours,
                   closed_links=closed_links)

    @abstractmethod
    def update_od_file(self,
                       old_nodes: str,
                       old_det: str,
                       new_nodes: str,
                       new_det: str,
                       od: str,
                       origin_ids: list[int]
                       ) -> pd.DataFrame:
        """
        Update origin-destination (OD) file from changes in population data.

        This function updates an OD file by calculating the population changes
        at each node between two time steps and generates trips originating
        from specified origins and ending at nodes where the population has
        increased. The updated OD data is saved to a new file.

        Args__
            old_nodes (str): Path to the CSV file containing the node
                             information at the previous time step.
            old_det (str): Path to the JSON file containing the building
                            information at the previous time step.
            new_nodes (str): Path to the CSV file containing the node
                             information at the current time step.
            new_det (str): Path to the JSON file containing the building
                            information at the current time step.
            od (str): Path to the existing OD file to be updated.
            origin_ids (List[int]): List of IDs representing possible origins
                                    for generating trips.

        Returns__
            pd.DataFrame: The updated OD DataFrame with new trips based on
                            population changes.

        Raises__
            FileNotFoundError: If any of the provided file paths are incorrect
                                or the files do not exist.
            json.JSONDecodeError: If the JSON files cannot be read or parsed
                                    correctly.
            KeyError: If expected keys are missing in the JSON data.
            ValueError: If there are issues with data conversions or
                        calculations.

        Examples__
            >>> update_od_file(
                    'old_nodes.csv',
                    'old_building_data.json',
                    'new_nodes.csv',
                    'new_building_data.json',
                    'existing_od.csv',
                    [1, 2, 3, 4]
                )
            The function will process the provided files, calculate population
            changes, and update the OD file with new trips. The result will be
            saved to 'updated_od.csv'.

        Notes__
            - Ensure that the columns `lat`, `lon`, and `node_id` are present
                in the nodes CSV files.
            - The `det` files should contain the `Buildings` and
                `GeneralInformation` sections with appropriate keys.
            - The OD file should have the columns `agent_id`, `origin_nid`,
                `destin_nid`, `hour`, and `quarter`.
        """
        # Extract the building information from the det file and convert it to
        # a pandas dataframe
        def extract_building_from_det(det):
            # Open the det file
            with Path.open(det, encoding="utf-8") as file:
                # Return the JSON object as a dictionary
                json_data = json.load(file)

            # Extract the required information and convert it to a pandas
            # dataframe
            extracted_data = []

            for aim_id, info in json_data['Buildings']['Building'].items():
                general_info = info.get('GeneralInformation', {})
                extracted_data.append({
                    'AIM_id': aim_id,
                    'Latitude': general_info.get('Latitude'),
                    'Longitude': general_info.get('Longitude'),
                    'Population': general_info.get('Population')
                })

            return pd.DataFrame(extracted_data)

        # Aggregate the population in buildings to the closest road network
        # node
        def closest_neighbour(building_df, nodes_df):
            # Find the nearest road network node to each building
            nodes_xy = np.array([nodes_df['lat'].to_numpy(),
                                 nodes_df['lon'].to_numpy()]).transpose()
            building_df['closest_node'] = building_df.apply(lambda x: cdist(
                [(x['Latitude'], x['Longitude'])], nodes_xy).argmin(), axis=1)

            # Merge the road network and building dataframes
            merged_df = nodes_df.merge(building_df,
                                       left_on='node_id',
                                       right_on='closest_node',
                                       how='left')
            merged_df = merged_df.drop(
                columns=['AIM_id', 'Latitude', 'Longitude', 'closest_node'])
            merged_df = merged_df.fillna(0)

            # Aggregate population of  neareast buildings to the road network
            # node
            updated_nodes_df = merged_df.groupby('node_id').agg(
                {'lon': 'first', 'lat': 'first', 'geometry': 'first',
                 'Population': 'sum'}).reset_index()

            return updated_nodes_df  # noqa: RET504

        # Function to add the population information to the nodes file
        def update_nodes_file(nodes, det):
            # Read the nodes file
            nodes_df = pd.read_csv(nodes)
            # Extract the building information from the det file and convert it
            # to a pandas dataframe
            building_df = extract_building_from_det(det)
            # Aggregate the population in buildings to the closest road network
            # node
            updated_nodes_df = closest_neighbour(building_df, nodes_df)

            return updated_nodes_df  # noqa: RET504

        # Read the od file
        od_df = pd.read_csv(od)
        # Add the population information to nodes dataframes at the last and
        # current time steps
        old_nodes_df = update_nodes_file(old_nodes, old_det)
        new_nodes_df = update_nodes_file(new_nodes, new_det)
        # Calculate the population changes at each node (assuming that
        # population will only increase at each node)
        population_change_df = old_nodes_df.copy()
        population_change_df['Population Change'] = new_nodes_df['Population']\
            - old_nodes_df['Population']
        population_change_df['Population Change'].astype(int)
        # Randomly generate the trips that start at one of the connections
        # between Alameda Island and Oakland, and end at the nodes where
        # population increases and append them to the od file
        # Generate OD data for each node with increased population and append
        # it to the od file
        for _, row in population_change_df.iterrows():
            # Only process rows with positive population difference
            if row['Population_Difference'] > 0:
                for _ in range(row['Population_Difference']):
                    # Generate random origin_nid
                    origin_nid = np.random.choice(origin_ids)
                    # Set the destin_nid to the node_id of the increased
                    # population
                    destin_nid = row['node_id']
                    # Generate random hour and quarter
                    hour = np.random.randint(5, 24)
                    quarter = np.random.randint(0, 5)
                    # Append to od dataframe
                    od_df = od_df.append({
                        'agent_id': 0,
                        'origin_nid': origin_nid,
                        'destin_nid': destin_nid,
                        'hour': hour,
                        'quarter': quarter
                    }, ignore_index=True)
        od_df.to_csv('updated_od.csv')
        return od_df
