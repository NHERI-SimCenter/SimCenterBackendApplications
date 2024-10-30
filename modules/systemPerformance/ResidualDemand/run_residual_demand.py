#  # noqa: INP001, D100
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
#
# This file is part of pelicun.
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
# pelicun. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Jinyan Zhao

import argparse
import json
import logging
import os
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.animation import FuncAnimation
from shapely import wkt
from shapely.wkt import loads
from transportation import TransportationPerformance


def select_realizations_to_run(damage_input, run_dir):
    """
    Select the realizations to run based on the damage input and available realizations.

    Parameters
    ----------
    damage_input : dict
        Dictionary containing the damage input parameters.
    run_dir : str
        Directory where the results are stored.

    Returns
    -------
    list
        List of realizations to run.
    """
    # Get the available realizations
    results_files = [f for f in os.listdir(run_dir) if f.startswith('Results_') and f.endswith('.json')]
    rlzs_available = sorted([
        int(file.split('_')[1].split('.')[0])
        for file in results_files
        if file.split('_')[1].split('.')[0].isnumeric()
    ])
    # Get the number of realizations
    if damage_input['Type'] == 'SpecificRealization':
        rlz_filter = damage_input['Parameters']['Filter']
        rlzs_requested = []
        for rlzs in rlz_filter.split(','):
            if '-' in rlzs:
                rlzs_low, rlzs_high = rlzs.split('-')
                rlzs_requested += list(range(int(rlzs_low), int(rlzs_high) + 1))
            else:
                rlzs_requested.append(int(rlzs))
        rlzs_requested = np.array(rlzs_requested)
        rlzs_in_available = np.in1d(rlzs_requested, rlzs_available)  # noqa: NPY201
        if rlzs_in_available.sum() != 0:
            rlzs_to_run = rlzs_requested[np.where(rlzs_in_available)[0]]
        else:
            rlzs_to_run = []
    if damage_input['Type'] == 'SampleFromRealizations':
        sample_size = damage_input['Parameters']['SampleSize']
        seed = damage_input['Parameters']['SampleSeed']
        if sample_size <= len(rlzs_available):
            np.random.seed(seed)
            rlzs_to_run = np.sort(
                np.random.choice(rlzs_available, sample_size, replace=False)
            ).tolist()
        else:
            msg = 'Sample size is larger than the number of available realizations'
            raise ValueError(msg)
    return rlzs_to_run

def create_congestion_animation(edge_vol_dir, output_file_name):
    """
    Create an animation of congestion over time.

    Parameters
    ----------
    edge_vol_dir : str
        Directory containing edge volume CSV files.
    output_file_name : str
        Name of the output animation file.
    """
    all_frames = sorted(os.listdir(edge_vol_dir))
    all_frames = [f for f in all_frames if f.startswith('edge_vol_') and f.endswith('.csv')]
    times = []
    for ii in range(len(all_frames)):
        hour = all_frames[ii][11:12]
        minute = int(all_frames[ii][15:16]) * 10

        minute_end = minute + 10

        if minute_end == 60:  # noqa: PLR2004
            minute_end = 0
            hour_end = str(int(hour)+1)
        else:
            hour_end = hour
        time_of_day = 'Between ' + hour + ':' + str(minute).zfill(2) + ' and ' + str(int(hour_end)) + ':' + str(minute_end).zfill(2)
        times.append(time_of_day)
    figure, ax = plt.subplots(figsize=(10, 10))
    animation = FuncAnimation(figure,
                          func = animation_function,
                          frames = np.arange(0, len(all_frames), 1),
                          fargs=(ax, edge_vol_dir, all_frames, times),
                          interval = 10)
    animation.save(output_file_name, writer='imagemagick', fps=2)

def get_highest_congestion(edge_vol_dir, edges_csv):
    """
    Calculate the highest congestion for each edge over time.

    Parameters
    ----------
    edge_vol_dir : str
        Directory containing edge volume CSV files.
    edges_csv : str
        Path to the CSV file containing edge information.

    Returns
    -------
    DataFrame
        DataFrame containing the highest congestion for each edge.
    """
    all_frames = sorted(os.listdir(edge_vol_dir))
    all_frames = [f for f in all_frames if f.startswith('edge_vol_') and f.endswith('.csv')]
    all_edges = pd.read_csv(edges_csv)
    congestion = np.zeros(len(all_edges))
    for ii in range(len(all_frames)):
        edge_vol = pd.read_csv(edge_vol_dir / all_frames[ii])[['uniqueid', 'vol_tot', 'capacity']]
        all_edges_vol = all_edges.merge(edge_vol, left_on='uniqueid', right_on='uniqueid', how='left')
        congestion_i = (all_edges_vol['vol_tot']/all_edges_vol['capacity_x']).fillna(0)
        congestion = np.maximum(congestion, congestion_i)
    congestion = congestion.to_frame(name='congestion')
    all_edges = all_edges.merge(congestion, left_index=True, right_index=True)
    return all_edges[['id', 'congestion']].groupby('id').max()

def animation_function(ii, ax, results_dir, all_frames, times):
    """
    Update the plot for each frame in the animation.

    Parameters
    ----------
    ii : int
        The current frame index.
    ax : matplotlib.axes.Axes
        The axes to plot on.
    results_dir : str
        Directory containing the results CSV files.
    all_frames : list
        List of all frame filenames.
    times : list
        List of time labels for each frame.
    """
    ax.clear()

    results_df = pd.read_csv(results_dir / all_frames[ii])
    results_df['geometry'] = results_df['geometry'].apply(wkt.loads)

    color_array = np.array(results_df.vol_tot / results_df.capacity)
    color_array = [min(1, x) for x in color_array]

    results_df['color'] = [[1, 0, 0]] * len(results_df)

    results_df['color'] = results_df['color'].apply(np.array) * color_array

    results_df['s'] = np.sqrt(results_df.vol_tot / results_df.capacity)

    gdf = gpd.GeoDataFrame(results_df, geometry='geometry', crs='EPSG:4326')
    # gdf = gdf.to_crs(epsg=3857)

    gdf.plot(ax=ax, linewidth=gdf.s, color=gdf.color)
    ax.set_title(times[ii])
    minx, miny, maxx, maxy = gdf.total_bounds
    ax.set_xlim(minx*0.95, maxx*1.05)
    ax.set_ylim(miny*0.95, maxy*1.05)
    # Calculate width and height of the bounding box
    width = maxx - minx
    height = maxy - miny
    # Determine the larger of the two dimensions
    max_dim = max(width, height)
    # Adjust the bounds to make a square
    if width > height:
        # Expand height to match width
        mid_y = (miny + maxy) / 2
        miny = mid_y - max_dim / 2
        maxy = mid_y + max_dim / 2
    else:
        # Expand width to match height
        mid_x = (minx + maxx) / 2
        minx = mid_x - max_dim / 2
        maxx = mid_x + max_dim / 2
    # Set the new square extent
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    cx.add_basemap(ax, crs=gdf.crs, source=cx.providers.CartoDB.Positron, zoom='auto')

def create_delay_agg(od_file_pre, od_file_post):
    """
    Create initial delay aggregation data structures.

    Parameters
    ----------
    od_file_pre : str
        Path to the CSV file containing pre-event origin-destination data.
    od_file_post : str
        Path to the CSV file containing post-event origin-destination data.

    Returns
    -------
    tuple
        A tuple containing two dictionaries: undamaged_time and damaged_time.
    """
    od_df_pre = pd.read_csv(od_file_pre)
    od_df_pre_agent_id = od_df_pre['agent_id'].tolist()
    od_df_pre_data = np.zeros((len(od_df_pre), 0))
    undamaged_time = {'agent_id': od_df_pre_agent_id, 'data': od_df_pre_data}
    od_df_post = pd.read_csv(od_file_post)
    od_df_post_agent_id = od_df_post['agent_id'].tolist()
    od_df_post_data = np.zeros((len(od_df_post), 0))
    damaged_time = {'agent_id': od_df_post_agent_id, 'data': od_df_post_data}
    return undamaged_time, damaged_time

def append_to_delay_agg(undamaged_time, damaged_time, trip_info_file):
    """
    Append new delay data to the existing aggregation.

    Parameters
    ----------
    undamaged_time : dict
        Dictionary containing undamaged travel times.
    damaged_time : dict
        Dictionary containing damaged travel times.
    trip_info_file : str
        Path to the CSV file containing trip information.

    Returns
    -------
    tuple
        Updated undamaged_time and damaged_time dictionaries.
    """
    trip_info = pd.read_csv(trip_info_file).set_index('agent_id')
    undamaged_time_new = trip_info.loc[undamaged_time['agent_id'], 'travel_time_used_undamaged'].to_numpy()
    undamaged_time_new = undamaged_time_new.reshape((len(undamaged_time_new), 1))
    undamaged_time['data'] = np.append(undamaged_time['data'], undamaged_time_new, axis=1)
    damaged_time_new = trip_info.loc[damaged_time['agent_id'], 'travel_time_used_damaged'].to_numpy()
    damaged_time_new = damaged_time_new.reshape((len(damaged_time_new), 1))
    damaged_time['data'] = np.append(damaged_time['data'], damaged_time_new, axis=1)
    return undamaged_time, damaged_time

def aggregate_delay_results(undamaged_time, damaged_time,
                            od_file_pre, od_file_post):
    """
    Aggregate delay results and save to a CSV file.

    Parameters
    ----------
    undamaged_time : dict
        Dictionary containing undamaged travel times.
    damaged_time : dict
        Dictionary containing damaged travel times.
    od_file_pre : str
        Path to the CSV file containing pre-event origin-destination data.
    od_file_post : str
        Path to the CSV file containing post-event origin-destination data.
    """
    od_df_pre = pd.read_csv(od_file_pre)
    od_df_post = pd.read_csv(od_file_post)[['agent_id']]
    compare_df = od_df_pre.merge(od_df_post, on='agent_id', how='outer')
    compare_df = compare_df.set_index('agent_id')

    compare_df['mean_time_used_undamaged'] = np.nan
    compare_df['std_time_used_undamaged'] = np.nan
    compare_df['mean_time_used_damaged'] = np.nan
    compare_df['std_time_used_damaged'] = np.nan
    compare_df.loc[undamaged_time['agent_id'], 'mean_time_used_undamaged'] = \
        undamaged_time['data'].mean(axis=1)
    compare_df.loc[undamaged_time['agent_id'], 'std_time_used_undamaged'] = \
        undamaged_time['data'].std(axis=1)
    compare_df.loc[damaged_time['agent_id'], 'mean_time_used_damaged'] = \
        damaged_time['data'].mean(axis=1)
    compare_df.loc[damaged_time['agent_id'], 'std_time_used_damaged'] = \
        damaged_time['data'].std(axis=1)

    inner_agents = od_df_pre.merge(od_df_post, on='agent_id', how='inner')['agent_id'].tolist()
    indices_in_undamaged = [undamaged_time['agent_id'].index(value) for value in inner_agents if value in undamaged_time['agent_id']]
    indices_in_damaged = [damaged_time['agent_id'].index(value) for value in inner_agents if value in damaged_time['agent_id']]
    delay_duration = damaged_time['data'][indices_in_damaged,:] - \
        undamaged_time['data'][indices_in_undamaged,:]
    delay_ratio = delay_duration/undamaged_time['data'][indices_in_undamaged,:]
    delay_df = pd.DataFrame(data={
        'agent_id': inner_agents,
        'mean_delay_duration': delay_duration.mean(axis = 1),
        'mean_delay_ratio': delay_ratio.mean(axis = 1),
        'std_delay_duration': delay_duration.std(axis = 1),
        'std_delay_ratio': delay_ratio.std(axis = 1)
    })

    compare_df = compare_df.merge(delay_df, on='agent_id', how='left')
    compare_df.to_csv('travel_delay_stats.csv', index=False)

def compile_r2d_results_geojson(residual_demand_dir, results_det_file):
    """
    Compile the R2D results into a GeoJSON file for visualization.

    Parameters
    ----------
    residual_demand_dir : str
        Directory where the residual demand results are stored.
    results_det_file : str
        Path to the JSON file containing detailed results.

    Returns
    -------
    None
    """
    with open(results_det_file, encoding='utf-8') as f:  # noqa: PTH123
        res_det = json.load(f)
    metadata = {
            'WorkflowType': 'ResidualDemandSimulation',
            'Time': datetime.now().strftime('%m-%d-%Y %H:%M:%S'),  # noqa: DTZ005
        }
    # create the geojson for R2D visualization
    geojson_result = {
        'type': 'FeatureCollection',
        'crs': {
            'type': 'name',
            'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'},
        },
        'metadata': metadata,
        'features': [],
    }
    for asset_type in res_det.keys():  # noqa: SIM118
        for assetSubtype, subtypeResult in res_det[asset_type].items():  # noqa: N806
            allAssetIds = sorted([int(x) for x in subtypeResult.keys()])  # noqa: SIM118, N806
            for asset_id in allAssetIds:
                ft = {'type': 'Feature'}
                asst_GI = subtypeResult[str(asset_id)][  # noqa: N806
                    'GeneralInformation'
                ].copy()
                asst_GI.update({'assetType': asset_type})
                try:
                    if 'geometry' in asst_GI:
                        asst_geom = shapely.wkt.loads(asst_GI['geometry'])
                        asst_geom = shapely.geometry.mapping(asst_geom)
                        asst_GI.pop('geometry')
                    elif 'Footprint' in asst_GI:
                        asst_geom = json.loads(asst_GI['Footprint'])['geometry']
                        asst_GI.pop('Footprint')
                    else:
                        # raise ValueError("No valid geometric information in GI.")
                        asst_lat = asst_GI['location']['latitude']
                        asst_lon = asst_GI['location']['longitude']
                        asst_geom = {
                            'type': 'Point',
                            'coordinates': [asst_lon, asst_lat],
                        }
                        asst_GI.pop('location')
                except:  # noqa: E722
                    warnings.warn(  # noqa: B028
                        UserWarning(
                            f'Geospatial info is missing in {assetSubtype} {asset_id}'
                        )
                    )
                    continue
                if asst_GI.get('units', None) is not None:
                    asst_GI.pop('units')
                ft.update({'geometry': asst_geom})
                ft.update({'properties': asst_GI})
                ft['properties'].update(subtypeResult[str(asset_id)]['R2Dres'])
                geojson_result['features'].append(ft)
    with open(residual_demand_dir / 'R2D_results.geojson', 'w', encoding='utf-8') as f:  # noqa: PTH123
        json.dump(geojson_result, f, indent=2)

def create_congestion_agg(edge_file):
    """
    Create initial congestion aggregation data structures.

    Parameters
    ----------
    edge_file : str
        Path to the CSV file containing edge information.

    Returns
    -------
    tuple
        A tuple containing two numpy arrays: undamaged_congestion and damaged_congestion.
    """
    all_edges = pd.read_csv(edge_file)[['id', 'uniqueid']]
    all_edges = all_edges.groupby('id').count()
    undamaged_congestion = np.zeros((len(all_edges),0))
    damaged_congestion = np.zeros((len(all_edges),0))
    return undamaged_congestion, damaged_congestion

def append_to_congestion_agg(undamaged_congestion, damaged_congestion,
                              undamaged_edge_vol_dir, damaged_edge_vol_dir,
                              edges_csv):
    """
    Append new congestion data to the existing aggregation.

    Parameters
    ----------
    undamaged_congestion : numpy.ndarray
        Array containing undamaged congestion data.
    damaged_congestion : numpy.ndarray
        Array containing damaged congestion data.
    undamaged_edge_vol_dir : str
        Directory containing undamaged edge volume CSV files.
    damaged_edge_vol_dir : str
        Directory containing damaged edge volume CSV files.
    edges_csv : str
        Path to the CSV file containing edge information.

    Returns
    -------
    tuple
        Updated undamaged_congestion and damaged_congestion arrays.
    """
    undamaged_congest = get_highest_congestion(undamaged_edge_vol_dir, edges_csv).sort_index()
    damaged_congest = get_highest_congestion(damaged_edge_vol_dir,edges_csv).sort_index()
    undamaged_congestion = np.append(undamaged_congestion,
                                      undamaged_congest[['congestion']].values,
                                      axis=1)
    damaged_congestion = np.append(damaged_congestion,
                                    damaged_congest[['congestion']].values,
                                    axis=1)
    return undamaged_congestion, damaged_congestion

def aggregate_congestions_results_to_det(undamaged_congestion, damaged_congestion,
                                            results_det_file, edges_csv):
    """
    Aggregate congestion results and update the detailed results file.

    Parameters
    ----------
    undamaged_congestion : numpy.ndarray
        Array containing undamaged congestion data.
    damaged_congestion : numpy.ndarray
        Array containing damaged congestion data.
    results_det_file : str
        Path to the JSON file containing detailed results.
    edges_csv : str
        Path to the CSV file containing edge information.

    Returns
    -------
    None
    """
    all_edges = pd.read_csv(edges_csv)[['id', 'uniqueid']]
    all_edges = all_edges.groupby('id').count().sort_index()
    congestion_increase = (damaged_congestion - undamaged_congestion)
    all_edges['mean_increase'] = congestion_increase.mean(axis=1)
    all_edges['std_increase'] = congestion_increase.std(axis=1)
    with Path(results_det_file).open() as f:
        results_det = json.load(f)
    for asset_id, asset_id_dict in results_det['TransportationNetwork']['Roadway'].items():
        asset_id_dict['R2Dres'].update(
            {'R2Dres_mean_CongestionIncrease': all_edges.loc[int(asset_id)]['mean_increase']}
                )
        asset_id_dict['R2Dres'].update(
            {'R2Dres_std_CongestionIncrease': all_edges.loc[int(asset_id)]['std_increase']}
                )
    with Path(results_det_file).open('w') as f:
        json.dump(results_det, f)
    # If TransportationNetwork_det.json exists, sync the changes
    results_folder = Path(results_det_file).parent
    transportation_det_file = results_folder/"TransportationNetwork"/ 'TransportationNetwork_det.json'
    if transportation_det_file.exists():
        transportation_det = {"TransportationNetwork": results_det['TransportationNetwork']}
        with transportation_det_file.open('w') as f:
            json.dump(transportation_det, f)

def run_on_undamaged_network(edge_file, node_file, od_file_pre, damage_det_file, config_file_dict):
    """
    Run the simulation on the undamaged network.

    Parameters
    ----------
    edge_file : str
        Path to the CSV file containing edge information.
    node_file : str
        Path to the CSV file containing node information.
    od_file_pre : str
        Path to the CSV file containing pre-event origin-destination data.
    damage_det_file : str
        Path to the JSON file containing detailed damage information.
    config_file_dict : dict
        Dictionary containing configuration parameters.

    Returns
    -------
    None
    """
    with Path(damage_det_file).open() as f:
        damage_det = json.load(f)
    assets = list(damage_det['TransportationNetwork'].keys())
    # If create animation
    create_animation = config_file_dict['CreateAnimation']

    residual_demand_simulator = TransportationPerformance(
        assets=assets,
        csv_files={'network_edges': edge_file,
                         'network_nodes': node_file,
                         'edge_closures': None,
                         'od_pairs': str(od_file_pre)},
        capacity_map=config_file_dict['CapacityMap'],
        od_file=od_file_pre,
        hour_list=config_file_dict['HourList'],

    )

    # run simulation on undamged network
    Path('trip_info').mkdir()
    Path('edge_vol').mkdir()
    residual_demand_simulator.simulation_outputs = Path.cwd()
    residual_demand_simulator.system_performance(state=None)
    if create_animation:
        create_congestion_animation(Path.cwd() /'edge_vol', Path.cwd() /'congestion.gif')

def run_one_realization(edge_file, node_file, undamaged_dir, od_file_post, damage_rlz_file,
                        damage_det_file, config_file_dict):
    """
    Run the simulation for a single realization.

    Parameters
    ----------
    edge_file : str
        Path to the CSV file containing edge information.
    node_file : str
        Path to the CSV file containing node information.
    undamaged_dir : str
        Directory containing undamaged network simulation results.
    od_file_post : str
        Path to the CSV file containing post-event origin-destination data.
    damage_rlz_file : str
        Path to the JSON file containing damage realization information.
    damage_det_file : str
        Path to the JSON file containing detailed damage information.
    config_file_dict : dict
        Dictionary containing configuration parameters.

    Returns
    -------
    bool
        True if the simulation runs successfully.
    """
    with Path(damage_rlz_file).open() as f:
        damage_rlz = json.load(f)

    assets = list(damage_rlz['TransportationNetwork'].keys())
    # If create animation
    create_animation = config_file_dict['CreateAnimation']

    # residual_demand_simulator = TransportationPerformance(
    #     assets=assets,
    #     csv_files={'network_edges': edge_file,
    #                      'network_nodes': node_file,
    #                      'edge_closures': None,
    #                      'od_pairs': str(od_file_pre)},
    #     capacity_map=config_file_dict['CapacityMap'],
    #     od_file=od_file_pre,
    #     hour_list=config_file_dict['HourList'],

    # )

    # run simulation on undamged network
    # Path('undamaged').mkdir()
    # Path(Path('undamaged')/'trip_info').mkdir()
    # Path(Path('undamaged')/'edge_vol').mkdir()
    # residual_demand_simulator.simulation_outputs = Path.cwd() / 'undamaged'
    # residual_demand_simulator.system_performance(state=None)
    # if create_animation:
    #     create_congestion_animation(Path.cwd() / 'undamaged'/'edge_vol', Path.cwd() / 'undamaged'/'congestion.gif')

    # Create residual demand simulator
    residual_demand_simulator = TransportationPerformance(
        assets=assets,
        csv_files={'network_edges': edge_file,
                         'network_nodes': node_file,
                         'edge_closures': None,
                         'od_pairs': str(od_file_post)},
        capacity_map=config_file_dict['CapacityMap'],
        od_file=od_file_post,
        hour_list=config_file_dict['HourList'],

    )
    # update the capacity due to damage
    damaged_edge_file = residual_demand_simulator.update_edge_capacity(damage_rlz_file, damage_det_file)
    residual_demand_simulator.csv_files.update({'network_edges': damaged_edge_file})
    # run simulation on damaged network
    Path('damaged').mkdir()
    Path(Path('damaged')/'trip_info').mkdir()
    Path(Path('damaged')/'edge_vol').mkdir()
    residual_demand_simulator.simulation_outputs = Path.cwd() / 'damaged'
    residual_demand_simulator.system_performance(state=None)
    if create_animation:
        create_congestion_animation(Path.cwd() / 'damaged'/'edge_vol', Path.cwd() / 'damaged'/'congestion.gif')

    # conpute the delay time of each trip
    undamaged_trip_info = pd.read_csv(undamaged_dir/'trip_info'/'trip_info_simulation_out.csv')
    damaged_trip_info = pd.read_csv(Path.cwd() / 'damaged'/'trip_info'/'trip_info_simulation_out.csv')
    # trip_info_compare = undamaged_trip_info.merge(damaged_trip_info, on='agent_id', suffixes=('_undamaged', '_damaged'))
    # trip_info_compare['delay_duration'] = trip_info_compare['travel_time_used_damaged'] - \
    #     trip_info_compare['travel_time_used_undamaged']
    # trip_info_compare['delay_ratio'] = trip_info_compare['delay_duration'] / trip_info_compare['travel_time_used_undamaged']
    # trip_info_compare.to_csv('trip_info_compare.csv', index=False)
    trip_info_compare = undamaged_trip_info.merge(damaged_trip_info, on='agent_id', suffixes=('_undamaged', '_damaged'), how='outer')
    trip_info_compare['delay_duration'] = trip_info_compare['travel_time_used_damaged'] - \
        trip_info_compare['travel_time_used_undamaged']
    trip_info_compare['delay_ratio'] = trip_info_compare['delay_duration'] / trip_info_compare['travel_time_used_undamaged']
    trip_info_compare.to_csv('trip_info_compare.csv', index=False)
    return True

def run_residual_demand(  # noqa: C901
        edge_geojson,
        node_geojson,
        od_file_pre,
        od_file_post,
        config_file,
        r2d_run_dir,
        residual_demand_dir,
):
    """
    Run the residual demand simulation.

    Parameters
    ----------
    edge_geojson : str
        Path to the edges GeoJSON file.
    node_geojson : str
        Path to the nodes GeoJSON file.
    od_file_pre : str
        Path to the pre-event origin-destination CSV file.
    od_file_post : str
        Path to the post-event origin-destination CSV file.
    config_file : str
        Path to the configuration JSON file.
    r2d_run_dir : str
        Directory containing the R2D run results.
    residual_demand_dir : str
        Directory to store the residual demand results.

    Returns
    -------
    None
    """
    if r2d_run_dir is None:
        run_dir = Path.cwd()
    else:
        run_dir = Path(r2d_run_dir)

    # Make a dir for ResidualDemand
    if residual_demand_dir is None:
        geojson_needed = False
        residual_demand_dir = run_dir / 'ResidualDemand'
    else:
        geojson_needed = True
        residual_demand_dir = Path(residual_demand_dir)

    if residual_demand_dir.exists():
        msg = 'ResidualDemand directory already exists'
        # Remove all the files and subfolders
        for filename in os.listdir(str(residual_demand_dir)):
            file_path = residual_demand_dir / filename
            try:
                # If it's a file, remove it
                if file_path.is_file() or Path(file_path).is_symlink():
                    file_path.unlink()
                # If it's a folder, remove it and its contents
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            except (OSError, shutil.Error) as e:
                msg = f"Failed to delete {file_path}. Reason: {e}"
                raise RuntimeError(msg) from e
        # raise ValueError(msg)
    else:
       residual_demand_dir.mkdir()
    os.chdir(residual_demand_dir)

    # Load the config file
    with Path(config_file).open() as f:
        config_file_dict = json.load(f)

    # Prepare edges and nodes files
    edges_gdf = gpd.read_file(edge_geojson).to_crs(epsg=6500)
    edges_gdf['length'] = edges_gdf['geometry'].apply(lambda x: x.length)
    edges_gdf = edges_gdf.to_crs(epsg=4326)
    two_way_edges = config_file_dict['TwoWayEdges']
    if  two_way_edges:
        edges_gdf_copy = edges_gdf.copy()
        edges_gdf_copy['StartNode'] = edges_gdf['EndNode']
        edges_gdf_copy['EndNode'] = edges_gdf['StartNode']
        edges_gdf = pd.concat([edges_gdf, edges_gdf_copy], ignore_index=True)
    edges_gdf = edges_gdf.reset_index()
    edges_gdf = edges_gdf.rename(columns={'index': 'uniqueid',
                                          'StartNode': 'start_nid',
                                          'EndNode': 'end_nid',
                                          'NumOfLanes': 'lanes',
                                          'MaxMPH': 'maxspeed',
                                          })
    # Assume that the capacity for each lane is 1800
    edges_gdf['capacity'] = edges_gdf['lanes']*1800
    edges_gdf['normal_capacity'] = edges_gdf['capacity']
    edges_gdf['normal_maxspeed'] = edges_gdf['maxspeed']
    # edges_gdf['fft'] = edges_gdf['length']/edges_gdf['maxspeed'] * 2.23694
    edges_gdf.to_csv('edges.csv', index=False)
    nodes_gdf = gpd.read_file(node_geojson)
    nodes_gdf = nodes_gdf.rename(columns={'nodeID': 'node_id'})
    nodes_gdf['lat'] = nodes_gdf['geometry'].apply(lambda x: x.y)
    nodes_gdf['lon'] = nodes_gdf['geometry'].apply(lambda x: x.x)
    nodes_gdf.to_csv('nodes.csv', index=False)

    edges_csv = residual_demand_dir / 'edges.csv'

    # Get Damage Input
    damage_input = config_file_dict['DamageInput']

    # Run the undamaged network
    undamged_dir_path = residual_demand_dir / 'undamaged'
    undamged_dir_path.mkdir()
    os.chdir(undamged_dir_path)
    run_on_undamaged_network(residual_demand_dir/'edges.csv',
                             residual_demand_dir/'nodes.csv',
                             od_file_pre,
                             Path(run_dir / 'Results_det.json'), config_file_dict)
    os.chdir(residual_demand_dir)

    if damage_input['Type'] == 'MostlikelyDamageState':
        # Create a Results_rlz.json file for the most likely damage state
        rlz = 0
        with Path(run_dir / f'Results_{rlz}.json').open() as f:
            results_rlz = json.load(f)
        with Path(run_dir/'Results_det.json').open() as f:
            results_det = json.load(f)
        for asset_type, asset_type_dict in results_rlz.items():
            for asset_subtype, asset_subtype_dict in asset_type_dict.items():
                for asset_id, asset_id_dict in asset_subtype_dict.items():
                    damage_dict = asset_id_dict['Damage']
                    for comp in damage_dict:
                        damage_dict[comp] = int(results_det[asset_type][asset_subtype]\
                        [asset_id]['R2Dres']['R2Dres_MostLikelyCriticalDamageState'])
                    if 'Loss' in asset_id_dict:
                        loss_dist = asset_id_dict['Loss']
                        for comp in loss_dist['Repair']['Cost']:
                            mean_cost_key = next(x for x in results_det[asset_type][asset_subtype]\
                                [asset_id]['R2Dres'] if x.startswith('R2Dres_mean_RepairCost'))
                            # A minmum cost of 0.1 is set to avoid division by zero
                            loss_dist['Repair']['Cost'][comp] = max(results_det[asset_type][asset_subtype]\
                            [asset_id]['R2Dres'][mean_cost_key], 0.1)
                        for comp in loss_dist['Repair']['Time']:
                            mean_time_key = next(x for x in results_det[asset_type][asset_subtype]\
                                [asset_id]['R2Dres'] if x.startswith('R2Dres_mean_RepairTime'))
                            # A minmum time of 0.1 is set to avoid division by zero
                            loss_dist['Repair']['Time'][comp] = max(results_det[asset_type][asset_subtype]\
                            [asset_id]['R2Dres'][mean_time_key], 0.1)
        rlz = 'mostlikely'
        with (run_dir /f'Results_{rlz}.json').open('w') as f:
            json.dump(results_rlz, f)

        # Create a directory for the realization
        Path(f'workdir.{rlz}').mkdir()
        os.chdir(f'workdir.{rlz}')
        rlz_run_dir = Path.cwd()
        ## Create arrays to store the delay results and congestion results
        undamaged_time, damaged_time = create_delay_agg(od_file_pre, od_file_post)
        undamaged_congestion, damaged_congestion = create_congestion_agg(edges_csv)
        # Run the simulation
        run_one_realization(residual_demand_dir/'edges.csv', residual_demand_dir/'nodes.csv',
                            undamged_dir_path, od_file_post, Path(run_dir / f'Results_{rlz}.json'),
                             Path(run_dir / 'Results_det.json'),
                            config_file_dict)
        # Append relay and congestion results to the aggregated results
        undamaged_time, damaged_time = append_to_delay_agg(
                undamaged_time, damaged_time, rlz_run_dir/'trip_info_compare.csv')
        undamaged_edge_vol_dir = undamged_dir_path / 'edge_vol'
        damaged_edge_vol_dir = rlz_run_dir / 'damaged'/'edge_vol'
        undamaged_congestion, damaged_congestion = append_to_congestion_agg(
            undamaged_congestion, damaged_congestion,
            undamaged_edge_vol_dir, damaged_edge_vol_dir,
            edges_csv)

        # Write the aggregated results to the travel_delay_stats.csv and Results_det.json file
        os.chdir(residual_demand_dir)
        aggregate_delay_results(undamaged_time, damaged_time, od_file_pre, od_file_post)
        aggregate_congestions_results_to_det(undamaged_congestion, damaged_congestion,
                                              Path(run_dir / 'Results_det.json'),
                                              edges_csv)

    elif damage_input['Type'] == 'SpecificRealization' or \
            damage_input['Type'] == 'SampleFromRealizations':
        # Get the realizations to run
        rlz_to_run = select_realizations_to_run(damage_input, run_dir)

        ## Create arrays to store the delay results and congestion results
        undamaged_time, damaged_time = create_delay_agg(od_file_pre, od_file_post)
        undamaged_congestion, damaged_congestion = create_congestion_agg(edges_csv)

        for rlz in rlz_to_run:
            print(f'Running realization {rlz}')  # noqa: T201
            # Create a directory for the realization
            rlz_run_dir = residual_demand_dir/f'workdir.{rlz}'
            rlz_run_dir.mkdir()
            os.chdir(rlz_run_dir)

            # Run the simulation
            run_one_realization(residual_demand_dir/'edges.csv', residual_demand_dir/'nodes.csv',
                            undamged_dir_path, od_file_post, Path(run_dir / f'Results_{rlz}.json'),
                             Path(run_dir / 'Results_det.json'),
                            config_file_dict)

            # Append relay and congestion results to the aggregated results
            undamaged_time, damaged_time = append_to_delay_agg(
                    undamaged_time, damaged_time, rlz_run_dir/'trip_info_compare.csv')
            undamaged_edge_vol_dir = undamged_dir_path / 'edge_vol'
            damaged_edge_vol_dir = rlz_run_dir / 'damaged'/'edge_vol'
            undamaged_congestion, damaged_congestion = append_to_congestion_agg(
                undamaged_congestion, damaged_congestion,
                undamaged_edge_vol_dir, damaged_edge_vol_dir,
                edges_csv)
            print(f'Rrealization {rlz} completed')  # noqa: T201
        # Write the aggregated results to the travel_delay_stats.csv and Results_det.json file
        os.chdir(residual_demand_dir)
        aggregate_delay_results(undamaged_time, damaged_time, od_file_pre, od_file_post)
        aggregate_congestions_results_to_det(undamaged_congestion, damaged_congestion,
                                              Path(run_dir / 'Results_det.json'),
                                              edges_csv)
    else:
        msg = 'Damage input type not recognized'
        raise ValueError(msg)

    if geojson_needed:
        # If run in tool box, compile a geojson for visualization
        # Otherwise, the geojson is compiled in rWHALE
        compile_r2d_results_geojson(residual_demand_dir,
                                    Path(run_dir / 'Results_det.json'))

    # f.close()


if __name__ == '__main__':
    # Defining the command line arguments

    workflowArgParser = argparse.ArgumentParser(  # noqa: N816
        'Run Residual Demand from the NHERI SimCenter rWHALE workflow for a set of assets.',
        allow_abbrev=False,
    )

    workflowArgParser.add_argument(
        '--edgeFile', help='Edges geojson', required=True
    )

    workflowArgParser.add_argument(
        '--nodeFile', help='Nodes geojson', required=True
    )

    workflowArgParser.add_argument(
        '--ODFilePre', help='Origin-Destination CSV file before hazard event', required=True
    )

    workflowArgParser.add_argument(
        '--ODFilePost', help='Origin-Destination CSV file after hazard event', required=True
    )

    workflowArgParser.add_argument(
        '--configFile', help='Config JSON file', required=True
    )

    workflowArgParser.add_argument(
        '--r2dRunDir',
        default=None,
        help='R2D run directory containing the results',
    )

    workflowArgParser.add_argument(
        '--residualDemandRunDir',
        default=None,
        help='Residual demand run directory',
    )

    workflowArgParser.add_argument(
        '--input',
        default=None,
        help='This is temporary for running in rWHALE and need to be remove in the future',
    )

    # Parsing the command line arguments
    wfArgs = workflowArgParser.parse_args()  # noqa: N816

    run_residual_demand(
        edge_geojson = wfArgs.edgeFile,
        node_geojson = wfArgs.nodeFile,
        od_file_pre = wfArgs.ODFilePre,
        od_file_post = wfArgs.ODFilePost,
        config_file = wfArgs.configFile,
        r2d_run_dir=wfArgs.r2dRunDir,
        residual_demand_dir=wfArgs.residualDemandRunDir,
    )
