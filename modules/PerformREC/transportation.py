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

import json
import pandas as pd
from pathlib import Path
from brails.workflow.TransportationElementHandler import (
    TransportationElementHandler)

import numpy as np
from scipy.spatial.distance import cdist
from Typing import List


def system_state(detfile: str, csv_file_dir: str) -> None:
    """
    Process a given det file and update the system state.

    This function reads a JSON file containing damage states and updates it
    with  capacity ratios based on the damage state. It also checks for the
    existence of required CSV files, created them if missing, and generates
    a file listing closed edges.

    Args__
        detfile (str): Path to the JSON file containing the asset data.
        csv_file_dir (str): Directory containing the CSV files needed for
                                running network simulations.

    Returns__
        The function does not return any value. It creates updated det file
        and CSV file necessary to run network simulations

    Raises__
        FileNotFoundError: If the `detfile` or any required CSV files are not
                            found.
        json.JSONDecodeError: If the `detfile` cannot be parsed as JSON.
        KeyError: If expected keys are missing in the JSON data.
        ValueError: If there are issues with data conversions, e.g., JSON to
                    integer.

    Examples__
        >>> system_state('damage_states.json', '/path/to/csv/files')
        Missing files: nodes.csv
        All required files are present.
        # This will process 'damage_states.json', update it, and use CSV files
            in '/path/to/csv/files'

        >>> system_state('damage_states.json', '/path/to/nonexistent/dir')
        Missing files: edges.csv, nodes.csv
        # This will attempt to process 'damage_states.json' and handle missing
            files in a non-existent directory
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
            print(f"Missing files: {', '.join(missing_files)}")
            out = False
        else:
            print("All required files are present.")
            out = True

        return out

    assets = ['Bridge', 'Roadway', 'Tunnel']
    csv_filenames = ['edges.csv', 'nodes.csv', 'closed_edges.csv']

    no_identifier = {'Bridge': 'StructureNumber',
                     'Roadway': 'TigerOID',
                     'Tunnel': 'TunnelNumber'}
    capacity_map = {0: 1, 1: 1, 2: 1, 3: 0.5, 4: 0}

    # Read damage states for det file and determine element capacity ratios
    # 1 denotes fully open and 0 denotes fully closed:
    capacity_dict = {}
    with open(detfile, "r") as f:
        temp = json.load(f)
        data = temp['TransportationNetwork']
        for asset_type in assets:
            datadict = data[asset_type]
            for key in datadict.keys():
                item_id = datadict[key]['GeneralInformation'][no_identifier[
                    asset_type]]
                ds = int(datadict[key]['R2Dres']
                         ['R2Dres_MostLikelyCriticalDamageState'])
                capacity_ratio = capacity_map[ds]
                capacity_dict[item_id] = capacity_ratio
                datadict[key]['GeneralInformation']['Open'] = capacity_ratio

    # Update det file with closure information:
    temp = detfile.split('.')
    detfile_updated = temp[0] + '_updated.' + temp[1]

    with open(detfile_updated, 'w') as file:
        json.dump(data, file, indent=2)

    # Create link closures for network simulations:
    fexists = files_exist(csv_file_dir, csv_filenames[:-1])

    if fexists:
        graph_edge_file = csv_file_dir + '/' + csv_filenames[0]
    else:
        element_handler = TransportationElementHandler(assets)
        element_handler.get_graph_network(detfile, csv_file_dir)
        graph_edge_file = element_handler['output_files']['graph_network'][0]

    df = pd.read_csv(graph_edge_file)
    closed_edges = []
    for key in datadict.keys():
        matches = df[df['name'] == key]

    if not matches.empty and datadict[key] == 1:
        uniqueids = matches['uniqueid'].tolist()
        closed_edges.append(uniqueids)

    # Write closed edges:
    edge_closure_file = csv_file_dir + '/' + csv_filenames[-1]
    with open(edge_closure_file, 'w') as file:
        # Write each item on a new line
        for item in closed_edges:
            file.write(item + '\n')


def update_od_file(
    old_nodes: str,
    old_det: str,
    new_nodes: str,
    new_det: str,
    od: str,
    origin_ids: List[int]
) -> pd.DataFrame:
    """
    Update origin-destination (OD) file based on changes in population data.

    This function updates an OD file by calculating the population changes at
    each node between two time steps and generates trips originating from
    specified origins and ending at nodes where the population has increased.
    The updated OD data is saved to a new file.

    Args__
        old_nodes (str): Path to the CSV file containing the node information
                            at the previous time step.
        old_det (str): Path to the JSON file containing the building
                        information at the previous time step.
        new_nodes (str): Path to the CSV file containing the node information
                            at the current time step.
        new_det (str): Path to the JSON file containing the building
                        information at the current time step.
        od (str): Path to the existing OD file to be updated.
        origin_ids (List[int]): List of IDs representing possible origins for
                                generating trips.

    Returns__
        pd.DataFrame: The updated OD DataFrame with new trips based on
                        population changes.

    Raises__
        FileNotFoundError: If any of the provided file paths are incorrect or
                            the files do not exist.
        json.JSONDecodeError: If the JSON files cannot be read or parsed
                                correctly.
        KeyError: If expected keys are missing in the JSON data.
        ValueError: If there are issues with data conversions or calculations.

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
        - Ensure that the columns `lat`, `lon`, and `node_id` are present in
            the nodes CSV files.
        - The `det` files should contain the `Buildings` and
            `GeneralInformation` sections with appropriate keys.
        - The OD file should have the columns `agent_id`, `origin_nid`,
            `destin_nid`, `hour`, and `quarter`.
    """
    # Extract the building information from the det file and convert it to a
    # pandas dataframe
    def extract_building_from_det(det):
        # Open the det file
        f = open(det)

        # Return the JSON object as a dictionary
        json_data = json.load(f)

        # Extract the required information and convert it to a pandas dataframe
        extracted_data = []

        for AIM_id, info in json_data['Buildings']['Building'].items():
            general_info = info.get('GeneralInformation', {})
            extracted_data.append({
                'AIM_id': AIM_id,
                'Latitude': general_info.get('Latitude'),
                'Longitude': general_info.get('Longitude'),
                'Population': general_info.get('Population')
            })

        df = pd.DataFrame(extracted_data)

        return df

    # Aggregate the population in buildings to the closest road network node
    def closest_neighbour(building_df, nodes_df):
        # Find the nearest road network node to each building
        nodes_xy = np.array(
            [nodes_df['lat'].values, nodes_df['lon'].values]).transpose()
        building_df['closest_node'] = building_df.apply(lambda x: cdist(
            [(x['Latitude'], x['Longitude'])], nodes_xy).argmin(), axis=1)

        # Merge the road network and building dataframes
        merged_df = pd.merge(
            nodes_df, building_df, left_on='node_id', right_on='closest_node',
            how='left')
        merged_df = merged_df.drop(
            columns=['AIM_id', 'Latitude', 'Longitude', 'closest_node'])
        merged_df = merged_df.fillna(0)

        # Aggregate population of  neareast buildings to the road network node
        updated_nodes_df = merged_df.groupby('node_id').agg(
            {'lon': 'first', 'lat': 'first', 'geometry': 'first',
             'Population': 'sum'}).reset_index()

        return updated_nodes_df

    # Function to add the population information to the nodes file
    def update_nodes_file(nodes, det):
        # Read the nodes file
        nodes_df = pd.read_csv(nodes)
        # Extract the building information from the det file and convert it to
        # a pandas dataframe
        building_df = extract_building_from_det(det)
        # Aggregate the population in buildings to the closest road network
        # node
        updated_nodes_df = closest_neighbour(building_df, nodes_df)

        return updated_nodes_df

    # Read the od file
    od_df = pd.read_csv(od)
    # Add the population information to nodes dataframes at the last and
    # current time steps
    old_nodes_df = update_nodes_file(old_nodes, old_det)
    new_nodes_df = update_nodes_file(new_nodes, new_det)
    # Calculate the population changes at each node (assuming that population
    # will only increase at each node)
    population_change_df = old_nodes_df.copy()
    population_change_df['Population Change'] = new_nodes_df['Population'] - \
        old_nodes_df['Population']
    population_change_df['Population Change'].astype(int)
    # Randomly generate the trips that start at one of the connections between
    # Alameda Island and Oakland, and end at the nodes where population
    # increases and append them to the od file
    # Generate OD data for each node with increased population and append it
    # to the od file
    for index, row in population_change_df.iterrows():
        # Only process rows with positive population difference
        if row['Population_Difference'] > 0:
            for i in range(row['Population_Difference']):
                # Generate random origin_nid
                origin_nid = np.random.choice(origin_ids)
                # Set the destin_nid to the node_id of the increased population
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
