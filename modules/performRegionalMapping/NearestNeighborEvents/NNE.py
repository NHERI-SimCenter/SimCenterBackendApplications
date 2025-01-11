#  # noqa: INP001, D100
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
# Adam Zsarnóczay
# Tamika Bassman
#

import argparse  # noqa: I001
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import geopandas as gpd
from pyproj import CRS

def load_sc_geojson(file_path):
    # Read the GeoJSON into a dictionary
    with Path(file_path).open() as f:
        geojson_data = json.load(f)
    crs = CRS.from_user_input(geojson_data['crs']['properties']['name'])
    # Create a GeoDataFrame from the GeoJSON
    return gpd.GeoDataFrame.from_features(geojson_data['features'], crs=crs)

def find_neighbors(  # noqa: C901, D103
    asset_file,
    event_grid_file,
    samples,
    neighbors,
    filter_label,
    seed,
    do_parallel,
):
    # check if running parallel
    num_processes = 1
    process_id = 0
    run_parallel = False

    if do_parallel == 'True':
        mpi_spec = importlib.util.find_spec('mpi4py')
        found = mpi_spec is not None
        if found:
            from mpi4py import MPI

            run_parallel = True
            comm = MPI.COMM_WORLD
            num_processes = comm.Get_size()
            process_id = comm.Get_rank()
            if num_processes < 2:  # noqa: PLR2004
                do_parallel = 'False'
                run_parallel = False
                num_processes = 1
                process_id = 0

    # read the event grid data file
    event_grid_path = Path(event_grid_file).resolve()
    event_dir = event_grid_path.parent
    event_grid_file = event_grid_path.name

    # Check if the file is a CSV or a GIS file
    file_extension = Path(event_grid_file).suffix.lower()

    if file_extension == '.csv':
        # Existing code for CSV files
        grid_df = pd.read_csv(event_dir / event_grid_file, header=0)

        # store the locations of the grid points in grid_locations
        lat_e = grid_df['Latitude']
        lon_e = grid_df['Longitude']
        grid_locations = np.array([[lo, la] for lo, la in zip(lon_e, lat_e)])

        if filter_label == '':
            grid_extra_keys = list(
                grid_df.drop(['GP_file', 'Longitude', 'Latitude'], axis=1).columns
            )
    elif file_extension == '.geojson':
        # Read the geojson file
        gdf = load_sc_geojson(event_dir / event_grid_file)

        # Ensure the GIS file is in a geographic coordinate system
        if not gdf.crs.is_geographic:
            gdf = gdf.to_crs(epsg=4326)

        lat_e = gdf['geometry'].apply(lambda pt: pt.y)
        lon_e = gdf['geometry'].apply(lambda pt: pt.x)
        grid_locations = np.array([[lo, la] for lo, la in zip(lon_e, lat_e)])
        if filter_label == '':
            grid_extra_keys = list(
                gdf.drop(['geometry'], axis=1).columns
            )

    else:
        # Else assume GIS files - works will all gis files that geopandas supports
        gdf = gpd.read_file(event_dir / event_grid_file)

        # Ensure the GIS file is in a geographic coordinate system
        if not gdf.crs.is_geographic:
            gdf = gdf.to_crs(epsg=4326)  # Convert to WGS84

        # Extract coordinates from the geometry
        gdf['Longitude'] = gdf.geometry.x
        gdf['Latitude'] = gdf.geometry.y

        # store the locations of the grid points in grid_locations
        lat_e = gdf['Latitude']
        lon_e = gdf['Longitude']
        grid_locations = np.array([[lo, la] for lo, la in zip(lon_e, lat_e)])

        if filter_label == '':
            grid_extra_keys = list(
                gdf.drop(['geometry', 'Longitude', 'Latitude'], axis=1).columns
            )

        # Convert GeoDataFrame to regular DataFrame for consistency with the rest of the code
        grid_df = pd.DataFrame(gdf.drop(columns='geometry'))

    # prepare the tree for the nearest neighbor search
    if filter_label != '' or len(grid_extra_keys) > 0:
        neighbors_to_get = min(neighbors * 10, len(lon_e))
    else:
        neighbors_to_get = neighbors

    nbrs = NearestNeighbors(n_neighbors=neighbors_to_get, algorithm='ball_tree').fit(
        grid_locations
    )

    # load the building data file
    with open(asset_file, encoding='utf-8') as f:  # noqa: PTH123
        asset_dict = json.load(f)

    # prepare a dataframe that holds asset filenames and locations
    aim_df = pd.DataFrame(
        columns=['Latitude', 'Longitude', 'file'], index=np.arange(len(asset_dict))
    )

    count = 0
    for i, asset in enumerate(asset_dict):
        if run_parallel == False or (i % num_processes) == process_id:  # noqa: E712
            with open(asset['file'], encoding='utf-8') as f:  # noqa: PTH123
                asset_data = json.load(f)

            asset_loc = asset_data['GeneralInformation']['location']
            aim_id = aim_df.index[count]
            aim_df.loc[aim_id, 'Longitude'] = asset_loc['longitude']
            aim_df.loc[aim_id, 'Latitude'] = asset_loc['latitude']
            aim_df.loc[aim_id, 'file'] = asset['file']
            count = count + 1

    # store building locations in bldg_locations
    bldg_locations = np.array(
        [
            [lo, la]
            for lo, la in zip(aim_df['Longitude'], aim_df['Latitude'])
            if not np.isnan(lo) and not np.isnan(la)
        ]
    )

    # collect the neighbor indices and distances for every building
    distances, indices = nbrs.kneighbors(bldg_locations)
    distances = distances + 1e-20

    # initialize the random generator
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    count = 0

    # iterate through the buildings and store the selected events in the AIM
    for asset_i, (aim_id, dist_list, ind_list) in enumerate(  # noqa: B007
        zip(aim_df.index, distances, indices)
    ):
        # open the AIM file
        aim_index_id = aim_df.index[aim_id]
        asst_file = aim_df.loc[aim_index_id, 'file']

        with open(asst_file, encoding='utf-8') as f:  # noqa: PTH123
            asset_data = json.load(f)

        if filter_label != '':
            # soil type of building
            asset_label = asset_data['GeneralInformation'][filter_label]
            # soil types of all initial neighbors
            grid_label = grid_df[filter_label][ind_list]

            # only keep the distances and indices corresponding to neighbors
            # with the same soil type
            dist_list = dist_list[(grid_label == asset_label).values]  # noqa: PD011, PLW2901
            ind_list = ind_list[(grid_label == asset_label).values]  # noqa: PD011, PLW2901

            # return dist_list & ind_list with a length equals neighbors
            # assuming that at least neighbors grid points exist with
            # the same filter_label as the building

            # because dist_list, ind_list sorted initially in order of increasing
            # distance, just take the first neighbors grid points of each
            dist_list = dist_list[:neighbors]  # noqa: PLW2901
            ind_list = ind_list[:neighbors]  # noqa: PLW2901

        if len(grid_extra_keys) > 0:
            filter_labels = []
            for key in asset_data['GeneralInformation'].keys():  # noqa: SIM118
                if key in grid_extra_keys:
                    filter_labels.append(key)  # noqa: PERF401

            filter_list = [True for i in dist_list]
            for filter_label in filter_labels:  # noqa: PLR1704
                asset_label = asset_data['GeneralInformation'][filter_label]
                grid_label = grid_df[filter_label][ind_list]
                filter_list_i = (grid_label == asset_label).values  # noqa: PD011
                filter_list = filter_list and filter_list_i

            # only keep the distances and indices corresponding to neighbors
            # with the same soil type
            dist_list = dist_list[filter_list]  # noqa: PLW2901
            ind_list = ind_list[filter_list]  # noqa: PLW2901

            # return dist_list & ind_list with a length equals neighbors
            # assuming that at least neighbors grid points exist with
            # the same filter_label as the building

            # because dist_list, ind_list sorted initially in order of increasing
            # distance, just take the first neighbors grid points of each
            dist_list = dist_list[:neighbors]  # noqa: PLW2901
            ind_list = ind_list[:neighbors]  # noqa: PLW2901

        # calculate the weights for each neighbor based on their distance
        dist_list = 1.0 / (dist_list**2.0)  # noqa: PLW2901
        weights = np.array(dist_list) / np.sum(dist_list)

        # get the pre-defined number of samples for each neighbor
        nbr_samples = np.where(rng.multinomial(1, weights, samples) == 1)[1]

        # this is the preferred behavior, the else clause is left for legacy inputs
        if file_extension == '.csv':
            if grid_df.iloc[0]['GP_file'][-3:] == 'csv':
                # We assume that every grid point has the same type and number of
                # event data. That is, you cannot mix ground motion records and
                # intensity measures and you cannot assign 10 records to one point
                # and 15 records to another.

                # Load the first file and identify if this is a grid of IM or GM
                # information. GM grids have GM record filenames defined in the
                # grid point files.
                first_file = pd.read_csv(
                    event_dir / grid_df.iloc[0]['GP_file'], header=0
                )
                if first_file.columns[0] == 'TH_file':
                    event_type = 'timeHistory'
                else:
                    event_type = 'intensityMeasure'
                event_count = first_file.shape[0]

                # collect the list of events and scale factors
                event_list = []
                scale_list = []

                # for each neighbor
                for sample_j, nbr in enumerate(nbr_samples):
                    # make sure we resample events if samples > event_count
                    event_j = sample_j % event_count

                    # get the index of the nth neighbor
                    nbr_index = ind_list[nbr]

                    # if the grid has ground motion records...
                    if event_type == 'timeHistory':
                        # load the file for the selected grid point
                        event_collection_file = grid_df.iloc[nbr_index]['GP_file']
                        event_df = pd.read_csv(
                            event_dir / event_collection_file, header=0
                        )

                        # append the GM record name to the event list
                        event_list.append(event_df.iloc[event_j, 0])

                        # append the scale factor (or 1.0) to the scale list
                        if len(event_df.columns) > 1:
                            scale_list.append(float(event_df.iloc[event_j, 1]))
                        else:
                            scale_list.append(1.0)

                    # if the grid has intensity measures
                    elif event_type == 'intensityMeasure':
                        # save the collection file name and the IM row id
                        event_list.append(
                            grid_df.iloc[nbr_index]['GP_file'] + f'x{event_j}'
                        )

                        # IM collections are not scaled
                        scale_list.append(1.0)

            # TODO: update the LLNL input data and remove this clause  # noqa: TD002
            else:
                event_list = []
                for e, i in zip(nbr_samples, ind_list):
                    event_list += [
                        grid_df.iloc[i]['GP_file'],
                    ] * e

                scale_list = np.ones(len(event_list))
        if file_extension == '.geojson':
            # collect the list of events and scale factors
            event_list = []
            scale_list = []
            # for each neighbor
            columns = [x for x in gdf.columns if x != 'geometry']
            event_count = len(gdf[columns[0]].iloc[0])
            if columns[0] == 'TH_file':
                event_type = 'timeHistory'
            else:
                event_type = 'intensityMeasure'
            for sample_j, nbr in enumerate(nbr_samples):
                # make sure we resample events if samples > event_count
                event_j = sample_j % event_count

                # get the index of the nth neighbor
                nbr_index = ind_list[nbr]

                # if the grid has ground motion records...
                if event_type == 'timeHistory':
                    # load the file for the selected grid point
                    event_collection_file = grid_df.iloc[nbr_index]['GP_file']
                    event_df = pd.read_csv(
                        event_dir / event_collection_file, header=0
                    )

                    # append the GM record name to the event list
                    event_list.append(event_df.iloc[event_j, 0])

                    # append the scale factor (or 1.0) to the scale list
                    if len(event_df.columns) > 1:
                        scale_list.append(float(event_df.iloc[event_j, 1]))
                    else:
                        scale_list.append(1.0)

                # if the grid has intensity measures
                elif event_type == 'intensityMeasure':
                    # save the collection file name and the IM row id
                    im_columns = gdf.columns
                    im_list = [x[event_j] for x in gdf.iloc[nbr_index][im_columns]]
                    event_list.append(
                        gdf.iloc[nbr_index]['GP_file'] + f'x{event_j}'
                    )

                    # IM collections are not scaled
                    scale_list.append(1.0)
        else:
            event_list = []
            scale_list = []
            event_type = 'intensityMeasure'

            # Determine event_count (number of IMs per grid point)
            im_columns = [
                col
                for col in grid_df.columns
                if col not in ['geometry', 'Longitude', 'Latitude']
            ]
            # event_count = len(im_columns)
            event_count = 1

            # for each neighbor
            for sample_j, nbr in enumerate(nbr_samples):
                # make sure we resample events if samples > event_count
                event_j = sample_j % event_count

                # get the index of the nth neighbor
                nbr_index = ind_list[nbr]

                # For GIS files, create a new CSV file
                csv_filename = f'Site_{nbr_index}.csv'

                csv_path = event_dir / csv_filename

                if not csv_path.exists():
                    # Create a CSV file with data from the GIS file
                    # Use actual data from the GIS file if available, otherwise use dummy data
                    im_columns = [
                        col
                        for col in grid_df.columns
                        if col not in ['geometry', 'Longitude', 'Latitude']
                    ]

                    im_data = pd.DataFrame(
                        {
                            col: [grid_df.iloc[nbr_index][col]] * event_count
                            for col in im_columns
                        }
                    )

                    im_data.to_csv(csv_path, index=False)
                # save the collection file name and the IM row id
                event_list.append(csv_filename + f'x{event_j}')

                # IM collections are not scaled
                scale_list.append(1.0)

        # prepare a dictionary of events
        event_list_json = []
        for e_i, event in enumerate(event_list):
            # event_list_json.append({
            #    #"EventClassification": "Earthquake",
            #    "fileName": f'{event}x{e_i:05d}',
            #    "factor": scale_list[e_i],
            #    #"type": event_type
            #    })
            event_list_json.append([f'{event}x{e_i:05d}', scale_list[e_i]])

        # save the event dictionary to the AIM
        # TODO: we assume there is only one event  # noqa: TD002
        # handling multiple events will require more sophisticated inputs

        if 'Events' not in asset_data:
            asset_data['Events'] = [{}]
        elif len(asset_data['Events']) == 0:
            asset_data['Events'].append({})

        asset_data['Events'][0].update(
            {
                # "EventClassification": "Earthquake",
                'EventFolderPath': str(event_dir),
                'Events': event_list_json,
                'type': event_type,
                # "type": "SimCenterEvents"
            }
        )

        with open(asst_file, 'w', encoding='utf-8') as f:  # noqa: PTH123
            json.dump(asset_data, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assetFile')
    parser.add_argument('--filenameEVENTgrid')
    parser.add_argument('--samples', type=int)
    parser.add_argument('--neighbors', type=int)
    parser.add_argument('--filter_label', default='')
    parser.add_argument('--doParallel', default='False')
    parser.add_argument('-n', '--numP', default='8')
    parser.add_argument('-m', '--mpiExec', default='mpiexec')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    find_neighbors(
        args.assetFile,
        args.filenameEVENTgrid,
        args.samples,
        args.neighbors,
        args.filter_label,
        args.seed,
        args.doParallel,
    )
