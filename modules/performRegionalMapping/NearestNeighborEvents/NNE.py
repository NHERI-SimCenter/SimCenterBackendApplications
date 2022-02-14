# -*- coding: utf-8 -*-
#
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

import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.neighbors import NearestNeighbors

def find_neighbors(building_file, event_grid_file, samples, neighbors, filter_label, seed):

    # read the event grid data file
    event_grid_path = Path(event_grid_file).resolve()
    event_dir = event_grid_path.parent
    event_grid_file = event_grid_path.name

    grid_df = pd.read_csv(event_dir / event_grid_file, header=0)

    # store the locations of the grid points in X
    lat_E = grid_df['Latitude']
    lon_E = grid_df['Longitude']
    X = np.array([[lo, la] for lo, la in zip(lon_E, lat_E)])

    # prepare the tree for the nearest neighbor search
    if filter_label != "":
        neighbors_to_get = min(neighbors*10, len(lon_E))
    else:
        neighbors_to_get = neighbors
    nbrs = NearestNeighbors(n_neighbors = neighbors_to_get, algorithm='ball_tree').fit(X)

    # load the building data file
    with open(building_file, 'r') as f:
        bldg_dict = json.load(f)

    # prepare a dataframe that holds building filenames and locations
    bim_df = pd.DataFrame(columns=['Latitude', 'Longitude', 'file'],
                          index=np.arange(len(bldg_dict)))
    for i, bldg in enumerate(bldg_dict):
        with open(bldg['file'], 'r') as f:
            bldg_data = json.load(f)

        bldg_loc = bldg_data['GeneralInformation']['location']
        bim_df.iloc[i]['Longitude'] = bldg_loc['longitude']
        bim_df.iloc[i]['Latitude'] = bldg_loc['latitude']
        bim_df.iloc[i]['file'] = bldg['file']

    # store building locations in Y
    Y = np.array([[lo, la] for lo, la in zip(bim_df['Longitude'], bim_df['Latitude'])])

    # collect the neighbor indices and distances for every building
    distances, indices = nbrs.kneighbors(Y)
    distances = distances + 1e-20

    # initialize the random generator
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # iterate through the buildings and store the selected events in the BIM
    for bldg_i, (bim_id, dist_list, ind_list) in enumerate(zip(bim_df.index,
                                                          distances,
                                                          indices)):

        # open the BIM file
        bldg_file = bim_df.iloc[bim_id]['file']
        with open(bldg_file, 'r') as f:
            bldg_data = json.load(f)

        if filter_label != '':
            # soil type of building
            bldg_label = bldg_data['GeneralInformation'][filter_label]
            # soil types of all initial neighbors
            grid_label = grid_df[filter_label][ind_list]

            # only keep the distances and indices corresponding to neighbors
            # with the same soil type
            dist_list  = dist_list[(grid_label==bldg_label).values]
            ind_list   = ind_list[(grid_label==bldg_label).values]

            # return dist_list & ind_list with a length equals neighbors
            # assuming that at least neighbors grid points exist with
            # the same filter_label as the building

            # because dist_list, ind_list sorted initially in order of increasing
            # distance, just take the first neighbors grid points of each
            dist_list = dist_list[:neighbors]
            ind_list = ind_list[:neighbors]

        # calculate the weights for each neighbor based on their distance
        dist_list = 1./(dist_list**2.0)
        weights = np.array(dist_list)/np.sum(dist_list)

        # get the pre-defined number of samples for each neighbor
        nbr_samples = np.where(rng.multinomial(1, weights, samples) == 1)[1]

        # this is the preferred behavior, the else caluse is left for legacy inputs
        if grid_df.iloc[0]['GP_file'][-3:] == 'csv':

            # We assume that every grid point has the same type and number of
            # event data. That is, you cannot mix ground motion records and
            # intensity measures and you cannot assign 10 records to one point
            # and 15 records to another.

            # Load the first file and identify if this is a grid of IM or GM
            # information. GM grids have GM record filenames defined in the
            # grid point files.
            first_file = pd.read_csv(event_dir / grid_df.iloc[0]['GP_file'],
                                     header=0)
            if first_file.columns[0]=='TH_file':
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
                    event_df = pd.read_csv(event_dir / event_collection_file,
                                           header=0)

                    # append the GM record name to the event list
                    event_list.append(event_df.iloc[event_j,0])

                    # append the scale factor (or 1.0) to the scale list
                    if len(event_df.columns) > 1:
                        scale_list.append(float(event_df.iloc[event_j,1]))
                    else:
                        scale_list.append(1.0)

                # if the grid has intensity measures
                elif event_type == 'intensityMeasure':

                    # save the collection file name and the IM row id
                    event_list.append(grid_df.iloc[nbr_index]['GP_file']+f'x{event_j}')

                    # IM collections are not scaled
                    scale_list.append(1.0)

        # TODO: update the LLNL input data and remove this clause
        else:
            event_list = []
            for e, i in zip(nbr_samples, ind_list):
                event_list += [grid_df.iloc[i]['GP_file'],]*e

            scale_list = np.ones(len(event_list))

        # prepare a dictionary of events
        event_list_json = []
        for e_i, event in enumerate(event_list):
            #event_list_json.append({
            #    #"EventClassification": "Earthquake",
            #    "fileName": f'{event}x{e_i:05d}',
            #    "factor": scale_list[e_i],
            #    #"type": event_type
            #    })
            event_list_json.append([f'{event}x{e_i:05d}', scale_list[e_i]])

        # save the event dictionary to the BIM
        bldg_data['Events'] = {
            #"EventClassification": "Earthquake",
            "EventFolderPath": str(event_dir),
            "Events": event_list_json,
            "type": event_type
            #"type": "SimCenterEvents"
        }

        with open(bldg_file, 'w') as f:
            json.dump(bldg_data, f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--buildingFile')
    parser.add_argument('--filenameEVENTgrid')
    parser.add_argument('--samples', type=int)
    parser.add_argument('--neighbors', type=int)
    parser.add_argument('--filter_label', default="")
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    find_neighbors(args.buildingFile, args.filenameEVENTgrid,
                   args.samples,args.neighbors, args.filter_label,
                   args.seed)