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

import argparse, posixpath, json
import numpy as np
import pandas as pd
from numpy.random import multinomial

from sklearn.neighbors import NearestNeighbors

# distance, area, volume
m = 1.

mm = 0.001 * m
cm = 0.01 * m
km = 1000. * m

inch = 0.0254
ft = 12. * inch

# acceleration
mps2 = m

inchps2 = inch
ftps2 = ft

g = 9.80665 * mps2

def find_neighbors(building_file, event_grid_file, samples, neighbors, filter_label):
    
    # read the event grid data file
    grid_df = pd.read_csv(event_grid_file, sep='\s+',header=0)
    event_dir = posixpath.dirname(event_grid_file)
    
    # store the locations of the grid points in X
    lat_E = grid_df['lat']
    lon_E = grid_df['lon']
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
    bim_df = pd.DataFrame(columns=['lat', 'lon', 'file'], index=np.arange(len(bldg_dict)))
    for i, bldg in enumerate(bldg_dict):
        with open(bldg['file'], 'r') as f:
            bldg_data = json.load(f)

        if 'GI' in bldg_data:
            bldg_loc = bldg_data['GI']['location']
        else:
            bldg_loc = bldg_data['GeneralInformation']['location']
        bim_df.iloc[i]['lon'] = bldg_loc['longitude']
        bim_df.iloc[i]['lat'] = bldg_loc['latitude']
        bim_df.iloc[i]['file'] = bldg['file']

    # store building locations in Y
    Y = np.array([[lo, la] for lo, la in zip(bim_df['lon'], bim_df['lat'])])

    # collect the neighbor indices and distances for every building
    distances, indices = nbrs.kneighbors(Y)

    # iterate through the buildings and store the selected events in the BIM
    for bldg_i, (bim_id, dist_list, ind_list) in enumerate(zip(bim_df.index, 
                                                          distances, 
                                                          indices)):

        # open the BIM file
        bldg_file = bim_df.iloc[bim_id]['file']
        with open(bldg_file, 'r') as f:
            bldg_data = json.load(f)

        # temporary - check the acceleration unit for time history analysis
        acc_unit = bldg_data['GI']['units'].get('acceleration', None)
        length_unit = bldg_data['GI']['units'].get('length', None)
        if acc_unit is not None:
            if acc_unit == 'inps2':
                acc_unit = 'inchps2'
            if length_unit == 'in':
                length_unit = 'inch'

            acc_scale = globals()[acc_unit] / globals()[length_unit]
        else:
            acc_scale = 1.0

        if filter_label != '':
            # soil type of building
            bldg_label = bldg_data['GI'][filter_label]
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
        nbr_samples = np.where(multinomial(1, weights, samples) == 1)[1]

        # this is the preferred behavior, the else caluse is left for legacy inputs
        if grid_df.iloc[0]['sta'][-3:] == 'csv':

            # We assume that every grid point has the same type and number of 
            # event data. That is, you cannot mix ground motion records and
            # intensity measures and you cannot assign 10 records to one point
            # and 15 records to another.

            # Load the first file and identify if this is a grid of IM or GM 
            # information. GM grids have GM record filenames defined in the 
            # grid point files.
            first_file = pd.read_csv(
                posixpath.join(event_dir, grid_df.iloc[0]['sta']), header=0)
            if first_file.columns[0]=='GM_file':
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
                    event_collection_file = grid_df.iloc[nbr_index]['sta']
                    event_df = pd.read_csv(
                        posixpath.join(event_dir, event_collection_file), header=0)
                        
                    # append the GM record name to the event list
                    event_list.append(event_df.iloc[event_j,0])

                    # append the scale factor (or 1.0) to the scale list
                    if len(event_df.columns) > 1:
                        scale_list.append(event_df.iloc[event_j,1] * acc_scale)
                    else:
                        scale_list.append(1.0 * acc_scale) 

                # if the grid has intensity measures
                elif event_type == 'intensityMeasure':

                    # save the collection file name and the IM row id
                    event_list.append(grid_df.iloc[nbr_index]['sta']+f'x{event_j}')

                    # IM collections are not scaled
                    scale_list.append(1.0)            

        # TODO: update the LLNL input data and remove this clause
        else:
            event_list = []
            for e, i in zip(nbr_samples, ind_list):
                event_list += [grid_df.iloc[i]['sta'],]*e

            scale_list = np.ones(len(event_list))

        # prepare a dictionary of events
        event_list_json = []
        for e_i, event in enumerate(event_list):
            event_list_json.append({
                "EventClassification": "Earthquake",
                "fileName": f'{event}x{e_i:05d}',
                "factor": scale_list[e_i],
                "type": event_type
                })

        # save the event dictionary to the BIM
        bldg_data['Events'] = {
            "EventClassification": "Earthquake",
            "Events": event_list_json,
            "type": "SimCenterEvents"
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
    args = parser.parse_args()

    find_neighbors(args.buildingFile, args.filenameEVENTgrid,
                   args.samples,args.neighbors, args.filter_label)