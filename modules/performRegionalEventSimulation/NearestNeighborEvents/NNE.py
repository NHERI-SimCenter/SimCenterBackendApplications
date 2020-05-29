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

def find_neighbors(building_file, event_metadata, samples, neighbors):
    
    meta_df = pd.read_csv(event_metadata, sep='\s+',header=0)
    
    lat_E = meta_df['lat']
    lon_E = meta_df['lon']
    X = np.array([[lo, la] for lo, la in zip(lon_E, lat_E)])

    nbrs = NearestNeighbors(n_neighbors = neighbors, algorithm='ball_tree').fit(X)

    with open(building_file, 'r') as f:
        bldg_dict = json.load(f)

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

    Y = np.array([[lo, la] for lo, la in zip(bim_df['lon'], bim_df['lat'])])

    distances, indices = nbrs.kneighbors(Y)

    for i, (bim_id, dist_list, ind_list) in enumerate(zip(bim_df.index, 
                                                          distances, 
                                                          indices)):

        dist_list = 1./(dist_list**2.0)
        weights = np.array(dist_list)/np.sum(dist_list)

        evt_count = multinomial(samples, weights)

        if meta_df.iloc[0]['sta'][-3:] == 'csv':
            gm_dir = posixpath.dirname(event_metadata)

            event_list = []
            scale_list = []
            for e, i in zip(evt_count, ind_list):
                gm_collection_file = meta_df.iloc[i]['sta']

                gm_df = pd.read_csv(posixpath.join(gm_dir, gm_collection_file), header=None)
                
                event_list +=  gm_df.iloc[:,0].values.tolist() * e 

                if len(gm_df.columns) > 1:
                    scale_list += gm_df.iloc[:,1].values.tolist() * e
                else:
                    scale_list += np.ones(gm_df.shape[0]).tolist() * e                 

        else:
            

            event_list = []
            for e, i in zip(evt_count, ind_list):
                event_list += [meta_df.iloc[i]['sta'],]*e

            scale_list = np.ones(len(event_list))

        event_list_json = []
        for e_i, event in enumerate(event_list):
            event_list_json.append({
                "EventClassification": "Earthquake",
                "fileName": f'{event}x{e_i:03d}',
                "factor": scale_list[e_i],
                "type": "SW4"
                })

        bldg_file = bim_df.iloc[bim_id]['file']
        with open(bldg_file, 'r') as f:
            bldg_data = json.load(f)

        bldg_data['Events'] = {
            "EventClassification": "Earthquake",
            "Events": event_list_json,
            "type": "SW4_Events"
        }

        with open(bldg_file, 'w') as f:
            json.dump(bldg_data, f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--buildingFile')
    parser.add_argument('--filenameEVENTmeta')
    parser.add_argument('--samples', type=int)
    parser.add_argument('--neighbors', type=int)
    args = parser.parse_args()

    find_neighbors(args.buildingFile, args.filenameEVENTmeta,
                   args.samples,args.neighbors)