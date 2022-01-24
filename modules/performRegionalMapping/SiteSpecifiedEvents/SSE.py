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
# Stevan Gavrilovic
# Adam Zsarnóczay
# Tamika Bassman
#

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def create_event(building_file, event_grid_file):

    # read the event grid data file
    event_grid_path = Path(event_grid_file).resolve()
    event_dir = event_grid_path.parent
    event_grid_file = event_grid_path.name

    grid_df = pd.read_csv(event_dir / event_grid_file, header=0)
    
    # The subgrid that will hold the grid points only for the building
    sub_grid = pd.DataFrame()
    

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
    
    # Get the subgrid for the particular subset of buildings
    for idx, row in grid_df.iterrows():
        
        lon = row['Longitude']
        lat = row['Latitude']
        
        for it in Y:
            #print(it[0],it[1])
            if np.isclose(it[0],lon) == True and np.isclose(it[1],lat) == True:
                sub_grid = sub_grid.append(row)
                break
    
    # print(sub_grid)
    
    # store the locations of the grid points in X
    lat_E = sub_grid['Latitude']
    lon_E = sub_grid['Longitude']
    X = np.array([[lo, la] for lo, la in zip(lon_E, lat_E)])

    
    if Y.size != X.size :
        print("Error, the number of buildings needs to be equal to the number of grid points")
        return
        
    for it in zip(np.nditer(X),np.nditer(Y)):
        if np.isclose(it[0],it[1]) == False :
            print("Error, the lat/lon coordinates between the buildings and the event grid points need to match")

    # iterate through the buildings and store the selected events in the BIM
    for idx, bim_id in enumerate(bim_df.index):

        # open the BIM file
        bldg_file = bim_df.iloc[bim_id]['file']
        with open(bldg_file, 'r') as f:
            bldg_data = json.load(f)


        # this is the preferred behavior, the else clause is left for legacy inputs
        if sub_grid.iloc[0]['GP_file'][-3:] == 'csv':

            # We assume that every grid point has the same type and number of
            # event data. That is, you cannot mix ground motion records and
            # intensity measures and you cannot assign 10 records to one point
            # and 15 records to another.

            # Load the first file and identify if this is a grid of IM or GM
            # information. GM grids have GM record filenames defined in the
            # grid point files.
            first_file = pd.read_csv(event_dir / sub_grid.iloc[0]['GP_file'],
                                     header=0)
            if first_file.columns[0]=='TH_file':
                event_type = 'timeHistory'
            else:
                event_type = 'intensityMeasure'
            event_count = first_file.shape[0]

            # collect the list of events and scale factors
            event_list = []
            scale_list = []


            # if the grid has ground motion records...
            if event_type == 'timeHistory':

                # load the file for the selected grid point
                event_collection_file = sub_grid.iloc[idx]['GP_file']
                event_df = pd.read_csv(event_dir / event_collection_file,
                                       header=0)

                # append the GM record name to the event list
                event_list.append(event_df.iloc[idx,0])

                # append the scale factor (or 1.0) to the scale list
                if len(event_df.columns) > 1:
                    scale_list.append(float(event_df.iloc[idx,1]))
                else:
                    scale_list.append(1.0)

            # if the grid has intensity measures
            elif event_type == 'intensityMeasure':

                # save the collection file name and the IM row id
                event_list.append(sub_grid.iloc[idx]['GP_file']+f'x{0}')

                # IM collections are not scaled
                scale_list.append(1.0)

        # TODO: update the LLNL input data and remove this clause
        else:
            event_list = []
            for e, i in zip(nbr_samples, ind_list):
                event_list += [sub_grid.iloc[i]['GP_file'],]*e

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
    args = parser.parse_args()

    create_event(args.buildingFile, args.filenameEVENTgrid)
