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
from scipy.cluster.vq import vq
import importlib
import os

def create_event(asset_file, event_grid_file, multipleEvents, doParallel):


    # check if running parallel
    numP = 1
    procID = 0
    runParallel = False
    
    if doParallel == "True":
        mpi_spec = importlib.util.find_spec("mpi4py")
        found = mpi_spec is not None
        if found:
            import mpi4py
            from mpi4py import MPI
            runParallel = True
            comm = MPI.COMM_WORLD
            numP = comm.Get_size()
            procID = comm.Get_rank();
            if numP < 2:
                doParallel = "False"
                runParallel = False
                numP = 1
                procID = 0
    
    # read the event grid data file
    event_grid_path = Path(event_grid_file).resolve()
    event_dir = event_grid_path.parent
    event_grid_file = event_grid_path.name

    grid_df = pd.read_csv(event_dir / event_grid_file, header=0)
    
    # store the locations of the grid points in X
    lat_E = grid_df['Latitude']
    lon_E = grid_df['Longitude']
    X = np.array([[lo, la] for lo, la in zip(lon_E, lat_E)])
    
    # load the asset data file
    with open(asset_file, 'r', encoding="utf-8") as f:
        asset_dict = json.load(f)

    # prepare a dataframe that holds asset filenames and locations
    AIM_df = pd.DataFrame(columns=['Latitude', 'Longitude', 'file'], index=np.arange(len(asset_dict)))

    count = 0    
    for i, asset in enumerate(asset_dict):

        if runParallel == False or (i % numP) == procID:

            with open(asset['file'], 'r', encoding="utf-8") as f:
                asset_data = json.load(f)

            asset_loc = asset_data['GeneralInformation']['location']
            AIM_df.iloc[count]['Longitude'] = asset_loc['longitude']
            AIM_df.iloc[count]['Latitude'] = asset_loc['latitude']
            AIM_df.iloc[count]['file'] = asset['file']
            count = count + 1
        
    # store asset locations in Y
    Y = np.array([[lo, la] for lo, la in zip(AIM_df['Longitude'], AIM_df['Latitude']) if not np.isnan(lo) and not np.isnan(la)])
    
    #print(Y)
    #print(sub_grid)
    
    # Find the index of the closest point - each index corresponds to the gridpoint index
    closest, distances = vq(Y, X)
    
#    print("****closest",closest)
#    print("****distances",distances)
#
#    print("****num found",len(closest))
#    print("****num Y",np.size(Y, 0))
#    print("****num X",np.size(X, 0))


    # check to ensure we found all of the assets
    if len(closest) != np.size(Y, 0) :
        print("Error, the number of assets needs to be equal to the number of grid points")
        print("The number of assets is "+str(np.size(Y, 0))+" and the number of grid points is " + len(closest))
        return 1
        
        
    # iterate through the assets and store the selected events in the AIM
    for idx, AIM_id in enumerate(AIM_df.index):

        # open the AIM file
        asset_file = AIM_df.iloc[AIM_id]['file']
       
        with open(asset_file, 'r', encoding="utf-8") as f:
            asset_data = json.load(f)

        # this is the preferred behavior, the else caluse is left for legacy inputs
        if grid_df.iloc[0]['GP_file'][-3:] == 'csv':

            # We assume that every grid point has the same type and number of
            # event data. That is, you cannot mix ground motion records and
            # intensity measures and you cannot assign 10 records to one point
            # and 15 records to another.

            # Load the first file and identify if this is a grid of IM or GM
            # information. GM grids have GM record filenames defined in the
            # grid point files.
            first_file = pd.read_csv(event_dir / grid_df.iloc[0]['GP_file'],header=0)
            
            if first_file.columns[0]=='TH_file':
                event_type = 'timeHistory'
            else:
                event_type = 'intensityMeasure'
                
            event_count = first_file.shape[0]

            # collect the list of events and scale factors
            event_list = []
            scale_list = []
            
            closestPnt = grid_df.iloc[closest[idx]]

            # if the grid has ground motion records...
            if event_type == 'timeHistory':

                # load the file for the selected grid point
                event_collection_file = closestPnt['GP_file']
                
                event_df = pd.read_csv(event_dir / event_collection_file, header=0)

                # append the GM record name to the event list
                event_list.append(event_df.iloc[0,0])

                # append the scale factor (or 1.0) to the scale list
                if len(event_df.columns) > 1:
                    scale_list.append(float(event_df.iloc[0,1]))
                else:
                    scale_list.append(1.0)
                
                # If GP_file contains multiple events    
                if multipleEvents:
                    # Read the GP_file
                    if event_df.shape[0] > 1:
                        for row in range(1,event_df.shape[0]):
                            event_list.append(event_df.iloc[row,0])
                            # append the scale factor (or 1.0) to the scale list
                            if len(event_df.columns) > 1:
                                scale_list.append(float(event_df.iloc[row,1]))
                            else:
                                scale_list.append(1.0)

            # if the grid has intensity measures
            elif event_type == 'intensityMeasure':

                # save the collection file name and the IM row id
                event_list.append(closestPnt['GP_file']+f'x{0}')

                # IM collections are not scaled
                scale_list.append(1.0)

                # If GP_file contains multiple events  
                if multipleEvents:
                    # Read the GP_file
                    GP_file = os.path.join(event_dir, closestPnt['GP_file'])
                    GP_file_df = pd.read_csv(GP_file, header=0)
                    if GP_file_df.shape[0] > 1:
                        for row in range(1,GP_file_df.shape[0]):
                            event_list.append(closestPnt['GP_file']+f'x{row}')
                            scale_list.append(1.0)

        # TODO: update the LLNL input data and remove this clause
        else:
            event_list = []
            for e, i in zip(nbr_samples, ind_list):
                event_list += [closestPnt['GP_file'],]*e

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

        
        # save the event dictionary to the AIM

        # save the event dictionary to the BIM                                          
        asset_data['Events'] = [{}]        
        asset_data['Events'][0] = {
            #"EventClassification": "Earthquake",                               
            "EventFolderPath": str(event_dir),
            "Events": event_list_json,
            "type": event_type
            #"type": "SimCenterEvents"                                          
        }
        
        with open(asset_file, 'w', encoding="utf-8") as f:
            json.dump(asset_data, f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--assetFile')
    parser.add_argument('--filenameEVENTgrid')
    parser.add_argument('--multipleEvents', default="True")
    parser.add_argument('--doParallel', default="False")    
    parser.add_argument("-n", "--numP", default='8')
    parser.add_argument("-m", "--mpiExec", default='mpixece')
    args = parser.parse_args()

    create_event(args.assetFile, args.filenameEVENTgrid, args.multipleEvents, args.doParallel)
