# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 The Regents of the University of California
# Copyright (c) 2019 Leland Stanford Junior University
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
# SimCenter Backend Applications. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Frank McKenna
# Adam Zsarnóczay
# Wael Elhaddad
# Stevan Gavrilovic

import argparse, sys, os

def create_asset_files(output_file, asset_source_file, asset_filter, doParallel):

    # these imports are here to save time when the app is called without
    # the -getRV flag
    import json
    import numpy as np
    import pandas as pd
    import importlib

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

    # Get the out dir, may not always be in the results folder if multiple assets are used
    outDir = os.path.dirname(output_file)
    
    # check if a filter is provided
    if asset_filter is not None:
        assets_requested = []
        for assets in asset_filter.split(','):
            if "-" in assets:
                asset_low, asset_high = assets.split("-")
                assets_requested += list(range(int(asset_low), int(asset_high)+1))
            else:
                assets_requested.append(int(assets))
        assets_requested = np.array(assets_requested)
        
    # load the CSV file with the asset information
    assets_df = pd.read_csv(asset_source_file, header=0, index_col=0)

    # if there is a filter, then pull out only the required assets
    if asset_filter is not None:
        assets_available = assets_df.index.values
        assets_to_run = assets_requested[
            np.where(np.in1d(assets_requested, assets_available))[0]]
        selected_assets = assets_df.loc[assets_to_run]
    else:
        selected_assets = assets_df

    # identify the labels
    labels = selected_assets.columns.values

    assets_array = []

    # for each asset...
    count = 0
    for asset_id, asset in selected_assets.iterrows():

        if runParallel == False or (count % numP) == procID:

            # initialize the AIM file
            AIM_i = {
                "RandomVariables": [],
                "GeneralInformation": dict(
                    AIM_id = str(int(asset_id)),
                    location = {
                        'latitude': asset["Latitude"],
                        'longitude': asset["Longitude"]
                    }
                )
            }

            # save every label as-is
            for label in labels:
                AIM_i["GeneralInformation"].update({label: asset[label]})

            AIM_file_name = "{}-AIM.json".format(asset_id)
        
            AIM_file_name = os.path.join(outDir,AIM_file_name)
            
            with open(AIM_file_name, 'w') as f:
                json.dump(AIM_i, f, indent=2)

            assets_array.append(dict(id=str(asset_id), file=AIM_file_name))

        count = count + 1

    if procID != 0:

        # if not P0, write data to output file with procID in name and barrier

        output_file = os.path.join(outDir,f'tmp_{procID}.json')

        with open(output_file, 'w') as f:
            json.dump(assets_array, f, indent=0)
    
        comm.Barrier()        

    else:

        if runParallel == True:

            # if parallel & P0, barrier so that all files written above, then loop over other processor files: open, load data and append
            comm.Barrier()        

            for i in range(1, numP):
                fileToAppend = os.path.join(outDir,f'tmp_{i}.json')
                with open(fileToAppend, 'r') as data_file:
                    json_data = data_file.read()
                assetsToAppend = json.loads(json_data)
                assets_array += assetsToAppend

        with open(output_file, 'w') as f:
            json.dump(assets_array, f, indent=2)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--assetFile',
        help = "Path to the file that will contain a list of asset ids and "
               "corresponding AIM filenames")
    parser.add_argument('--assetSourceFile',
        help = "Path to the CSV file with the asset inventory")
    parser.add_argument('--filter',
        help = "Filter applied to select a subset of assets from the "
               "inventory",
        default=None)
    parser.add_argument('--doParallel', default="False")    
    parser.add_argument("-n", "--numP", default='8')
    parser.add_argument("-m", "--mpiExec", default='mpiexec')
    parser.add_argument('--getRV',
        help = "Identifies the preparational stage of the workflow. This app "
               "is only used in that stage, so it does not do anything if "
               "called without this flag.",
        default=False,
        nargs='?', const=True)

    args = parser.parse_args()

    if args.getRV:
        sys.exit(create_asset_files(args.assetFile, args.assetSourceFile, args.filter, args.doParallel))
    else:
        pass # not used
