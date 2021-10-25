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

import sys, argparse, json
import posixpath
from pathlib import Path, PurePath

def create_building_files(output_file, building_source_file, bldg_filter):

    # these imports are here to save time when the app is called without
    # the -getRV flag
    import numpy as np
    import pandas as pd

    # get the units
    main_dir = Path(PurePath(output_file).parent)
    #main_dir = posixpath.dirname(output_file)

    with open(main_dir / 'units.json', 'r') as f:
        units = json.load(f)

    #with open(posixpath.join(main_dir, 'units.json'), 'r') as f:
    #    units = json.load(f)

    # check if a filter is provided
    if bldg_filter is not None:
        bldgs_requested = []
        for bldgs in bldg_filter.split(','):
            if "-" in bldgs:
                bldg_low, bldg_high = bldgs.split("-")
                bldgs_requested += list(range(int(bldg_low), int(bldg_high)+1))
            else:
                bldgs_requested.append(int(bldgs))
        bldgs_requested = np.array(bldgs_requested)

    # load the CSV file with the building information
    bldgs_df = pd.read_csv(building_source_file, header=0, index_col=0)

    # if there is a filter, then pull out only the required buildings
    if bldg_filter is not None:
        bldgs_available = bldgs_df.index.values
        bldgs_to_run = bldgs_requested[
            np.where(np.in1d(bldgs_requested, bldgs_available))[0]]
        selected_bldgs = bldgs_df.loc[bldgs_to_run]
    else:
        selected_bldgs = bldgs_df

    # identify the labels
    labels = selected_bldgs.columns.values

    buildings_array = []

    # for each building...
    for bldg_id, bldg in selected_bldgs.iterrows():

        # initialize the BIM file
        BIM_i = {
            "RandomVariables": [],
            "GeneralInformation": dict(
                BIM_id = str(int(bldg_id)),
                location = {
                    'latitude': bldg["Latitude"],
                    'longitude': bldg["Longitude"]
                },
                units = units
            )
        }

        # save every label as-is
        for label in labels:
            BIM_i["GeneralInformation"].update({label: bldg[label]})

        BIM_file_name = "{}-BIM.json".format(bldg_id)

        with open(BIM_file_name, 'w') as f:
            json.dump(BIM_i, f, indent=2)

        buildings_array.append(dict(id=str(bldg_id), file=BIM_file_name))

    with open(output_file, 'w') as f:
        json.dump(buildings_array, f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--buildingFile',
        help = "Path to the file that will contain a list of building ids and "
               "corresponding BIM filenames")
    parser.add_argument('--buildingSourceFile',
        help = "Path to the CSV file with the building inventory")
    parser.add_argument('--filter',
        help = "Filter applied to select a subset of buildings from the "
               "inventory",
        default=None)
    parser.add_argument('--getRV',
        help = "Identifies the preparational stage of the workflow. This app "
               "is only used in that stage, so it does not do anything if "
               "called without this flag.",
        default=False,
        nargs='?', const=True)

    args = parser.parse_args()

    if args.getRV:
        sys.exit(create_building_files(args.buildingFile,
                                       args.buildingSourceFile, args.filter))
    else:
        pass # not used
