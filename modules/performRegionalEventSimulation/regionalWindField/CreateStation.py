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
# Kuanshi Zhong
#

import json

import numpy as np
import pandas as pd


def get_label(options, labels, label_name):  # noqa: ANN001, ANN201, D103
    for option in options:
        if option in labels:
            labels = labels[labels != option]
            return option, labels

    print(f'WARNING: Could not identify the label for the {label_name}')  # noqa: T201, RET503


def create_stations(input_file, output_file, min_id, max_id):  # noqa: ANN001, ANN201
    """Reading input csv file for stations and saving data to output json file
    Input:
        input_file: the filename of the station csv file
        output_file: the filename of the output json file
        min_id: the min ID to start
        max_id: the max ID to end
    Output:
        run_tag: 0 - success, 1 - input failure, 2 - output failure
    """  # noqa: D205, D400, D401, D415
    # Reading csv data
    run_tag = 1
    try:
        stn_df = pd.read_csv(input_file, header=0, index_col=0)
    except:  # noqa: E722
        run_tag = 0
        return run_tag  # noqa: RET504
    # Max and Min IDs
    stn_ids_min = np.min(stn_df.index.values)
    stn_ids_max = np.max(stn_df.index.values)
    if min_id is None:
        min_id = stn_ids_min
    if max_id is None:
        max_id = stn_ids_max
    min_id = np.max([stn_ids_min, min_id])
    max_id = np.min([stn_ids_max, max_id])
    selected_stn = stn_df.loc[min_id:max_id, :]
    # Extracting data
    labels = selected_stn.columns.values  # noqa: PD011
    lon_label, labels = get_label(
        ['Longitude', 'longitude', 'lon', 'Lon'], labels, 'longitude'
    )
    lat_label, labels = get_label(
        ['Latitude', 'latitude', 'lat', 'Lat'], labels, 'latitude'
    )
    stn_file = {'Stations': []}
    for stn_id, stn in selected_stn.iterrows():
        # Collecting station data
        tmp = {'ID': stn_id, 'Longitude': stn[lon_label], 'Latitude': stn[lat_label]}
        stn_file['Stations'].append(tmp)
    # Saving data to the output file
    if output_file:
        with open(output_file, 'w') as f:  # noqa: PTH123
            json.dump(stn_file, f, indent=2)
    # Returning the final run state
    return stn_file
