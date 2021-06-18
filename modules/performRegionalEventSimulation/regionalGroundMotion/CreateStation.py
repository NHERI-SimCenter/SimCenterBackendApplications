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
# Kuanshi Zhong
#

import json
import numpy as np
import pandas as pd


def get_label(options, labels, label_name):

    for option in options:
        if option in labels:
            labels = labels[labels != option]
            return option, labels

    print(f'WARNING: Could not identify the label for the {label_name}')


class Station:
    """
    A class for stations in an earthquake scenario
    """
    def __init__(self, lon, lat, vs30 = None, z2p5 = None):
        # Initializing the location, vs30, z2.5, Tcond and other Tags
        self.lon = lon
        self.lat = lat
        self.vs30 = vs30
        self.z2p5 = z2p5

    def get_location():
        # Returning the geo location
        return self.lon, self.lat

    def get_vs30():
        # Returning the Vs30 at the station
        return self.vs30

    def get_z2p5():
        # Returning the z2.5 of the station
        return self.z2p5


def create_stations(input_file, output_file, min_id, max_id, vs30_tag, z1_tag, z25_tag):
    """
    Reading input csv file for stations and saving data to output json file
    Input:
        input_file: the filename of the station csv file
        output_file: the filename of the output json file
        min_id: the min ID to start
        max_id: the max ID to end
        vs30_tag: 1 - interpolate global Vs30, 2 - Thompson vs30, 0 - leave Vs30
        z1_tag: z1pt0 tag: 1 - using empirical equation, 0 - leave it as null
        z2pt5_tag: z2pt5 tag: 1 - using empirical equation, 0 - leave it as null
    Output:
        stn_file: dictionary of station data
    """
    # Reading csv data
    run_tag = 1
    try:
        stn_df = pd.read_csv(input_file, header=0, index_col=0)
    except:
        run_tag = 0
        return run_tag
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
    labels = selected_stn.columns.values
    lon_label, labels = get_label(['Longitude', 'longitude', 'lon', 'Lon'], labels, 'longitude')
    lat_label, labels = get_label(['Latitude', 'latitude', 'lat', 'Lat'], labels, 'latitude')
    if any([i in ['Vs30', 'vs30', 'Vs_30', 'vs_30'] for i in labels]):
        vs30_label, labels = get_label(['Vs30', 'vs30', 'Vs_30', 'vs_30'], labels, 'vs30')
    else:
        vs30_label = 'vs30'
    if any([i in ['Z2p5', 'z2p5', 'Z2pt5', 'z2pt5', 'Z25', 'z25', 'Z2.5', 'z2.5'] for i in labels]):
        z2p5_label, labels = get_label(['Z2p5', 'z2p5', 'Z2pt5', 'z2pt5', 'Z25', 'z25', 'Z2.5', 'z2.5'], labels, 'z2p5')
    else:
        z2p5_label = 'z2p5'
    if any([i in ['Z1p0', 'z1p0', 'Z1pt0', 'z1pt0', 'Z1', 'z1', 'Z1.0', 'z1.0'] for i in labels]):
        z1p0_label, labels = get_label(['Z1p0', 'z1p0', 'Z1pt0', 'z1pt0', 'Z1', 'z1', 'Z1.0', 'z1.0'], labels, 'z1p0')
    else:
        z1p0_label = 'z1p0'
    if any([i in ['zTR', 'ztr', 'ZTR', 'DepthToRock'] for i in labels]):
        zTR_label, labels = get_label(['zTR', 'ztr', 'ZTR', 'DepthToRock'], labels, 'zTR')
    else:
        zTR_label = 'zTR'
    STN = []
    stn_file = {
        'Stations': []
    }
    # Interpolate Vs30
    if vs30_label not in selected_stn.keys() and vs30_tag == 1:
        print('CreateStation: Interpolating global Vs30 map for defined stations.')
        selected_stn.loc[:,vs30_label] = get_vs30_global(selected_stn[lat_label], selected_stn[lon_label])
    if vs30_label not in selected_stn.keys() and vs30_tag == 2:
        print('CreateStation: Interpolating Thompson Vs30 map for defined stations.')
        selected_stn.loc[:,vs30_label] = get_vs30_thompson(selected_stn[lat_label], selected_stn[lon_label])
    # Interpolate zTR
    if zTR_label not in selected_stn.keys():
        print('CreateStation: Interpolating global depth to rock map for defined stations.')
        selected_stn.loc[:,zTR_label] = [max(0,x) for x in get_zTR_global(selected_stn[lat_label], selected_stn[lon_label])]
    
    for stn_id, stn in selected_stn.iterrows():
        # Creating a Station object
        STN.append(Station(
            stn[lon_label], stn[lat_label],
            stn.get(vs30_label, 760.0), stn.get(z2p5_label, 9.0)
        ))
        # Collecting station data
        tmp = {
            'ID': stn_id,
            'Longitude': stn[lon_label],
            'Latitude': stn[lat_label]
        }

        if stn.get(vs30_label):
            tmp.update({'Vs30': stn.get(vs30_label)})
        else:
            if vs30_tag == 1:
                tmp.update({'Vs30': get_vs30_global([stn[lat_label]], [stn[lon_label]])[0]})
            elif vs30_tag == 2:
                tmp.update({'Vs30': get_vs30_thompson([stn[lat_label]], [stn[lon_label]])[0]})

        if stn.get(z1p0_label):
            tmp.update({'z1pt0': stn.get(z2p5_label)})
        else:
            if z1_tag:
                if not tmp.get('Vs30'):
                    tmp.update({'Vs30': get_vs30_global(stn[lat_label], stn[lon_label])[0]})
                tmp.update({'z1pt0': get_z1(tmp['Vs30'])})

        if stn.get(z2p5_label):
            tmp.update({'z2.5': stn.get(z2p5_label)})
        else:
            if z25_tag:
                if not tmp.get('z1pt0'):
                    if not tmp.get('Vs30'):
                        tmp.update({'Vs30': get_vs30_global(stn[lat_label], stn[lon_label])[0]})
                    tmp.update({'z1pt0': get_z1(tmp['Vs30'])})
                tmp.update({'z2pt5': get_z25(tmp['z1pt0'])})

        if stn.get(zTR_label):
            tmp.update({'zTR': stn.get(zTR_label)})
        else:
            tmp.update({'zTR': max(0,get_zTR_global([stn[lat_label]], [stn[lon_label]])[0])})
        
        stn_file['Stations'].append(tmp)
        #stn_file['Stations'].append({
        #    'ID': stn_id,
        #    'Longitude': stn[lon_label],
        #    'Latitude': stn[lat_label],
        #    'Vs30': stn.get(vs30_label, 760.0),
        #    'z2.5': stn.get(z2p5_label, 9.0)
        #})
    # Saving data to the output file
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(stn_file, f, indent=2)
    # Returning the final run state
    return stn_file


def create_gridded_stations(input_file, output_file, div_lon = 2, div_lat = 2,
        delta_lon = None, delta = None):
    """
    Reading input csv file for the grid, generating stations, and saving data
    to output json file
    Input:
        input_file: the filename of the station csv file
        output_file: the filename of the output json file
        div_lon: number of divisions along longitude
        div_lat: number of divisions along latitude
        delta_lon: delta degree along longitude
        delta_lat: delta degree along latitude
    Output:
        run_tag: 0 - success, 1 - input failure, 2 - outupt failure
    """
    # Reading csv data
    run_tag = 0
    try:
        gstn_df = pd.read_csv(input_file, header=0, index_col=0)
    except:
        run_tag = 1
        return run_tag
    if np.max(gstn_df.index.values) != 2:
        run_tag = 1
        return run_tag
    else:
        labels = gstn_df.columns.values
        lon_label, labels = get_label(['Longitude', 'longitude', 'lon', 'Lon'], labels, 'longitude')
        lat_label, labels = get_label(['Latitude', 'latitude', 'lat', 'Lat'], labels, 'latitude')
        lon_temp = []
        lat_temp = []
        for gstn_id, gstn in gstn_df.iterrows():
            lon_temp.append(gstn[lon_label])
            lat_temp.append(gstn[lat_label])
    # Generating the grid
    dlon = (np.max(lon_temp) - np.min(lon_temp)) / div_lon
    dlat = (np.max(lat_temp) - np.min(lat_temp)) / div_lat
    if delta_lon is not None:
        delta_lon = np.min([delta_lon, dlon])
    if delta_lat is not None:
        delta_lat = np.min([delta_lat, dlat])
    glon, glat = np.meshgrid(
        np.arange(np.min(lon_temp), np.max(lon_temp), delta_lon),
        np.arange(np.min(lat_temp), np.max(lat_temp), delta_lat)
    )


def get_vs30_global(lat, lon):
    """
    Interpolate global Vs30 at given latitude and longitude
    Input:
        lat: list of latitude
        lon: list of longitude
    Output:
        vs30: list of vs30
    """
    import pickle
    import os
    from scipy import interpolate
    # Loading global Vs30 data
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(cwd+'/database/site/global_vs30_4km.pkl', 'rb') as f:
        vs30_global = pickle.load(f)
    # Interpolation function (linear)
    interpFunc = interpolate.interp2d(vs30_global['Longitude'], vs30_global['Latitude'], vs30_global['Vs30'])
    vs30 = [float(interpFunc(x, y)) for x,y in zip(lon, lat)]
    # return
    return vs30


def get_vs30_thompson(lat, lon):
    """
    Interpolate global Vs30 at given latitude and longitude
    Input:
        lat: list of latitude
        lon: list of longitude
    Output:
        vs30: list of vs30
    """
    import pickle
    import os
    from scipy import interpolate
    # Loading Thompson Vs30 data
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(cwd+'/database/site/thompson_vs30_4km.pkl', 'rb') as f:
        vs30_thompson = pickle.load(f)
    # Interpolation function (linear)
    # Thompson's map gives zero values for water-covered region and outside CA -> use 760 for default
    print('CreateStation: Warning - approximate 760 m/s for sites not supported by Thompson Vs30 map (water/outside CA).')
    vs30_thompson['Vs30'][vs30_thompson['Vs30']<0.1] = 760
    interpFunc = interpolate.interp2d(vs30_thompson['Longitude'], vs30_thompson['Latitude'], vs30_thompson['Vs30'])
    vs30 = [float(interpFunc(x, y)) for x,y in zip(lon, lat)]
    
    # return
    return vs30


def get_z1(vs30):
    """
    Compute z1 based on the prediction equation by Chiou and Youngs (2013)
    """

    z1 = -7.15 / 4.0 * np.log((vs30 ** 4 + 571.0 ** 4) / (1360.0 ** 4 + 571.0 ** 4))
    # return
    return z1


def get_z25(z1):
    """
    Compute z25 based on the prediction equation by Campbell and Bozorgnia (2013)
    """
    z25 = 0.748 + 2.218 * z1
    # return
    return z25


def get_zTR_global(lat, lon):
    """
    Interpolate depth to rock at given latitude and longitude
    Input:
        lat: list of latitude
        lon: list of longitude
    Output:
        vs30: list of vs30
    """
    import pickle
    import os
    from scipy import interpolate
    # Loading depth to rock data
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(cwd+'/database/site/global_zTR_4km.pkl', 'rb') as f:
        zTR_global = pickle.load(f)
    # Interpolation function (linear)
    interpFunc = interpolate.interp2d(zTR_global['Longitude'], zTR_global['Latitude'], zTR_global['zTR'])
    zTR = [float(interpFunc(x, y)) for x,y in zip(lon, lat)]
    # return
    return zTR


def export_site_prop(stn_file, output_dir, filename):
    """
    saving a csv file for stations
    Input:
        stn_file: a dictionary of station data
        output_path: output directory
        filename: output filename
    Output:
        run_tag: 0 - success, 1 - outupt failure
    """
    import os
    from pathlib import Path

    print(stn_file)
    station_name = ['site'+str(j)+'.csv' for j in range(len(stn_file))]
    lat = [stn_file[j]['Latitude'] for j in range(len(stn_file))]
    lon = [stn_file[j]['Longitude'] for j in range(len(stn_file))]
    vs30 = [stn_file[j]['Vs30'] for j in range(len(stn_file))]
    df = pd.DataFrame({
        'GP_file': station_name,
        'Longitude': lon,
        'Latitude': lat,
        'Vs30': vs30
    })
    df = pd.DataFrame.from_dict(stn_file)

    output_dir = os.path.join(os.path.dirname(Path(output_dir)),
                    os.path.basename(Path(output_dir)))
    try:
        os.makedirs(output_dir)
    except:
        print('HazardSimulation: output folder already exists.')
    # save the csv
    df.to_csv(os.path.join(output_dir, filename), index = False)