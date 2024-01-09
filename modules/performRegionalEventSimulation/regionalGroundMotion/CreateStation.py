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

import json, copy
import numpy as np
import pandas as pd
import socket
from tqdm import tqdm
if 'stampede2' not in socket.gethostname():
    from FetchOpenSHA import get_site_vs30_from_opensha
    from FetchOpenSHA import get_site_z1pt0_from_opensha, get_site_z2pt5_from_opensha


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

    def get_location(self):
        # Returning the geo location
        return self.lon, self.lat

    def get_vs30(self):
        # Returning the Vs30 at the station
        return self.vs30

    def get_z2p5(self):
        # Returning the z2.5 of the station
        return self.z2p5


def create_stations(input_file, output_file, filterIDs, vs30_tag, z1_tag, z25_tag, zTR_tag=0, soil_flag=False, soil_model_type=None, soil_user_fun=None):
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
    if filterIDs is not None:
        stns_requested = []
        for stns in filterIDs.split(','):
            if "-" in stns:
                stn_low, stn_high = stns.split("-")
                stns_requested += list(range(int(stn_low), int(stn_high)+1))
            else:
                stns_requested.append(int(stns))
        stns_requested = np.array(stns_requested)
        stns_available = stn_df.index.values
        stns_to_run = stns_requested[
            np.where(np.in1d(stns_requested, stns_available))[0]]
        selected_stn = stn_df.loc[stns_to_run]
    else:
        selected_stn = stn_df
    # stn_ids_min = np.min(stn_df.index.values)
    # stn_ids_max = np.max(stn_df.index.values)
    # if min_id is None:
    #     min_id = stn_ids_min
    # if max_id is None:
    #     max_id = stn_ids_max
    # min_id = np.max([stn_ids_min, min_id])
    # max_id = np.min([stn_ids_max, max_id])
    # selected_stn = copy.copy(stn_df.loc[min_id:max_id, :])
    selected_stn.index = list(range(len(selected_stn.index)))
    # Extracting data
    labels = selected_stn.columns.values
    lon_label, labels = get_label(['Longitude', 'longitude', 'lon', 'Lon'], labels, 'longitude')
    lat_label, labels = get_label(['Latitude', 'latitude', 'lat', 'Lat'], labels, 'latitude')
    if any([i in ['Vs30', 'vs30', 'Vs_30', 'vs_30'] for i in labels]):
        vs30_label, labels = get_label(['Vs30', 'vs30', 'Vs_30', 'vs_30'], labels, 'vs30')
    else:
        vs30_label = 'Vs30'
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
        zTR_label = 'DepthToRock'
    if soil_flag:
        if any([i in ['Model', 'model', 'SoilModel', 'soilModel'] for i in labels]):
            soil_model_label, labels = get_label(['Model', 'model', 'SoilModel', 'soilModel'], labels, 'Model')
        else:
            soil_model_label = 'Model'
            if soil_model_type is not None:
                model_map = {'Elastic Isotropic': 'EI',
                             'Multiaxial Cyclic Plasticity': 'BA', 
                             'User': 'USER'}
                soil_model_tag = model_map.get(soil_model_type, 'EI')
                # add a 'Model' column to selected_stn
                selected_stn[soil_model_label] = [soil_model_tag for x in range(len(selected_stn.index))]
    STN = []
    stn_file = {
        'Stations': []
    }
    # Get Vs30
    if vs30_label in selected_stn.keys():
        tmp = selected_stn.iloc[:,list(selected_stn.keys()).index(vs30_label)].values.tolist()
        if len(tmp):
            nan_loc = [x[0] for x in np.argwhere(np.isnan(tmp)).tolist()]
        else:
            nan_loc = []
    else:
        nan_loc = list(range(len(selected_stn.index)))
    if len(nan_loc) and vs30_tag == 1:
        print('CreateStation: Interpolating global Vs30 map for defined stations.')
        selected_stn.loc[nan_loc,vs30_label] = get_vs30_global(selected_stn.iloc[nan_loc,list(selected_stn.keys()).index(lat_label)].values.tolist(), 
                                                               selected_stn.iloc[nan_loc,list(selected_stn.keys()).index(lon_label)].values.tolist())
    if len(nan_loc) and vs30_tag == 2:
        print('CreateStation: Interpolating Thompson Vs30 map for defined stations.')
        selected_stn.loc[nan_loc,vs30_label] = get_vs30_thompson(selected_stn.iloc[nan_loc,list(selected_stn.keys()).index(lat_label)].values.tolist(), 
                                                                 selected_stn.iloc[nan_loc,list(selected_stn.keys()).index(lon_label)].values.tolist())
    if len(nan_loc) and vs30_tag == 3:
        print('CreateStation: Fetch National Crustal Model Vs for defined stations.')
        selected_stn.loc[nan_loc,vs30_label] = get_vs30_ncm(selected_stn.iloc[nan_loc,list(selected_stn.keys()).index(lat_label)].values.tolist(), 
                                                            selected_stn.iloc[nan_loc,list(selected_stn.keys()).index(lon_label)].values.tolist())
    if len(nan_loc) and vs30_tag == 0:
        print('CreateStation: Fetch OpenSHA Vs30 map for defined stations.')
        selected_stn.loc[nan_loc,vs30_label] = get_site_vs30_from_opensha(selected_stn.iloc[nan_loc,list(selected_stn.keys()).index(lat_label)].values.tolist(), 
                                                                          selected_stn.iloc[nan_loc,list(selected_stn.keys()).index(lon_label)].values.tolist())
    
    # Get zTR
    if zTR_label in selected_stn.keys():
        tmp = selected_stn.iloc[:,list(selected_stn.keys()).index(zTR_label)].values.tolist()
        if len(tmp):
            nan_loc = [x[0] for x in np.argwhere(np.isnan(tmp)).tolist()]
        else:
            nan_loc = []
    else:
        nan_loc = list(range(len(selected_stn.index)))
    if len(nan_loc) and zTR_tag == 0:
            print('CreateStation: Interpolating global depth to rock map for defined stations.')
            selected_stn.loc[nan_loc, zTR_label] = [max(0,x) for x in get_zTR_global(selected_stn.iloc[nan_loc,list(selected_stn.keys()).index(lat_label)].values.tolist(), 
                                                                                     selected_stn.iloc[nan_loc,list(selected_stn.keys()).index(lon_label)].values.tolist())]
    elif len(nan_loc) and zTR_tag == 1:
            print('CreateStation: Interpolating depth to rock map from National Crustal Model.')
            selected_stn.loc[nan_loc, zTR_label] = [max(0,x) for x in get_zTR_ncm(selected_stn.iloc[nan_loc,list(selected_stn.keys()).index(lat_label)].values.tolist(), 
                                                                                  selected_stn.iloc[nan_loc,list(selected_stn.keys()).index(lon_label)].values.tolist())]
    elif len(nan_loc):
        print('CreateStation: Default zore depth to rock for sites missing the data.')
        selected_stn[zTR_label] = [0.0 for x in range(len(selected_stn.index))]
        

    # rename column headers to standard keywords
    selected_stn.rename(columns={lat_label: 'Latitude', lon_label: 'Longitude', vs30_label: 'Vs30',
                                 z1p0_label: 'z1p0', z2p5_label: 'z2p5', zTR_label: 'DepthToRock'})
    if soil_flag:
        selected_stn.rename(columns={soil_model_label: 'Model'})

    # get soil model 
    if soil_flag:
        # get soil_model
        soil_model = selected_stn.iloc[:, list(selected_stn.keys()).index('Model')].values.tolist()
        # elastic istropic model
        row_EI = [i for i, x in enumerate(soil_model) if x == 'EI']
        # Borja & Amier model
        row_BA = [i for i, x in enumerate(soil_model) if x == 'BA']
        # User-defined model
        row_USER = [i for i, x in enumerate(soil_model) if x == 'USER']
        if len(row_EI):
            cur_param_list = ['Den']
            for cur_param in cur_param_list:
                if cur_param in selected_stn.keys():
                    tmp = selected_stn.iloc[row_EI,list(selected_stn.keys()).index(cur_param)].values.tolist()
                    if len(tmp):
                        nan_loc = [x[0] for x in np.argwhere(np.isnan(tmp)).tolist()]
                    else:
                        nan_loc = []
                else:
                    nan_loc = list(range(len(row_EI)))
                if len(nan_loc):
                    selected_stn.loc[row_EI,cur_param] = [get_soil_model_ei(param=cur_param) for x in range(len(row_EI))]
        
        if len(row_BA):
            cur_param_list = ['Su_rat', 'Den', 'h/G', 'm', 'h0', 'chi']
            for cur_param in cur_param_list:
                if cur_param in selected_stn.keys():
                    tmp = selected_stn.iloc[row_BA,list(selected_stn.keys()).index(cur_param)].values.tolist()
                    if len(tmp):
                        nan_loc = [x[0] for x in np.argwhere(np.isnan(tmp)).tolist()]
                    else:
                        nan_loc = []
                else:
                    nan_loc = list(range(len(row_BA)))
                if len(nan_loc):
                    selected_stn.loc[row_BA,cur_param] = [get_soil_model_ba(param=cur_param) for x in range(len(row_BA))]

        user_param_list = []
        if len(row_USER):
            if soil_user_fun is None:
                print('CreateStation: no fetching is conducted for the User soil model- please ensure all needed parameters are defined.')
                for cur_param in list(selected_stn.keys()):
                    if cur_param not in ['Longitude', 'Latitude', 'Vs30', 'DepthToRock', 'z1p0', 
                                         'z2p5', 'Model', 'Su_rat', 'Den', 'h/G', 'm', 'h0', 'chi']:
                        user_param_list.append(cur_param)
            else:
                selected_stn = get_soil_model_user(selected_stn, soil_user_fun)
                user_param_list = list(selected_stn.keys())
                for cur_param in user_param_list:
                    if cur_param in ['Longitude', 'Latitude', 'Vs30', 'DepthToRock', 'z1p0', 
                                     'z2p5', 'Model', 'Su_rat', 'Den', 'h/G', 'm', 'h0', 'chi']:
                        user_param_list.pop(user_param_list.index(cur_param))

    for ind in tqdm(range(selected_stn.shape[0]), desc='Stations'):
        stn = selected_stn.iloc[ind,:]
        stn_id = stn.index
    # for stn_id, stn in selected_stn.iterrows():
        # Creating a Station object
        STN.append(Station(
            stn['Longitude'], stn['Latitude'],
            stn.get('Vs30', 760.0), stn.get('z2p5', 9.0)
        ))
        # Collecting station data
        tmp = {
            'ID': stn_id,
            'Longitude': stn['Longitude'],
            'Latitude': stn['Latitude']
        }

        if stn.get('Vs30'):
            tmp.update({'Vs30': stn.get('Vs30')})
        else:
            tmp.update({'Vs30': 760.0})
            """
            if vs30_tag == 1:
                tmp.update({'Vs30': get_vs30_global([stn[lat_label]], [stn[lon_label]])[0]})
            elif vs30_tag == 2:
                tmp.update({'Vs30': get_vs30_thompson([stn[lat_label]], [stn[lon_label]])[0]})
            elif vs30_tag == 3:
                tmp.update({'Vs30': get_vs30_ncm([stn[lat_label]], [stn[lon_label]])[0]})
            elif vs30_tag == 0:
                tmp.update({'Vs30': get_site_vs30_from_opensha([stn[lat_label]], [stn[lon_label]])[0]})
            """

        if stn.get('z1pt0'):
            tmp.update({'z1pt0': stn.get('z1pt0')})
        else:
            if z1_tag==1:
                tmp.update({'z1pt0': get_z1(tmp['Vs30'])})
            elif z1_tag==2:
                z1pt0 = get_site_z1pt0_from_opensha(tmp['Latitude'], tmp['Longitude'])
                if np.isnan(z1pt0):
                    z1pt0 = get_z1(tmp.get('Vs30'))
                tmp.update({'z1pt0': z1pt0})
            elif z1_tag == 0:
                z1pt0 = get_z1(tmp.get('Vs30'))
                tmp.update({'z1pt0': z1pt0})

        if stn.get('z2pt5'):
            tmp.update({'z2pt5': stn.get('z2pt5')})
        else:
            if z25_tag==1:
                tmp.update({'z2pt5': get_z25(tmp['z1pt0'])})
            elif z25_tag==2:
                z2pt5 = get_site_z2pt5_from_opensha(tmp['Latitude'], tmp['Longitude'])
                if np.isnan(z2pt5):
                    z2pt5 = get_z25(tmp['z1pt0'])
                tmp.update({'z2pt5': z2pt5})
            elif z25_tag ==0:
                z2pt5 = get_z25(tmp['z1pt0'])
                tmp.update({'z2pt5': z2pt5})

        if stn.get('DepthToRock'):
            tmp.update({'DepthToRock': stn.get('DepthToRock')})
        else:
            #tmp.update({'zTR': max(0,get_zTR_global([stn[lat_label]], [stn[lon_label]])[0])})
            tmp.update({'DepthToRock': 0.0})

        if soil_flag:
            tmp.update({'Model': stn.get('Model', 'EI')})
            for cur_param in ['Su_rat', 'Den', 'h/G', 'm', 'h0', 'chi']+user_param_list:
                tmp.update({cur_param: stn.get(cur_param, None)})
        
        stn_file['Stations'].append(tmp)
        #stn_file['Stations'].append({
        #    'ID': stn_id,
        #    'Longitude': stn[lon_label],
        #    'Latitude': stn[lat_label],
        #    'Vs30': stn.get(vs30_label, 760.0),
        #    'z2.5': stn.get(z2p5_label, 9.0)
        #})
    # Saving data to the output file
    df_csv = {
                'ID': [id for id, _ in enumerate(stn_file['Stations'])],
                'lon': [x['Longitude'] for x in stn_file['Stations']],
                'lat': [x['Latitude'] for x in stn_file['Stations']],
                'vs30': [x.get('Vs30',760) for x in stn_file['Stations']],
                'z1pt0': [x.get('z1pt0',9) for x in stn_file['Stations']],
                'z2pt5': [x.get('z2pt5',12) for x in stn_file['Stations']],
                'vs30measured': [x.get('vs30measured',0) for x in stn_file['Stations']],
                'DepthToRock': [x.get('DepthToRock',0) for x in stn_file['Stations']]
            }
            # no backarc by default
    if stn_file['Stations'][0].get('backarc',None):
        df_csv.update({
            'backarc': [x.get('backarc') for x in stn_file['Stations']]
        })
    pd.DataFrame.from_dict(df_csv).to_csv(output_file, index=False)
    # if output_file:
    #     if '.json' in output_file:
    #         with open(output_file, 'w') as f:
    #             json.dump(stn_file, f, indent=2)
    #     if 'OpenQuake' in output_file:
    #         df_csv = {
    #             'ID': [id for id, _ in enumerate(stn_file['Stations'])],
    #             'lon': [x['Longitude'] for x in stn_file['Stations']],
    #             'lat': [x['Latitude'] for x in stn_file['Stations']],
    #             'vs30': [x.get('Vs30',760) for x in stn_file['Stations']],
    #             'z1pt0': [x.get('z1pt0',9) for x in stn_file['Stations']],
    #             'z2pt5': [x.get('z2pt5',12) for x in stn_file['Stations']],
    #             'vs30measured': [x.get('vs30measured',0) for x in stn_file['Stations']],
    #             'DepthToRock': [x.get('DepthToRock',0) for x in stn_file['Stations']]
    #         }
    #         # no backarc by default
    #         if stn_file['Stations'][0].get('backarc',None):
    #             df_csv.update({
    #                 'backarc': [x.get('backarc') for x in stn_file['Stations']]
    #             })
    #         pd.DataFrame.from_dict(df_csv).to_csv(output_file, index=False)
    #     if 'SiteData' in output_file:
    #         df_csv = {
    #             'id': list(range(len(stn_file['Stations']))),
    #             'Longitude': [x['Longitude'] for x in stn_file['Stations']],
    #             'Latitude': [x['Latitude'] for x in stn_file['Stations']],
    #             'Vs30': [x.get('Vs30',760) for x in stn_file['Stations']],
    #             'DepthToRock': [x.get('DepthToRock',0) for x in stn_file['Stations']],
    #             'Model': [x.get('Model','EI') for x in stn_file['Stations']]
    #         }
    #         for cur_param in ['Su_rat', 'Den', 'h/G', 'm', 'h0', 'chi']+user_param_list:
    #             df_csv.update({cur_param: [x.get(cur_param) for x in stn_file['Stations']]})
    #         pd.DataFrame.from_dict(df_csv).to_csv(output_file, index=False)
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
    Compute z1 based on the prediction equation by Chiou and Youngs (2013) (unit of vs30 is meter/second and z1 is meter)
    """

    z1 = np.exp(-7.15 / 4.0 * np.log((vs30 ** 4 + 571.0 ** 4) / (1360.0 ** 4 + 571.0 ** 4)))
    # return
    return z1


def get_z25(z1):
    """
    Compute z25 based on the prediction equation by Campbell and Bozorgnia (2013)
    """
    z25 = 0.748 + 2.218 * z1
    # return
    return z25

def get_z25fromVs(vs):
    """
    Compute z25 (m) based on the prediction equation 33 by Campbell and Bozorgnia (2014)
    Vs is m/s
    """
    z25 = (7.089 - 1.144 * np.log(vs))*1000
    # return
    return z25


def get_zTR_global(lat, lon):
    """
    Interpolate depth to rock at given latitude and longitude
    Input:
        lat: list of latitude
        lon: list of longitude
    Output:
        zTR: list of zTR
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


def get_zTR_ncm(lat, lon):
    """
    Call USGS National Crustal Model services for zTR
    https://earthquake.usgs.gov/nshmp/ncm
    Input:
        lat: list of latitude
        lon: list of longitude
    Output:
        zTR: list of depth to bedrock
    """
    import requests

    zTR = []

    # Looping over sites
    for cur_lat, cur_lon in zip(lat, lon):
        url_geology = 'https://earthquake.usgs.gov/ws/nshmp/ncm/ws/nshmp/ncm/geologic-framework?location={}%2C{}'.format(cur_lat,cur_lon)
        # geological data (depth to bedrock)
        r1 = requests.get(url_geology)
        cur_res = r1.json()
        if not cur_res['response']['results'][0]['profiles']:
            # the current site is out of the available range of NCM (Western US only, 06/2021)
            # just append 0.0 to zTR
            print('CreateStation: Warning in NCM API call - could not get the site geological data and approximate 0.0 for zTR for site {}, {}'.format(cur_lat,cur_lon))
            zTR.append(0.0)
            continue
        else:
            # get the top bedrock data
            zTR.append(abs(cur_res['response']['results'][0]['profiles'][0]['top']))
    # return
    return zTR


def get_vsp_ncm(lat, lon, depth):
    """
    Call USGS National Crustal Model services for Vs30 profile
    https://earthquake.usgs.gov/nshmp/ncm
    Input:
        lat: list of latitude
        lon: list of longitude
        depth: [depthMin, depthInc, depthMax]
    Output:
        vsp: list of shear-wave velocity profile
    """
    import requests

    vsp = []
    depthMin, depthInc, depthMax = [abs(x) for x in depth]

    # Looping over sites
    for cur_lat, cur_lon in zip(lat, lon):
        url_geophys = 'https://earthquake.usgs.gov/ws/nshmp/ncm/ws/nshmp/ncm/geophysical?location={}%2C{}&depths={}%2C{}%2C{}'.format(cur_lat,cur_lon,depthMin,depthInc,depthMax)
        r1 = requests.get(url_geophys)
        cur_res = r1.json()
        if cur_res['status'] == 'error':
            # the current site is out of the available range of NCM (Western US only, 06/2021)
            # just append -1 to zTR
            print('CreateStation: Warning in NCM API call - could not get the site geopyhsical data.')
            vsp.append([])
            continue
        else:
            # get vs30 profile
            vsp.append([abs(x) for x in cur_res['response']['results'][0]['profile']['vs']])
    if len(vsp) == 1:
        vsp = vsp[0]
    # return
    return vsp


def compute_vs30_from_vsp(depthp, vsp):
    """
    Compute the Vs30 given the depth and Vs profile
    Input:
        depthp: list of depth for Vs profile
        vsp: Vs profile
    Output:
        vs30p: average VS for the upper 30-m depth
    """
    # Computing the depth interval
    delta_depth = np.diff([0] + depthp)
    # Computing the wave-travel time
    delta_t = [x / y for x,y in zip(delta_depth, vsp)]
    # Computing the Vs30
    vs30p = 30.0 / np.sum(delta_t)
    # return
    return vs30p


def get_vs30_ncm(lat, lon):
    """
    Fetch Vs30 at given latitude and longitude from NCM
    Input:
        lat: list of latitude
        lon: list of longitude
    Output:
        vs30: list of vs30
    """
    # Depth list (in meter)
    depth = [1.0, 1.0, 30.0]
    depthp = np.arange(depth[0], depth[2] + 1.0, depth[1])
    # Getting Vs profile
    vsp = [get_vsp_ncm([x], [y], depth) for x,y in zip(lat, lon)]
    # Computing Vs30
    vs30 = []
    for cur_vsp in vsp:
        if cur_vsp:
            vs30.append(compute_vs30_from_vsp(depthp, cur_vsp))
        else:
            print('CreateStation: Warning - approximate 760 m/s for sites not supported by NCM (Western US).')
            vs30.append(760.0)
    # return
    return vs30


def get_soil_model_ba(param=None):
    """
    Get modeling parameters for Borja and Amies 1994 J2 model
    Currently just assign default values
    Can be extended to have input soil properties to predict this pararmeters
    """
    su_rat = 0.26
    density = 2.0
    h_to_G = 1.0
    m = 1.0
    h0 = 0.2
    chi = 0.0

    if param == 'Su_rat':
        res = su_rat
    elif param == 'Den':
        res = density
    elif param == 'h/G':
        res = h_to_G
    elif param == 'm':
        res = m
    elif param == 'h0':
        res = h0
    elif param == 'chi':
        res = chi
    else:
        res = None

    return res


def get_soil_model_ei(param=None):
    """
    Get modeling parameters for elastic isotropic
    Currently just assign default values
    Can be extended to have input soil properties to predict this pararmeters
    """
    density = 2.0

    if param == 'Den':
        res = density
    else:
        res = None

    return res


def get_soil_model_user(df_stn, model_fun):

    # check if mode_fun exists
    import os, sys, importlib
    if not os.path.isfile(model_fun):
        print('CreateStation.get_soil_model_user: {} is not found.'.format(model_fun))
        return df_stn, []

    # try to load the model file
    from pathlib import Path
    try:
        path_model_fun = Path(model_fun).resolve()
        sys.path.insert(0, str(path_model_fun.parent)+'/')
        # load the function
        user_model= importlib.__import__(path_model_fun.name[:-3], globals(), locals(), [], 0)
    except:
        print('CreateStation.get_soil_model_user: {} cannot be loaded.'.format(model_fun))
        return df_stn

    # try to load the standard function: soil_model_fun(site_info=None)
    try:
        soil_model = user_model.soil_model
    except:
        print('CreateStation.get_soil_model_user: soil_model is nto found in {}.'.format(model_fun))
        return df_stn
    
    # get the parameters from soil_model_fun
    try:
        df_stn_new = soil_model(site_info=df_stn)
    except:
        print('CreateStation.get_soil_model_user: error in soil_model_fun(site_info=None).')
        return df_stn

    return df_stn_new