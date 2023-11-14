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

import os, time
import subprocess
import json
import random
import numpy as np
import pandas as pd
import socket
if 'stampede2' not in socket.gethostname():
	from FetchOpenSHA import *

def load_earthquake_rupFile(scenario_info, rupFilePath):
    # Getting earthquake rupture forecast data
    source_type = scenario_info['EqRupture']['Type']
    try:
        with open(rupFilePath, 'r') as f:
            user_scenarios = json.load(f)
    except:
        print('CreateScenario: source file {} not found.'.format(rupFilePath))
        return {}
    # number of features (i.e., ruptures)
    num_scenarios = len(user_scenarios.get('features',[]))
    if num_scenarios < 1:
        print('CreateScenario: source file is empty.')
        return {}
    # If there is a filter
    if scenario_info["Generator"].get("method", None) == "ScenarioSpecific":
        SourceIndex = scenario_info["Generator"].get("SourceIndex", None)
        RupIndex = scenario_info['Generator'].get('RuptureIndex', None)
        if (SourceIndex is None) or (RupIndex is None):
            print("Both SourceIndex and RuptureIndex are needed for"\
                  "ScenarioSpecific analysis")
            return
        rups_to_run = []
        for ind in range(len(user_scenarios.get('features'))):
            cur_rup = user_scenarios.get('features')[ind]
            cur_id_source = cur_rup.get('properties').get('Source', None)
            if cur_id_source != SourceIndex:
                continue
            cur_id_rupture = cur_rup.get('properties').get('Rupture', None)
            if cur_id_rupture == RupIndex:
                rups_to_run.append(ind)
                break
    elif scenario_info["Generator"].get("method", None) == "UserSelection":
        rup_filter = scenario_info["Generator"].get("filter", None)
        if rup_filter is None or len(rup_filter)==0:
            rups_to_run = list(range(0, num_scenarios))
        else:
            rups_requested = []
            for rups in rup_filter.split(','):
                if "-" in rups:
                    asset_low, asset_high = rups.split("-")
                    rups_requested += list(range(int(asset_low), int(asset_high)+1))
                else:
                    rups_requested.append(int(rups))
            rups_requested = np.array(rups_requested)
            rups_available = list(range(0, num_scenarios))
            rups_to_run = rups_requested[
                np.where(np.in1d(rups_requested, rups_available))[0]]
        # Select all
    elif scenario_info["Generator"].get("method", None) == "Subsampling":
        rups_to_run = list(range(0, num_scenarios))
    else:
        print(f'The scenario selection method {scenario_info["Generator"].get("method", None)} is not available')
        return {}
        
    # get rupture and source ids
    scenario_data = {}
    if source_type == "ERF":
        # source model
        source_model = scenario_info['EqRupture']['Model']
        for rup_tag in rups_to_run:
            cur_rup = user_scenarios.get('features')[rup_tag]
            cur_id_source = cur_rup.get('properties').get('Source', None)
            cur_id_rupture = cur_rup.get('properties').get('Rupture', None)
            scenario_data.update({rup_tag: {
                'Type': source_type,
                'RuptureForecast': source_model,
                'Name': cur_rup.get('properties').get('Name', ""),
                'Magnitude': cur_rup.get('properties').get('Magnitude', None),
                'MeanAnnualRate': cur_rup.get('properties').get('MeanAnnualRate', None),
                'SourceIndex': cur_id_source,
                'RuptureIndex': cur_id_rupture,
                'SiteSourceDistance': cur_rup.get('properties').get('Distance', None),
                'SiteRuptureDistance': cur_rup.get('properties').get('DistanceRup', None)
            }})
    elif source_type == "PointSource":
        for rup_tag in rups_to_run:
            try:
                cur_rup = user_scenarios.get('features')[rup_tag]
                magnitude = cur_rup.get('properties')['Magnitude']
                location = cur_rup.get('properties')['Location']
                average_rake = cur_rup.get('properties')['AverageRake']
                average_dip = cur_rup.get('properties')['AverageDip']
                scenario_data.update({0: {
                    'Type': source_type,
                    'Magnitude': magnitude,
                    'Location': location,
                    'AverageRake': average_rake,
                    'AverageDip': average_dip
                }})
            except:
                print('Please check point-source inputs.')
    
    # return
    return scenario_data


def load_earthquake_scenarios(scenario_info, stations, dir_info):

    # Number of scenarios
    source_num = scenario_info.get('Number', 1)
    # sampling method
    samp_method = scenario_info['EqRupture'].get('Sampling','Random')
    # source model
    source_model = scenario_info['EqRupture']['Model']
    eq_source = getERF(scenario_info)
    # Getting earthquake rupture forecast data
    source_type = scenario_info['EqRupture']['Type']
    # Collecting all sites
    lat = []
    lon = []
    for s in stations['Stations']:
        lat.append(s['Latitude'])
        lon.append(s['Longitude'])
    # load scenario file
    user_scenario_file = os.path.join(dir_info.get('Input'), scenario_info.get('EqRupture').get('UserScenarioFile'))
    try:
        with open(user_scenario_file, 'r') as f:
            user_scenarios = json.load(f)
    except:
        print('CreateScenario: source file {} not found.'.format(user_scenario_file))
        return {}
    # number of features (i.e., ruptures)
    num_scenarios = len(user_scenarios.get('features',[]))
    if num_scenarios < 1:
        print('CreateScenario: source file is empty.')
        return {}
    # get rupture and source ids
    scenario_data = {}
    for rup_tag in range(num_scenarios):
        cur_rup = user_scenarios.get('features')[rup_tag]
        cur_id_source = cur_rup.get('properties').get('Source', None)
        cur_id_rupture = cur_rup.get('properties').get('Rupture', None)
        if cur_id_rupture is None or cur_id_source is None:
            print('CreateScenario: rupture #{} does not have valid source/rupture ID - skipped.'.format(rup_tag))
            continue
        cur_source, cur_rupture = get_source_rupture(eq_source, cur_id_source, cur_id_rupture)
        scenario_data.update({rup_tag: {
            'Type': source_type,
            'RuptureForecast': source_model,
            'Name': str(cur_source.getName()),
            'Magnitude': float(cur_rupture.getMag()),
            'MeanAnnualRate': float(cur_rupture.getMeanAnnualRate(eq_source.getTimeSpan().getDuration())),
            'SourceIndex': cur_id_source,
            'RuptureIndex': cur_id_rupture,
            'SiteSourceDistance': get_source_distance(eq_source, cur_id_source, lat, lon),
            'SiteRuptureDistance': get_rupture_distance(eq_source, cur_id_source, cur_id_rupture, lat, lon)
        }})
    
    # return
    return scenario_data
    

def create_earthquake_scenarios(scenario_info, stations, work_dir, openquakeSiteFile = None):

    # # Number of scenarios
    # source_num = scenario_info.get('Number', 1)
    # if source_num == 'All':
    #     # Large number to consider all sources in the ERF
    #     source_num = 10000000
    out_dir = os.path.join(work_dir,"Output")
    if scenario_info['Generator'] == 'Simulation':
        # TODO:
        print('Physics-based earthquake simulation is under development.')
        return 1
    # Searching earthquake ruptures that fulfill the request
    elif scenario_info['Generator'] == 'Selection':
        # Collecting all possible earthquake scenarios
        lat = []
        lon = []
        for s in stations['Stations']:
            lat.append(s['Latitude'])
            lon.append(s['Longitude'])
        # Reference location
        mlat = np.mean(lat)
        mlon = np.mean(lon)
        ref_station = [mlat, mlon]
        # Getting earthquake rupture forecast data
        source_type = scenario_info['EqRupture']['Type']
        t_start = time.time()
        if source_type == 'ERF':
            if 'SourceIndex' in scenario_info['EqRupture'].keys() and 'RuptureIndex' in scenario_info['EqRupture'].keys():
                source_model = scenario_info['EqRupture']['Model']
                eq_source = getERF(scenario_info)
                # check source index list and rupture index list
                if type(scenario_info['EqRupture']['SourceIndex']) == int:
                    source_index_list = [scenario_info['EqRupture']['SourceIndex']]
                else:
                    source_index_list = scenario_info['EqRupture']['SourceIndex']
                if type(scenario_info['EqRupture']['RuptureIndex']) == int:
                    rup_index_list = [scenario_info['EqRupture']['RuptureIndex']]
                else:
                    rup_index_list = scenario_info['EqRupture']['RuptureIndex']
                if not(len(source_index_list) == len(rup_index_list)):
                    print('CreateScenario: source number {} should be matched by rupture number {}'.format(len(source_index_list),len(rup_index_list)))
                    return dict()
                # loop over all scenarios
                scenario_data = dict()
                for i in range(len(source_index_list)):
                    cur_source_index = source_index_list[i]
                    cur_rup_index = rup_index_list[i]
                    distToSource = get_source_distance(eq_source, cur_source_index, lat, lon)
                    scenario_data.update({i: {
                        'Type': source_type,
                        'RuptureForecast': source_model,
                        'SourceIndex': cur_source_index,
                        'RuptureIndex': cur_rup_index,
                        'SiteSourceDistance': distToSource,
                        'SiteRuptureDistance': get_rupture_distance(eq_source, cur_source_index, cur_rup_index, lat, lon)
                    }})
                return scenario_data
            else:
                source_model = scenario_info['EqRupture']['Model']
                source_name = scenario_info['EqRupture'].get('Name', None)
                min_M = scenario_info['EqRupture'].get('min_Mag', 5.0)
                max_M = scenario_info['EqRupture'].get('max_Mag', 9.0)
                max_R = scenario_info['EqRupture'].get('max_Dist', 1000.0)
                eq_source = getERF(scenario_info)
                erf_data = export_to_json(eq_source, ref_station, outfile = os.path.join(out_dir,'RupFile.json'), \
                                        EqName = source_name, minMag = min_M, \
                                        maxMag = max_M, maxDistance = max_R, \
                                        )
                # Parsing data
                # feat = erf_data['features']
                # """
                # tag = []
                # for i, cur_f in enumerate(feat):
                #     if source_name and (source_name not in cur_f['properties']['Name']):
                #         continue
                #     if min_M > cur_f['properties']['Magnitude']:
                #         continue
                #     tag.append(i)
                # # Abstracting desired ruptures
                # s_tag = random.sample(tag, min(source_num, len(tag)))
                # """
                # t_start = time.time()
                # s_tag = sample_scenarios(rup_info=feat, sample_num=source_num, sample_type=samp_method, source_name=source_name, min_M=min_M)
                # print('CreateScenario: scenarios sampled {0} sec'.format(time.time() - t_start))
                # #erf_data['features'] = list(feat[i] for i in s_tag)
                # erf_data['features'] = [feat[i] for i in range(source_num)]
                # scenario_data = dict()
                # t_start = time.time()
                # for i, rup in enumerate(erf_data['features']):
                #     scenario_data.update({i: {
                #         'Type': source_type,
                #         'RuptureForecast': source_model,
                #         'Name': rup['properties']['Name'],
                #         'Magnitude': rup['properties']['Magnitude'],
                #         'MeanAnnualRate': rup['properties']['MeanAnnualRate'],
                #         'SourceIndex': rup['properties']['Source'],
                #         'RuptureIndex': rup['properties']['Rupture'],
                #         'SiteSourceDistance': get_source_distance(eq_source, rup['properties']['Source'], lat, lon),
                #         'SiteRuptureDistance': get_rupture_distance(eq_source, rup['properties']['Source'], rup['properties']['Rupture'], lat, lon)
                #     }})
                # print('CreateScenario: scenarios collected {0} sec'.format(time.time() - t_start))
                # # Cleaning tmp outputs
                # del erf_data
        elif source_type == 'PointSource':
            # Export to a geojson format RupFile.json
            outfile = os.path.join(out_dir,'RupFile.json')
            pointSource_data = {"type": "FeatureCollection"}
            feature_collection = []
            newRup = {
                    'type': "Feature",
                    "properties":{
                    'Type': source_type,
                    'Magnitude': scenario_info['EqRupture']['Magnitude'],
                    'Location': scenario_info['EqRupture']['Location'],
                    'AverageRake': scenario_info['EqRupture']['AverageRake'],
                    'AverageDip': scenario_info['EqRupture']['AverageDip']}
            }
            newRup['geometry'] = dict()
            newRup['geometry'].update({'type': 'Point'})
            newRup['geometry'].update({'coordinates': [scenario_info['EqRupture']['Location']['Longitude'], scenario_info['EqRupture']['Location']['Latitude']]})
            feature_collection.append(newRup)
            pointSource_data.update({'features':feature_collection})
            if outfile is not None:
                print('The collected point source ruptures are saved in {}'.format(outfile))
                with open(outfile, 'w') as f:
                    json.dump(pointSource_data, f, indent=2)
        elif source_type=='oqSourceXML':
            import FetchOpenQuake
            siteFile = os.path.join(work_dir,'Input',openquakeSiteFile)
            FetchOpenQuake.export_rupture_to_json(scenario_info, mlon, mlat, siteFile, work_dir)
        print('CreateScenario: all scenarios configured {0} sec'.format(time.time() - t_start))
    # return
    return 


def sample_scenarios(rup_info=[], sample_num=1, sample_type='Random', source_name=None, min_M=0.0):

    if len(rup_info) == 0:
        print('CreateScenario.sample_scenarios: no available scenario provided - please relax earthquake filters.')
        return []

    feat = rup_info
    tag = []
    for i, cur_f in enumerate(feat):
        if source_name and (source_name not in cur_f['properties']['Name']):
            continue
        if min_M > cur_f['properties']['Magnitude']:
            continue
        tag.append(i)
    
    if sample_type == 'Random':
        s_tag = random.sample(tag, min(sample_num, len(tag)))
    
    elif sample_type == 'MAF':
        # maf list
        maf_list = [feat[x]['properties']['MeanAnnualRate'] for x in tag]
        # normalize maf list
        sum_maf = np.sum(maf_list)
        maf_list_n = [x/sum_maf for x in maf_list]
        # get sample
        s_tag = np.random.choice(tag, sample_num, p=maf_list_n).tolist()

    else:
        print('CreateScenario.sample_scenarios: please specify a sampling method.')
        s_tag = []

    # return
    return s_tag

def create_wind_scenarios(scenario_info, stations, data_dir):

    # Number of scenarios
    source_num = scenario_info.get('Number', 1)
    # Directly defining earthquake ruptures
    if scenario_info['Generator'] == 'Simulation':
        # Collecting site locations
        lat = []
        lon = []
        for s in stations['Stations']:
            lat.append(s['Latitude'])
            lon.append(s['Longitude'])
        # Save Stations.csv
        df = pd.DataFrame({
            'lat': lat,
            'lon': lon
        })
        df.to_csv(data_dir + 'Stations.csv', index = False, header = False)
        # Save Lat_w.csv
        lat_w = np.linspace(min(lat) - 0.5, max(lat) + 0.5, 100)
        df = pd.DataFrame({'lat_w': lat_w})
        df.to_csv(data_dir + 'Lat_w.csv', index = False, header = False)
        # Parsing Terrain info
        df = pd.read_csv(data_dir + scenario_info['Terrain']['Longitude'],
                         header = None, index_col = None)
        df.to_csv(data_dir + 'Long_wr.csv', header = False, index = False)
        df = pd.read_csv(data_dir + scenario_info['Terrain']['Latitude'],
                         header = None, index_col = None)
        df.to_csv(data_dir + 'Lat_wr.csv', header = False, index = False)
        df = pd.read_csv(data_dir + scenario_info['Terrain']['Size'],
                         header = None, index_col = None)
        df.to_csv(data_dir + 'wr_sizes.csv', header = False, index = False)
        df = pd.read_csv(data_dir + scenario_info['Terrain']['z0'],
                         header = None, index_col = None)
        df.to_csv(data_dir + 'z0r.csv', header = False, index = False)
        # Parsing storm properties
        param = []
        param.append(scenario_info['Storm']['Landfall']['Latitude'])
        param.append(scenario_info['Storm']['Landfall']['Longitude'])
        param.append(scenario_info['Storm']['LandingAngle'])
        param.append(scenario_info['Storm']['Pressure'])
        param.append(scenario_info['Storm']['Speed'])
        param.append(scenario_info['Storm']['Radius'])
        df = pd.DataFrame({'param': param})
        df.to_csv(data_dir + 'param.csv', index = False, header = False)
        df = pd.read_csv(data_dir + scenario_info['Storm']['Track'],
                         header = None, index_col = None)
        df.to_csv(data_dir + 'Track.csv', header = False, index = False)
        # Saving del_par.csv
        del_par = [0, 0, 0] # default
        df =pd.DataFrame({'del_par': del_par})
        df.to_csv(data_dir + 'del_par.csv', header = False, index = False)
        # Parsing resolution data
        delta_p = [1000., scenario_info['Resolution']['DivRad'], 1000000.]
        delta_p.extend([0., scenario_info['Resolution']['DivDeg'], 360.])
        delta_p.extend([scenario_info['MeasureHeight'], 10,
                       scenario_info['MeasureHeight']])
        df = pd.DataFrame({'delta_p': delta_p})
        df.to_csv(data_dir + 'delta_p.csv', header = False, index = False)
    else:
        print('Currently only supporting Simulation generator.')
