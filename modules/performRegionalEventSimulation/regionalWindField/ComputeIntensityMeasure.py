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

import os
import subprocess
import sys
import json
import copy
import shutil
import multiprocessing as mp
import numpy as np
import pandas as pd
from WindFieldSimulation import *

def run_model(scen, p, t, path_perturb, feat_perturb, res_mp):

    model = LinearAnalyticalModel_SnaikiWu_2017(cyclone_param = p, storm_track = t)
    if scen['Terrain']:
        model.add_reference_terrain(scen['Terrain'])
    model.set_cyclone_mesh(scen['StormMesh'])
    model.set_measure_height(scen['MeasureHeight'])
    model.define_track(scen['TrackSimu'])
    model.add_stations(scen['StationList'])
    delta_path = (np.random.rand(3) - 0.5) * path_perturb
    delta_feat = np.array(p[3:6]) + (np.random.rand(3) - 0.5) * feat_perturb
    # this just an engineering judgement that the pressure difference, moving speed, and max-wind-speed radius
    # should not be less than 0.0 in the value.
    delta_feat[delta_feat < 0.0] = 0.0
    print('dLatitude, dLongtitude, dAngle = ', delta_path)
    print('dP, v, Rmax = ', delta_feat)
    model.set_delta_path(delta_path)
    model.set_delta_feat(delta_feat)
    model.compute_wind_field()
    res_mp.append(model.get_station_data())

def simulate_storm(scenarios, event_info, model_type):

    if (model_type == 'LinearAnalytical'):
        num_per_site = event_info['NumberPerSite']
        if (num_per_site == 1):
            path_perturb = np.zeros(3)
            feat_perturb = np.zeros(3)
        else:
            if (len(event_info.get('Perturbation', [])) != 6): 
                print('ComputeIntensityMeasure: Perturbation should have a size of 6.')
                path_perturb = np.array([0.5, 0.5, 90.0])
                feat_perturb = np.array([10.0, 10.0, 10.0])
                print('ComputeIntensityMeasure: [1.0, 1.0, 90.0, 10.0, 10.0, 10.0] is used for perturbations.')
            else:
                path_perturb = np.array(event_info['Perturbation'][0:3])
                feat_perturb = np.array(event_info['Perturbation'][3:6])
        for i in range(len(scenarios)):
            if (i == 1):
                print('ComputeIntensityMeasure: currently supporting single scenario simulation only.')
                return -1
            cur_scen = scenarios[i]
            param = cur_scen['CycloneParam']
            track = cur_scen['StormTrack']
            np.random.seed(100)
            # parallel
            with mp.Manager() as manager:
                res_mp = manager.list([])
                proc_list = []
                for k in range(num_per_site):
                    proc = mp.Process(target = run_model,
                        args = (cur_scen, param, track, path_perturb, feat_perturb, res_mp))
                    proc_list.append(proc)
                for k in range(num_per_site):
                    proc = proc_list[k]
                    proc.start()
                for k in range(num_per_site):
                    proc = proc_list[k]
                    proc.join()
                # extract data
                res = [x for x in res_mp]
                
    else:
        print('ComputeIntensityMeasure: currently only supporting LinearAnalytical model')

    # return
    return res


def simulate_storm_cpp(site_info, scenario_info, event_info, model_type, dir_info):

    if (model_type == 'LinearAnalytical'):
        # save configuration file
        input_dir = dir_info['Input']
        output_dir = dir_info['Output']
        config = {
            "Scenario": scenario_info,
            "Event": event_info
        }
        abs_path_config = os.path.abspath(os.path.join(input_dir, 'SimuConfig.json'))
        with open (abs_path_config, "w") as f:
            json.dump(config, f)
        # site file
        abs_path_site = os.path.abspath(os.path.join(input_dir, site_info['input_file']))
        # track file
        abs_path_track = os.path.abspath(os.path.join(input_dir, scenario_info['Storm']['Track']))
        # lat_w file
        abs_path_latw = os.path.abspath(os.path.join(input_dir, scenario_info['Storm']['TrackSimu']))
        # terrain file
        if ('Terrain' in scenario_info.keys()):
            abs_path_terrain = os.path.abspath(os.path.join(input_dir, scenario_info['Terrain']))
        else:
            # default terrain z0 = 0.01 everywhere for the defined domain
            abs_path_terrain = os.path.abspath(os.path.join(input_dir, 'DefaultTerrain.geojson'))
            dict_dt = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                        [-90.0, -180.0],
                        [90.0, -180.0],
                        [90.0, 180.0],
                        [-90.0, 180.0]]
                    },
                    "properties": {
                        "z0": 0.01
                    }
                }
                ]
            }
            with open(abs_path_terrain, 'w') as f:
                json.dump(dict_dt, f, indent=2)
                
        # configuring perturbation
        num_per_site = event_info['NumberPerSite']
        if (num_per_site == 1):
            path_perturb = np.zeros(3)
            feat_perturb = np.zeros(3)
        else:
            if (len(event_info.get('Perturbation', [])) != 6): 
                print('ComputeIntensityMeasure: Perturbation should have a size of 6.')
                path_perturb = np.array([0.5, 0.5, 90.0])
                feat_perturb = np.array([10.0, 10.0, 10.0])
                print('ComputeIntensityMeasure: [1.0, 1.0, 90.0, 10.0, 10.0, 10.0] is used for perturbations.')
            else:
                path_perturb = np.array(event_info['Perturbation'][0:3])
                feat_perturb = np.array(event_info['Perturbation'][3:6])
        for i in range(int(scenario_info['Number'])):
            if (i == 1):
                print('ComputeIntensityMeasure: currently supporting single scenario simulation only.')
                return -1
            np.random.seed(100)
            res = []
            # parallel
            pert_list = []
            args_list = []
            odir_list = []
            if sys.platform.startswith('win'):
                windsimu_bin = os.path.dirname(__file__) + '/WindFieldSimulation.exe'
            else:
                windsimu_bin = os.path.dirname(__file__) + '/WindFieldSimulation'
            ## preparing files
            for j in range(num_per_site):
                delta_path = (np.random.rand(3) - 0.5) * path_perturb
                delta_feat = (np.random.rand(3) - 0.5) * feat_perturb
                pert_dict = {
                    "dLatitude": delta_path[0],
                    "dLongitude": delta_path[1],
                    "dAngle": delta_path[2],
                    "dP": delta_feat[0],
                    "dV": delta_feat[1],
                    "dR": delta_feat[2]
                }
                abs_path_pert = os.path.abspath(os.path.join(input_dir, 'Perturbation' + str(j) + '.json'))
                with open(abs_path_pert, "w") as f:
                    json.dump(pert_dict, f)
                print('dLatitude, dLongtitude, dAngle = ', delta_path)
                print('dP, dv, dR = ', delta_feat)
                output_subdir = os.path.abspath(os.path.join(output_dir, 'simu' + str(j)))
                if os.path.exists(output_subdir):
                    shutil.rmtree(output_subdir)
                os.makedirs(output_subdir)
                args = [windsimu_bin, "--config", abs_path_config, "--site", abs_path_site, 
                    "--track", abs_path_track, "--latw", abs_path_latw, "--pert", abs_path_pert,
                    "--terrain", abs_path_terrain, "--z0", output_subdir, 
                    "--output", output_subdir]

                pert_list.append(abs_path_pert)
                args_list.append(args)
                odir_list.append(output_subdir)
            ## running
            procs_list = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for cmd in args_list]
            for proc in procs_list:
                proc.wait()
            ## loading output
            for j in range(num_per_site):
                os.remove(pert_list[j])
                station_res = {
                    'Latitude': [],
                    'Longitude': [],
                    'z0': [],
                    'PWS': {
                        'height': [],
                        'duration': 600.0,
                        'windspeed': []
                    }
                }
                df = pd.read_csv(os.path.join(os.path.abspath(odir_list[j]), 'StationZ0.csv'), header = None, index_col = None)
                station_res['z0'] = list(np.concatenate(df.values.tolist()).flat)
                df = pd.read_csv(os.path.join(os.path.abspath(odir_list[j]), 'MeasureHeight.csv'), header = None, index_col = None)
                station_res['PWS']['height'] = df.values.tolist()[0]
                df = pd.read_csv(os.path.join(os.path.abspath(odir_list[j]), 'MaxWindSpeed.csv'), header = None, index_col = None)
                station_res['PWS']['windspeed'] = df.values.tolist()
                res.append(station_res)
                shutil.rmtree(odir_list[j])
        # house-keeping
        os.remove(abs_path_config)
    else:
        print('ComputeIntensityMeasure: currently only supporting LinearAnalytical model')

    # return
    return res


def convert_wind_speed(event_info, simu_res):

    print('ComputeIntensityMeasure: converting peak wind speed to specificed exposure, measuring height, and gust duration.')

    if ('HAZUS' in event_info['IntensityMeasure']['Type']):
        # Exposure type C: z0 = 0.03
        exposure = 'C'
        # 10-m measuring height
        reference_height = 10.0
        # 3-s gust duration
        gust_duration = 3.0
    else:
        exposure = event_info['IntensityMeasure']['Exposure']
        if exposure not in ['A', 'B', 'C', 'D']:
            print('ComputeIntensityMeasure: the Exposure should be A, B, C, or D.')
            return -1
        gust_duration = event_info['IntensityMeasure']['GustDuration']
        reference_height = event_info['IntensityMeasure']['ReferenceHeight']

    pws_mr = []
    for i in range(len(simu_res)):
        cur_res = simu_res[i]
        # Reading simulation heights
        measure_height = cur_res['PWS']['height']
        # Reading simulated wind speed
        pws_raw = np.array(cur_res['PWS']['windspeed'])
        # Reading z0 in the simulation
        z0_simu = np.array(cur_res['z0'])
        # Reading gust duration in the simulation
        gust_duration_simu = cur_res['PWS']['duration']
        # quick check the size
        if pws_raw.shape[1] != len(measure_height):
            print('ComputeIntensityMeasure: please check the output wind speed results.')
            return -1
        # ASCE 7-16 conversion (Chapter C26)
        # station-wise empirical exponent \alpha
        alpha = 5.65 * (z0_simu ** (-0.133))
        # station-wise gradient height
        zg = 450.0 * (z0_simu ** 0.125)
        # target exposure alpha and graident height
        if (exposure == 'B'):
            alpha_t = 7.0
            zg_t = 365.76
        elif (exposure == 'D'):
            alpha_t = 11.5
            zg_t = 213.36
        else:
            # 'C'
            alpha_t = 9.5
            zg_t = 274.32
        # conversion
        pws_raw = interp_wind_by_height(pws_raw, measure_height, reference_height)
        print(np.max(pws_raw))
        # computing gradient-height wind speed
        pws_tmp = pws_raw * (zg / reference_height) ** (1.0 / alpha)
        # converting exposure
        pws_tmp = pws_tmp * (reference_height / zg_t) ** (1.0 / alpha_t)
        pws = pws_tmp * gust_factor_ESDU(gust_duration_simu, gust_duration)
        print(np.max(pws))        
        # appending to pws_mr
        pws_mr.append(pws)

    print('ComputeIntensityMeasure: wind speed conversion completed.')
    # return
    return pws_mr


def interp_wind_by_height(pws_ip, height_simu, height_ref):
    """
    interp_wind_by_height: interpolating the wind simulation results by the reference height
    """
    num_stat = pws_ip.shape[0]
    pws_op = np.zeros(num_stat)
    for i in range(num_stat):
        pws_op[i] = np.interp(height_ref, height_simu, pws_ip[i, :], left = pws_ip[i, 0], right = pws_ip[i, -1])

    # return
    return pws_op


def gust_factor_ESDU(gd_c, gd_t):
    """
    gust_factor_ESDU: return a gust facto between gd_c and gd_t
    """
    # gust duration (sec)
    gd = [1.0, 2.0, 5.0, 10.0, 20.0, 
        50.0, 100.0, 200.0, 500.0, 1000.0, 
        2000.0, 3600.0]
    # gust factor w.r.t. 3600 sec
    gf = [1.59, 1.55, 1.47, 1.40, 1.32, 
        1.20, 1.15, 1.10, 1.055, 1.045, 
        1.02, 1.00]
    # interpolation
    gf_t = np.interp(gd_t, gd, gf, left = gf[0], right = gf[-1]) \
        / np.interp(gd_c, gd, gf, left = gf[0], right = gf[-1])
    # return
    return gf_t


def export_pws(stations, pws, output_dir, filename = 'EventGrid.csv'):

    print('ComputeIntensityMeasure: saving results.')

    # collecting site locations
    lat = []
    lon = []
    for s in stations['Stations']:
        lat.append(s['Latitude'])
        lon.append(s['Longitude'])

    # saving data
    station_num = len(lat)
    csv_file = [str(x + 1)+'.csv' for x in range(station_num)]
    d = {
        'Station': csv_file,
        'Latitude': lat,
        'Longitude': lon
    }
    df = pd.DataFrame.from_dict(d)
    df.to_csv(os.path.join(output_dir, filename), index = False)
    for i in range(station_num):
        pws_op = [pws[0][i]]
        if len(pws) > 1:
            for j in range(len(pws) - 1):
                pws_op.append(pws[j + 1][i])
        d = {
            'PWS': pws_op    
        }
        df = pd.DataFrame.from_dict(d)
        df.to_csv(os.path.join(output_dir, csv_file[i]), index = False)

    print('ComputeIntensityMeasure: simulated wind speed field saved.')
        

