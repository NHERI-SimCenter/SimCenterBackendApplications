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
import time
import glob
import re
import shutil
import sys
from pathlib import Path
R2D = True
if not R2D:
    from selenium import webdriver
import json
import random
import numpy as np
import pandas as pd
import zipfile
import csv
import copy


class GM_Selector:

    def __init__(self, gmdb_im_df=dict(), num_records=1, sf_min=None, sf_max=None, target_im=None):

        self.set_gmdb_im_df(gmdb_im_df)
        self.set_num_records(num_records)
        self.set_sf_range(sf_min, sf_max)
        self.set_target_im(target_im)

    def set_gmdb_im_df(self, gmdb_im_df):
        self.gmdb_im_df = gmdb_im_df
        self.num_gm = len(gmdb_im_df['RSN'])
        tmp_list = list(gmdb_im_df.keys())
        tmp_list.remove('RSN')
        self.im_list = tmp_list
        tmp_scalable = []
        for cur_im in self.im_list:
            if cur_im.startswith('DS'):
                tmp_scalable.append(0)
            else:
                tmp_scalable.append(1)
        self.scalable = tmp_scalable

    def set_num_records(self, num_records):
        self.num_records = num_records

    def set_sf_range(self, sf_min, sf_max):
        if sf_min is None:
            self.sf_min = 0.0001
        else:
            self.sf_min = sf_min
        if sf_max is None:
            self.sf_max = 100000.0
        else:
            self.sf_max = sf_max
        self.sf_range = [self.sf_min, self.sf_max]

    def set_target_im(self, target_im):
        self.target_im = [target_im for k in range(self.num_gm)]

    def select_records(self):

        im_table = self.gmdb_im_df.iloc[:,1:]
        min_err = 1000000.0
        for s in self.sf_range:
            cur_im_table = copy.copy(im_table)
            for i in range(cur_im_table.shape[1]):
                if self.scalable[i]:
                    cur_im_table.iloc[:,i] = cur_im_table.iloc[:,i]*s
            err = np.linalg.norm(np.exp(self.target_im) - cur_im_table.to_numpy(), axis = 1)
            if np.min(err) < min_err:
                min_err = np.min(err)
                tmp_tag = err.argmin()
                sf = s

        self.loc_tag = tmp_tag
        self.min_err = min_err
        self.rsn_tag = self.gmdb_im_df['RSN'].values.tolist()[tmp_tag]
        self.sf = sf


def select_ground_motion(im_list, target_ln_im, gmdb_file, sf_max, sf_min,
                         output_dir, output_file, stations):

    # Loading gmdb
    if gmdb_file == 'NGAWest2':
        cwd = os.path.dirname(os.path.realpath(__file__))
        gmdb = pd.read_csv(cwd+'/database/gmdb/NGAWest2.csv', header = 0, index_col = None, low_memory=False)
        # Parsing spectral data
        num_gm = len(gmdb['RecId'])
        tmp = gmdb.keys()[37:147]
        T_db = [float(a.replace('T','').replace('S','')) for a in tmp]
        psa_db = gmdb.iloc[:, 37:147]
        pga = gmdb.iloc[:, 34]
        pgv = gmdb.iloc[:, 35]
        pgd = gmdb.iloc[:, 36]
        # Scaling factors
        sf_range = np.linspace(sf_min, sf_max, 100)
        # Selected ground motion ID
        gm_id = []
        sf_data = []
        filename = []
        # get available key names
        # Parese im_list
        target_period = []
        im_map = {"PGA": 34,
                  "PGV": 35,
                  "PGD": 36,
                  "DS575H": 151,
                  "DS595H": 152}
        im_loc_tag = []
        gmdb_im_dict = dict()
        gmdb_im_dict.update({'RSN':gmdb['RecId'].values.tolist()})
        for cur_im in im_list:
            if cur_im.startswith('SA'):
                cur_period = float(cur_im[3:-1])
                gmdb_im_dict.update({cur_im:[np.interp(cur_period, T_db, psa_db.iloc[k, :]) for k in range(num_gm)]})
            else:
                im_loc_tag.append(im_map.get(cur_im, None))
                gmdb_im_dict.update({cur_im:[x[0] for x in gmdb.iloc[:, im_loc_tag].values.tolist()]})
        # ground motion database intensity measure data frame
        gmdb_im_df = pd.DataFrame.from_dict(gmdb_im_dict)
        tmp_scen = 0
        # Looping over all scenarios
        for cur_target in target_ln_im:
            tmp_scen = tmp_scen + 1
            print('-Scenario #'+str(tmp_scen))
            num_stations, num_periods, num_simu = cur_target.shape
            tmp_id = np.zeros((num_stations, num_simu))
            tmp_sf = np.zeros((num_stations, num_simu))
            tmp_min_err = np.zeros((num_stations, num_simu))
            tmp_filename = []
            for i in range(num_simu):
                print('--Realization #'+str(i+1))
                for j in range(num_stations):
                    # create a ground motion selector
                    gm_selector = GM_Selector(gmdb_im_df=gmdb_im_df, num_records=1, sf_min=sf_min, sf_max=sf_max, target_im=cur_target[j,:,i])
                    # select records
                    gm_selector.select_records()
                    # collect results
                    tmp_min_err[j, i] = gm_selector.min_err
                    tmp_id[j, i] = int(gmdb['RecId'][gm_selector.loc_tag])
                    tmp_sf[j, i] = gm_selector.sf
                    tmp_filename.append('RSN'+str(int(tmp_id[j,i]))+'_'+gmdb['FileNameHorizontal1'][gm_selector.loc_tag].replace("\\","_").replace("/","_"))
                    tmp_filename.append('RSN'+str(int(tmp_id[j,i]))+'_'+gmdb['FileNameHorizontal2'][gm_selector.loc_tag].replace("\\","_").replace("/","_"))
                    #print('---Station #'+str(j+1))
            # Collecting results in one scenario
            gm_id.append(tmp_id)
            sf_data.append(tmp_sf)
            filename.extend(tmp_filename)
            #print(tmp_min_err)
    else:
        print('SelectGroundMotion: currently only supporting NGAWest2.')
        return 1

    # output data
    station_name = ['site'+str(j)+'.csv' for j in range(len(stations))]
    lat = [stations[j]['Latitude'] for j in range(len(stations))]
    lon = [stations[j]['Longitude'] for j in range(len(stations))]
    vs30 = [stations[j]['Vs30'] for j in range(len(stations))]
    zTR = [stations[j]['zTR'] for j in range(len(stations))]
    df = pd.DataFrame({
        'GP_file': station_name,
        'Longitude': lon,
        'Latitude': lat,
		'Vs30': vs30,
		'zTR': zTR
    })
    output_dir = os.path.join(os.path.dirname(Path(output_dir)),
                              os.path.basename(Path(output_dir)))
    df.to_csv(os.path.join(output_dir, output_file), index = False)
    for cur_scen in range(len(gm_id)):
        if len(gm_id) > 1:
            cur_scen_folder = 'scenario'+str(cur_scen+1)
            try:
                os.mkdir(os.path.join(output_dir, cur_scen_folder))
            except:
                print('SelectGroundMotion: scenario folder already exists.')
            cur_output_dir = os.path.join(output_dir, cur_scen_folder)
        else:
            cur_output_dir = output_dir
        for i, site_id in enumerate(station_name):
            gm_file = ['RSN'+str(int(j)) for j in gm_id[cur_scen][i]]
            factor = [j for j in sf_data[cur_scen][i]]
            df = pd.DataFrame({
                'TH_file': gm_file,
                'factor': factor
            })
            df.to_csv(os.path.join(cur_output_dir, site_id), index = False)
    # return
    return gm_id, filename


def output_all_ground_motion_info(gm_id, gm_file, output_dir, filename):

    # Writing all record names to a csv file
    print(gm_file)
    try:
        with open(os.path.join(output_dir, filename), 'w') as f:
            w = csv.writer(f)
            if gm_file:
                w.writerow(gm_file)
        with open(os.path.join(output_dir, 'RSN.csv'), 'w') as f:
            w = csv.writer(f)
            if gm_id:
                w.writerow(gm_id)
        return 1
    except:
        return 0

""" Uncommenting below if use this tool alone to download records from PEER

def download_ground_motion(gm_id, user_name, user_password, output_dir, spectra_only=False):

    from selenium import webdriver
    # Setting chrome options
    if sys.platform.startswith('win'):
        chromedriver = os.path.dirname(__file__) + '/bin/chromedriver/chromedriver.exe'
    elif sys.platform.startswith('linux'):
        chromedriver = os.path.dirname(__file__) + '/bin/chromedriver/chromedriver_linux'
    elif sys.platform.startswith('darwin'):
        chromedriver = os.path.dirname(__file__) + '/bin/chromedriver/chromedriver_mac'
        os.chmod(chromedriver, 755)
    else:
        print('Currently supoorting win32, linux, and mac.')
    chromeOptions = webdriver.ChromeOptions()
    output_dir = os.path.join(os.path.dirname(Path(output_dir)),
                              os.path.basename(Path(output_dir)))
    prefs = {"download.default_directory" : output_dir, "directory_upgrade": True}
    chromeOptions.add_experimental_option("prefs", prefs)
    chromeOptions.add_experimental_option('excludeSwitches', ['enable-logging'])
    # Ground motion record numbers
    num_gm = len(gm_id)
    # Accessing NGA West-2 website
    gm_driver = webdriver.Chrome(executable_path=chromedriver, chrome_options=chromeOptions)
    gm_driver.get("https://ngawest2.berkeley.edu/users/sign_in?unauthenticated=true")
    try:
        gm_driver.find_element_by_id("user_email").send_keys(user_name)
        gm_driver.find_element_by_id("user_password").send_keys(user_password)
        gm_driver.find_element_by_id("user_submit").click()
        gm_driver.find_element_by_xpath('//a[@href="/spectras/new?sourceDb_flag=1"]').click()
        gm_driver.find_element_by_xpath('//button[@onclick="OnSubmit();"]').click()
        time.sleep(1)
    except:
        gm_driver.close()
        print('Please provide valid account name and password.')
        return 0

    # Grouping every 100 records (NGA West website allows 100 records/time)
    for r in range(int(np.ceil(num_gm/100))):
        cur_id = [f"{c}" for c in gm_id[r*100:min(r*100+100, num_gm)]]
        s = ","
        s = s.join(cur_id)
        gm_driver.find_element_by_id("search_search_nga_number").clear()
        gm_driver.find_element_by_id("search_search_nga_number").send_keys(s)
        gm_driver.find_element_by_xpath('//button[@onclick="uncheck_plot_selected();reset_selectedResult();OnSubmit();"]').click()
        time.sleep(10)
        if spectra_only:
            gm_driver.find_element_by_xpath('//button[@onclick="getSaveSearchResult()"]')
            time.sleep(5)
        else:
            gm_driver.find_element_by_xpath('//button[@onclick="getSelectedResult(true)"]').click()
            gm_driver.switch_to_alert().accept()
            gm_driver.switch_to_alert().accept()
            time.sleep(40)
    # Closing
    gm_driver.close()

    record_path = output_dir
    record_files = os.listdir(record_path)
    raw_record_folder = 'raw'
    if not spectra_only:
        try:
            os.mkdir(os.path.join(record_path, raw_record_folder))
        except:
            print('SelectGroundMotion: the /record/raw folder already exists.')
        for cur_file in record_files:
            if 'zip' in cur_file:
                with zipfile.ZipFile(os.path.join(record_path, cur_file), 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(record_path, raw_record_folder))
                os.remove(os.path.join(record_path, cur_file))
    # return
    return os.path.join(record_path, raw_record_folder)

def readNGAWest2record(ngaW2FilePath):
    series = []
    dt = 0.0
    with open(ngaW2FilePath, 'r') as recordFile:
        data_flag = False
        for line in recordFile:
            if(data_flag):
                # seismogram
                series.extend([float(value) for value in line.split()])
            elif("NPTS=" in line):
                # sampling rate
                dt = float(re.match(r"NPTS=.+, DT=\s+([0-9\.]+)\s+SEC", line).group(1))
                data_flag = True
    # return
    return series, dt


def parse_record(gm_file, raw_dir, output_dir, input_format, output_format):
    gm_file = np.reshape(gm_file, (-1, 2))
    for cur_id in gm_file:
        # Reading raw data
        if input_format == 'NGAWest2':
            if(len(cur_id) != 2):
                print('Error finding NGA West 2 files.\n'\
                'Please download the files for record {} '\
                .format(cur_id))
                exit(-1)
            acc_1, dt_1 = readNGAWest2record(os.path.join(raw_dir, cur_id[0]))
            acc_2, dt_2 = readNGAWest2record(os.path.join(raw_dir, cur_id[1]))
        else:
            print('Currently only supporting NGAWest2')
        # Parsing output files
        rsn = cur_id[0].split('_')[0]
        if output_format == 'SimCenterEvent':
            tmp = {
                "name": str(rsn),
                "dT": dt_1,
                "data_x": acc_1,
                "data_y": acc_2,
                "PGA_x": max(abs(np.array(acc_1))),
                "PGA_y": max(abs(np.array(acc_2)))
            }
            with open(output_dir+str(rsn)+'.json', 'w') as f:
                json.dump(tmp, f, indent = 2)
        else:
            print('Currently only supporting SimCenterEvent')

    # removing raw files
    shutil.rmtree(raw_dir)
    # return
    return output_dir

Uncommenting above if use this tool alone to download records from PEER
"""
