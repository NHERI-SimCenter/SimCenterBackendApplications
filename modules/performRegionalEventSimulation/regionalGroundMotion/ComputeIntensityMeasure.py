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
# The computation method of ground motion intensity map using Markhivida et al. and
# the Baker-Jayaram correlation models is contributed by Dr. Anne Husley's
# seaturtles package (https://github.com/annehulsey/seaturtles).
#
# Contributors:
# Anne Husley
# Kuanshi Zhong
# Jinyan Zhao

import re

import h5py
import numpy as np

LOCAL_IM_GMPE = {
    'DS575H': ['Bommer, Stafford & Alarcon (2009)', 'Afshari & Stewart (2016)'],
    'DS595H': ['Bommer, Stafford & Alarcon (2009)', 'Afshari & Stewart (2016)'],
    'DS2080H': ['Afshari & Stewart (2016)'],
    'SA': [
        'Chiou & Youngs (2014)',
        'Abrahamson, Silva & Kamai (2014)',
        'Boore, Stewart, Seyhan & Atkinson (2014)',
        'Campbell & Bozorgnia (2014)',
    ],
    'PGA': [
        'Chiou & Youngs (2014)',
        'Abrahamson, Silva & Kamai (2014)',
        'Boore, Stewart, Seyhan & Atkinson (2014)',
        'Campbell & Bozorgnia (2014)',
    ],
    'PGV': [
        'Chiou & Youngs (2014)',
        'Abrahamson, Silva & Kamai (2014)',
        'Boore, Stewart, Seyhan & Atkinson (2014)',
        'Campbell & Bozorgnia (2014)',
    ],
}

OPENSHA_IM_GMPE = {
    'SA': [
        'Abrahamson, Silva & Kamai (2014)',
        'Boore, Stewart, Seyhan & Atkinson (2014)',
        'Campbell & Bozorgnia (2014)',
        'Chiou & Youngs (2014)',
    ],
    'PGA': [
        'Abrahamson, Silva & Kamai (2014)',
        'Boore, Stewart, Seyhan & Atkinson (2014)',
        'Campbell & Bozorgnia (2014)',
        'Chiou & Youngs (2014)',
    ],
    'PGV': [
        'Abrahamson, Silva & Kamai (2014)',
        'Boore, Stewart, Seyhan & Atkinson (2014)',
        'Campbell & Bozorgnia (2014)',
        'Chiou & Youngs (2014)',
    ],
}

IM_GMPE = {'LOCAL': LOCAL_IM_GMPE, 'OPENSHA': OPENSHA_IM_GMPE}

import collections  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
from gmpe import SignificantDurationModel, openSHAGMPE  # noqa: E402
from tqdm import tqdm  # noqa: E402

if 'stampede2' not in socket.gethostname():
    from FetchOpenQuake import get_site_rup_info_oq
    from FetchOpenSHA import *  # noqa: F403
import threading  # noqa: E402

import ujson  # noqa: E402


class IM_Calculator:  # noqa: N801, D101
    # Chiou & Youngs (2014) GMPE class
    CY = None
    # Abrahamson, Silvar, & Kamai (2014)
    ASK = None
    # Boore, Stewart, Seyhan & Atkinson (2014)
    BSSA = None
    # Campbell & Bozorgnia (2014)
    CB = None

    # profile
    timeGetRuptureInfo = 0  # noqa: N815
    timeGetIM = 0  # noqa: N815

    def __init__(  # noqa: ANN204, D107, PLR0913
        self,
        source_info=dict(),  # noqa: ANN001, B006, C408, ARG002
        im_dict=dict(),  # noqa: ANN001, B006, C408
        gmpe_dict=dict(),  # noqa: ANN001, B006, C408
        gmpe_weights_dict=dict(),  # noqa: ANN001, B006, C408
        im_type=None,  # noqa: ANN001
        site_info=dict(),  # noqa: ANN001, B006, C408
    ):
        # basic set-ups
        self.set_im_gmpe(im_dict, gmpe_dict, gmpe_weights_dict)
        self.set_im_type(im_type)
        self.set_sites(site_info)
        # self.set_source(source_info)

    def set_source(self, source_info):  # noqa: ANN001, ANN201, D102
        # set seismic source
        self.source_info = source_info.copy()
        gmpe_list = set()
        for _, item in self.gmpe_dict.items():  # noqa: PERF102
            gmpe_list = gmpe_list.union(set(item))
        if source_info['Type'] == 'ERF':
            if (
                'Chiou & Youngs (2014)' in gmpe_list
                or 'Abrahamson, Silva & Kamai (2014)' in gmpe_list
                or 'Boore, Stewart, Seyhan & Atkinson (2014)' in gmpe_list
                or 'Campbell & Bozorgnia (2014)' in gmpe_list
            ):
                source_index = source_info.get('SourceIndex', None)
                rupture_index = source_info.get('RuptureIndex', None)
                # start = time.process_time_ns()
                site_rup_dict, station_info = get_rupture_info_CY2014(  # noqa: F405
                    self.erf, source_index, rupture_index, self.site_info
                )
                # self.timeGetRuptureInfo += time.process_time_ns() - start
        elif source_info['Type'] == 'PointSource':
            if (
                'Chiou & Youngs (2014)' in gmpe_list
                or 'Abrahamson, Silva & Kamai (2014)' in gmpe_list
                or 'Boore, Stewart, Seyhan & Atkinson (2014)' in gmpe_list
                or 'Campbell & Bozorgnia (2014)' in gmpe_list
            ):
                # start = time.process_time_ns()
                site_rup_dict, station_info = get_PointSource_info_CY2014(  # noqa: F405
                    source_info, self.site_info
                )
                # self.timeGetRuptureInfo += time.process_time_ns() - start
        elif source_info['Type'] == 'oqSourceXML':  # noqa: SIM102
            if (
                'Chiou & Youngs (2014)' in gmpe_list
                or 'Abrahamson, Silva & Kamai (2014)' in gmpe_list
                or 'Boore, Stewart, Seyhan & Atkinson (2014)' in gmpe_list
                or 'Campbell & Bozorgnia (2014)' in gmpe_list
            ):
                # start = time.process_time_ns()
                site_rup_dict, station_info = get_site_rup_info_oq(
                    source_info, self.site_info
                )
                # self.timeGetRuptureInfo += time.process_time_ns() - start
        self.site_rup_dict = site_rup_dict
        self.site_info = station_info

    def set_im_gmpe(self, im_dict, gmpe_dict, gmpe_weights_dict):  # noqa: ANN001, ANN201, D102
        # set im and gmpe information
        self.im_dict = im_dict.copy()
        self.gmpe_dict = gmpe_dict.copy()
        self.gmpe_weights_dict = gmpe_weights_dict.copy()

    def set_im_type(self, im_type):  # noqa: ANN001, ANN201, D102
        # set im type
        if im_type is None:
            self.im_type = None
        elif list(self.im_dict.keys()) and (
            im_type not in list(self.im_dict.keys())
        ):
            print(  # noqa: T201
                f'IM_Calculator.set_im_type: warning - {im_type} is not in the defined IM lists.'
            )
            self.im_type = None
        else:
            self.im_type = im_type

    def set_sites(self, site_info):  # noqa: ANN001, ANN201, D102
        # set sites
        self.site_info = site_info

    def calculate_im(self):  # noqa: ANN201, C901, D102, PLR0912
        # set up intensity measure calculations
        # current im type
        im_type = self.im_type
        if im_type is None:
            print('IM_Calculator.calculate_im: error - no IM type found.')  # noqa: T201
            return None
        # get current im dict
        cur_im_dict = self.im_dict.get(im_type)
        # get gmpe list
        gmpe_list = self.gmpe_dict.get(im_type, None)
        if gmpe_list is None:
            print(  # noqa: T201
                f'IM_Calculator.calculate_im: error - no GMPE list found for {im_type}.'
            )
            return None
        # get gmpe weights
        gmpe_weights_list = self.gmpe_weights_dict.get(im_type, None)
        # parse the gmpe list (split the list to two - local and opensha)
        gmpe_list_local = []
        gmpe_weigts_list_local = []
        gmpe_list_opensha = []
        gmpe_weigts_list_opensha = []
        for i, cur_gmpe in enumerate(gmpe_list):
            if cur_gmpe in LOCAL_IM_GMPE.get(im_type, []):
                gmpe_list_local.append(cur_gmpe)
                if gmpe_weights_list is not None:
                    gmpe_weigts_list_local.append(gmpe_weights_list[i])
                else:
                    gmpe_weights_list_local = None
            elif cur_gmpe in OPENSHA_IM_GMPE.get(im_type, []):
                gmpe_list_opensha.append(cur_gmpe)
                if gmpe_weights_list is not None:
                    gmpe_weigts_list_opensha.append(gmpe_weights_list[i])
                else:
                    gmpe_weights_list_opensha = None
            else:
                print(  # noqa: T201
                    f'IM_Calculator.calculate_im: error - {cur_gmpe} is not supported.'
                )
                return None
        # now compute im values
        if len(gmpe_list_local) > 0:
            res_local = self.get_im_from_local(
                self.source_info,
                gmpe_list_local,
                im_type,
                cur_im_dict,
                gmpe_weights=gmpe_weights_list_local,
            )
        else:
            res_local = dict()  # noqa: C408
        if len(gmpe_list_opensha) > 0:
            res_opensha = self.get_im_from_opensha(
                self.source_info,
                gmpe_list_opensha,
                self.gmpe_dict.get('Parameters'),
                self.erf,
                self.site_info,
                im_type,
                cur_im_dict,
                gmpe_weights=gmpe_weights_list_opensha,
            )
        else:
            res_opensha = dict()  # noqa: C408

        # collect/combine im results
        if len(res_local) + len(res_opensha) == 0:
            print(  # noqa: T201
                'IM_Calculator.calculate_im: error - no results available... please check GMPE availability'
            )
            return dict()  # noqa: C408
        if len(res_local) == 0:
            res = res_opensha
        elif len(res_opensha) == 0:
            res = res_local
        else:
            res = compute_weighted_res(
                [res_local, res_opensha],
                [np.sum(gmpe_weights_list_local, np.sum(gmpe_weights_list_opensha))],
            )

        # return
        return res

    def get_im_from_opensha(  # noqa: ANN201, D102, PLR0913
        self,
        source_info,  # noqa: ANN001
        gmpe_list,  # noqa: ANN001
        gmpe_para,  # noqa: ANN001
        erf,  # noqa: ANN001
        station_info,  # noqa: ANN001
        im_type,  # noqa: ANN001
        im_info,  # noqa: ANN001
        gmpe_weights=None,  # noqa: ANN001
    ):
        # Computing IM
        res_list = []
        res = dict()  # noqa: C408
        curgmpe_info = {}
        station_list = station_info.get('SiteList')
        im_info.update({'Type': im_type})
        for cur_gmpe in gmpe_list:
            # set up site properties
            siteSpec, sites, site_prop = get_site_prop(cur_gmpe, station_list)  # noqa: N806, F405
            curgmpe_info['Type'] = cur_gmpe
            curgmpe_info['Parameters'] = gmpe_para
            cur_res, station_info = get_IM(  # noqa: F405
                curgmpe_info,
                erf,
                sites,
                siteSpec,
                site_prop,
                source_info,
                station_info,
                im_info,
            )
            cur_res.update({'IM': im_type})
            res_list.append(cur_res)
        # weighting if any
        if gmpe_weights is not None:
            res = compute_weighted_res(res_list, gmpe_weights)
        else:
            res = res_list[0]
        # return
        return res

    def get_im_from_local(  # noqa: ANN201, C901, D102, PLR0912, PLR0915
        self,
        source_info,  # noqa: ANN001
        gmpe_list,  # noqa: ANN001
        im_type,  # noqa: ANN001
        im_info,  # noqa: ANN001
        gmpe_weights=None,  # noqa: ANN001
    ):
        # initiate
        res_list = []
        res = dict()  # noqa: C408
        # check IM type
        if im_type not in list(LOCAL_IM_GMPE.keys()):
            print(  # noqa: T201
                f'ComputeIntensityMeasure.get_im_from_local: error - IM type {im_type} not supported'
            )
            return res
        # get available gmpe list
        avail_gmpe = LOCAL_IM_GMPE.get(im_type)
        # back compatibility for now (useful if other local GMPEs for SA is included)
        cur_T = im_info.get('Periods', None)  # noqa: N806
        # source and rupture
        if source_info['Type'] == 'PointSource':
            # magnitude
            eq_magnitude = source_info['Magnitude']
            eq_loc = [  # noqa: F841
                source_info['Location']['Latitude'],
                source_info['Location']['Longitude'],
                source_info['Location']['Depth'],
            ]
            # maf
            meanAnnualRate = None  # noqa: N806
        elif source_info['Type'] == 'ERF':
            source_index = source_info.get('SourceIndex', None)
            rupture_index = source_info.get('RuptureIndex', None)
            if None in [source_index, rupture_index]:
                print(  # noqa: T201
                    'ComputeIntensityMeasure.get_im_from_local: error - source/rupture index not given.'
                )
                return res
            # magnitude
            # eq_magnitude = erf.getSource(source_index).getRupture(rupture_index).getMag()
            eq_magnitude = source_info['Magnitude']
            # maf
            # timeSpan = erf.getTimeSpan()
            # meanAnnualRate = erf.getSource(source_index).getRupture(rupture_index).getMeanAnnualRate(timeSpan.getDuration())
            meanAnnualRate = source_info['MeanAnnualRate']  # noqa: N806
        elif source_info['Type'] == 'oqSourceXML':
            source_index = source_info.get('SourceIndex', None)
            rupture_index = source_info.get('RuptureIndex', None)
            if None in [source_index, rupture_index]:
                print(  # noqa: T201
                    'ComputeIntensityMeasure.get_im_from_local: error - source/rupture index not given.'
                )
                return res
            # magnitude
            eq_magnitude = source_info['Magnitude']
            # maf
            meanAnnualRate = source_info['MeanAnnualRate']  # noqa: N806
        else:
            print(  # noqa: T201
                'ComputeIntensityMeasure.get_im_from_local: error - source type {} not supported'.format(
                    source_info['Type']
                )
            )
            return res
        for cur_gmpe in gmpe_list:
            gm_collector = []
            if cur_gmpe not in avail_gmpe:
                print(  # noqa: T201
                    f'ComputeIntensityMeasure.get_im_from_local: warning - {cur_gmpe} is not available.'
                )
                continue
            for cur_site in self.site_info:
                # current site-rupture distance
                cur_dist = cur_site['rRup']
                cur_vs30 = cur_site['vs30']
                tmpResult = {  # noqa: N806
                    'Mean': [],
                    'TotalStdDev': [],
                    'InterEvStdDev': [],
                    'IntraEvStdDev': [],
                }
                if cur_gmpe == 'Bommer, Stafford & Alarcon (2009)':
                    mean, stdDev, interEvStdDev, intraEvStdDev = (  # noqa: N806
                        SignificantDurationModel.bommer_stafford_alarcon_ds_2009(
                            magnitude=eq_magnitude,
                            distance=cur_dist,
                            vs30=cur_vs30,
                            duration_type=im_type,
                        )
                    )
                    tmpResult['Mean'].append(float(mean))
                    tmpResult['TotalStdDev'].append(float(stdDev))
                    tmpResult['InterEvStdDev'].append(float(interEvStdDev))
                    tmpResult['IntraEvStdDev'].append(float(intraEvStdDev))
                elif cur_gmpe == 'Afshari & Stewart (2016)':
                    mean, stdDev, interEvStdDev, intraEvStdDev = (  # noqa: N806
                        SignificantDurationModel.afshari_stewart_ds_2016(
                            magnitude=eq_magnitude,
                            distance=cur_dist,
                            vs30=cur_vs30,
                            duration_type=im_type,
                        )
                    )
                    tmpResult['Mean'].append(float(mean))
                    tmpResult['TotalStdDev'].append(float(stdDev))
                    tmpResult['InterEvStdDev'].append(float(interEvStdDev))
                    tmpResult['IntraEvStdDev'].append(float(intraEvStdDev))
                elif cur_gmpe == 'Chiou & Youngs (2014)':
                    # start = time.process_time_ns()
                    tmpResult = self.CY.get_IM(  # noqa: N806
                        eq_magnitude, self.site_rup_dict, cur_site, im_info
                    )
                    # self.timeGetIM += time.process_time_ns() - start
                elif cur_gmpe == 'Abrahamson, Silva & Kamai (2014)':
                    # start = time.process_time_ns()
                    tmpResult = self.ASK.get_IM(  # noqa: N806
                        eq_magnitude, self.site_rup_dict, cur_site, im_info
                    )
                    # self.timeGetIM += time.process_time_ns() - start
                elif cur_gmpe == 'Boore, Stewart, Seyhan & Atkinson (2014)':
                    # start = time.process_time_ns()
                    tmpResult = self.BSSA.get_IM(  # noqa: N806
                        eq_magnitude, self.site_rup_dict, cur_site, im_info
                    )
                    # self.timeGetIM += time.process_time_ns() - start
                elif cur_gmpe == 'Campbell & Bozorgnia (2014)':
                    # start = time.process_time_ns()
                    tmpResult = self.CB.get_IM(  # noqa: N806
                        eq_magnitude, self.site_rup_dict, cur_site, im_info
                    )
                    # self.timeGetIM += time.process_time_ns() - start
                else:
                    print(  # noqa: T201
                        f'ComputeIntensityMeasure.get_im_from_local: gmpe_name {cur_gmpe} is not supported.'
                    )
                # collect sites
                # gm_collector.append({
                # 	"Location": {'Latitude':cur_site['lat'], 'Longitude':cur_site['lon']},
                #              "SiteData": {key: cur_site[key] for key in cur_site if key not in ['lat','lon']},
                # 			 'ln'+im_type: tmpResult
                # 			 })
                gm_collector.append({'ln' + im_type: tmpResult})

            # Final results
            cur_res = {
                'Magnitude': eq_magnitude,
                'MeanAnnualRate': meanAnnualRate,
                'SiteSourceDistance': source_info.get('SiteSourceDistance', None),
                'SiteRuptureDistance': source_info.get('SiteRuptureDistance', None),
                'Periods': cur_T,
                'IM': im_type,
                'GroundMotions': gm_collector,
            }
            # collect gmpes
            res_list.append(cur_res)

        # weighting if any
        if gmpe_weights is not None:
            res = compute_weighted_res(res_list, gmpe_weights)
        else:
            res = res_list[0]
        # return
        return res


def collect_multi_im_res(res_dict):  # noqa: ANN001, ANN201, C901, D103, PLR0912
    res_list = []
    if 'PGA' in res_dict.keys():  # noqa: SIM118
        res_list.append(res_dict['PGA'])
    if 'SA' in res_dict.keys():  # noqa: SIM118
        res_list.append(res_dict['SA'])
    if 'PGV' in res_dict.keys():  # noqa: SIM118
        res_list.append(res_dict['PGV'])
    res = dict()  # noqa: C408
    num_res = len(res_list)
    if num_res == 0:
        print('IM_Calculator._collect_res: error - the res_list is empty')  # noqa: T201
        return res
    for i, cur_res in enumerate(res_list):
        if i == 0:
            res = cur_res
            res['IM'] = [cur_res['IM']]
            if cur_res.get('Periods', None) is None:
                res['Periods'] = [None]
            elif type(cur_res.get('Periods')) in [float, int]:
                res['Periods'] = [cur_res.get('Periods')]
            else:
                res['Periods'] = cur_res.get('Periods')
        else:
            res['IM'].append(cur_res['IM'])
            if cur_res.get('Periods', None) is None:
                res['Periods'] = res['Periods'] + [None]
            elif type(cur_res.get('Periods')) in [float, int]:
                res['Periods'] = res['Periods'] + [cur_res.get('Periods')]
            else:
                res['Periods'] = res['Periods'] + cur_res.get('Periods')
            # combine ground motion characteristics
            for j in range(len(cur_res['GroundMotions'])):
                tmp_res = cur_res['GroundMotions'][j].get(
                    'ln{}'.format(cur_res['IM'])
                )
                res['GroundMotions'][j].update(
                    {'ln{}'.format(cur_res['IM']): tmp_res}
                )

    # return
    return res


def collect_multi_im_res_hdf5(res_list, im_list):  # noqa: ANN001, ANN201, D103
    res = dict()  # noqa: C408
    num_res = len(res_list)
    if num_res == 0:
        print('IM_Calculator._collect_res: error - the res_list is empty')  # noqa: T201
        return res
    num_sites = len(res_list[list(res_list.keys())[0]]['GroundMotions'])  # noqa: RUF015
    collected_mean = np.zeros([num_sites, len(im_list)])
    collected_intraStd = np.zeros([num_sites, len(im_list)])  # noqa: N806
    collected_interStd = np.zeros([num_sites, len(im_list)])  # noqa: N806
    for i, im in enumerate(im_list):
        if im.startswith('PGA'):
            collected_mean[:, i] = np.array(
                [x['lnPGA']['Mean'][0] for x in res_list['PGA']['GroundMotions']]
            )
            collected_interStd[:, i] = np.array(
                [
                    x['lnPGA']['InterEvStdDev'][0]
                    for x in res_list['PGA']['GroundMotions']
                ]
            )
            collected_intraStd[:, i] = np.array(
                [
                    x['lnPGA']['IntraEvStdDev'][0]
                    for x in res_list['PGA']['GroundMotions']
                ]
            )
        if im.startswith('SA'):
            period = float(re.search(r'\((.*?)\)', im).group(1))
            period_i = res_list['SA']['Periods'].index(period)
            collected_mean[:, i] = np.array(
                [
                    x['lnSA']['Mean'][period_i]
                    for x in res_list['SA']['GroundMotions']
                ]
            )
            collected_interStd[:, i] = np.array(
                [
                    x['lnSA']['InterEvStdDev'][period_i]
                    for x in res_list['SA']['GroundMotions']
                ]
            )
            collected_intraStd[:, i] = np.array(
                [
                    x['lnSA']['IntraEvStdDev'][period_i]
                    for x in res_list['SA']['GroundMotions']
                ]
            )
        if im.startswith('PGV'):
            collected_mean[:, i] = np.array(
                [x['lnPGV']['Mean'][0] for x in res_list['PGV']['GroundMotions']]
            )
            collected_interStd[:, i] = np.array(
                [
                    x['lnPGV']['InterEvStdDev'][0]
                    for x in res_list['PGV']['GroundMotions']
                ]
            )
            collected_intraStd[:, i] = np.array(
                [
                    x['lnPGV']['IntraEvStdDev'][0]
                    for x in res_list['PGV']['GroundMotions']
                ]
            )
    res.update({'Mean': collected_mean})
    res.update({'InterEvStdDev': collected_interStd})
    res.update({'IntraEvStdDev': collected_intraStd})
    # return
    return res


def get_im_dict(im_info):  # noqa: ANN001, ANN201, D103
    if im_info.get('Type', None) == 'Vector':
        im_dict = im_info.copy()
        im_dict.pop('Type')
        if 'PGV' in im_dict.keys():  # noqa: SIM118
            PGV_dict = im_dict.pop('PGV')  # noqa: N806
            im_dict.update({'PGV': PGV_dict})
    else:
        # back compatibility
        im_dict = {im_info.get('Type'): im_info.copy()}

    # return
    return im_dict


def get_gmpe_from_im_vector(im_info, gmpe_info):  # noqa: ANN001, ANN201, D103
    gmpe_dict = dict()  # noqa: C408
    gmpe_weights_dict = dict()  # noqa: C408
    # check IM info type
    if im_info.get('Type', None) != 'Vector':
        print(  # noqa: T201
            'ComputeIntensityMeasure.get_gmpe_from_im_vector: error: IntensityMeasure Type should be Vector.'
        )
        return gmpe_dict, gmpe_weights_dict
    else:  # noqa: RET505
        im_keys = list(im_info.keys())
        im_keys.remove('Type')
        for cur_im in im_keys:
            cur_gmpe = im_info[cur_im].get('GMPE', None)
            cur_weights = im_info[cur_im].get('GMPEWeights', None)
            if cur_gmpe is None:
                print(  # noqa: T201
                    f'ComputeIntensityMeasure.get_gmpe_from_im_vector: warning: GMPE not found for {cur_im}'
                )
            elif type(cur_gmpe) == str:  # noqa: E721
                if cur_gmpe == 'NGAWest2 2014 Averaged':
                    cur_gmpe = [
                        'Abrahamson, Silva & Kamai (2014)',
                        'Boore, Stewart, Seyhan & Atkinson (2014)',
                        'Campbell & Bozorgnia (2014)',
                        'Chiou & Youngs (2014)',
                    ]
                    cur_weights = [0.25, 0.25, 0.25, 0.25]
                else:
                    cur_gmpe = [cur_gmpe]
                    cur_weights = None
            gmpe_dict.update({cur_im: cur_gmpe})
            gmpe_weights_dict.update({cur_im: cur_weights})
    # global parameters if any
    gmpe_dict.update({'Parameters': gmpe_info.get('Parameters', dict())})  # noqa: C408
    # return
    return gmpe_dict, gmpe_weights_dict


def get_gmpe_from_im_legency(im_info, gmpe_info, gmpe_weights=None):  # noqa: ANN001, ANN201, D103
    # back compatibility for getting ims and gmpes
    gmpe_dict = dict()  # noqa: C408
    gmpe_weights_dict = dict()  # noqa: C408
    if gmpe_info['Type'] == 'NGAWest2 2014 Averaged':
        gmpe_list = [
            'Abrahamson, Silva & Kamai (2014)',
            'Boore, Stewart, Seyhan & Atkinson (2014)',
            'Campbell & Bozorgnia (2014)',
            'Chiou & Youngs (2014)',
        ]
        if gmpe_weights is None:
            gmpe_weights = [0.25, 0.25, 0.25, 0.25]
        im_type = im_info.get('Type')
        gmpe_dict = {im_type: gmpe_list}
    else:
        gmpe_list = [gmpe_info['Type']]
        gmpe_weights = None
        im_type = im_info.get('Type')
        # for im_type in im_types:
        gmpe_dict.update({im_type: gmpe_list})
        gmpe_weights_dict = {im_type: gmpe_weights}
    # global parameters if any
    gmpe_dict.update({'Parameters': gmpe_info.get('Parameters', dict())})  # noqa: C408
    # return
    return gmpe_dict, gmpe_weights_dict


def compute_im(  # noqa: ANN201, C901, D103, PLR0912, PLR0913, PLR0915
    scenarios,  # noqa: ANN001
    stations,  # noqa: ANN001
    EqRupture_info,  # noqa: ANN001, N803
    gmpe_info,  # noqa: ANN001
    im_info,  # noqa: ANN001
    generator_info,  # noqa: ANN001
    output_dir,  # noqa: ANN001
    filename='IntensityMeasureMeanStd.hdf5',  # noqa: ANN001
    mth_flag=True,  # noqa: ANN001, FBT002
):
    # Calling OpenSHA to compute median PSA
    if len(scenarios) < 10:  # noqa: PLR2004
        filename = 'IntensityMeasureMeanStd.json'
        saveInJson = True  # noqa: N806
        im_raw = {}
    else:
        saveInJson = False  # noqa: N806
    filename = os.path.join(output_dir, filename)  # noqa: PTH118
    im_list = []
    if 'PGA' in im_info.keys():  # noqa: SIM118
        im_list.append('PGA')
    if 'SA' in im_info.keys():  # noqa: SIM118
        for cur_period in im_info['SA']['Periods']:
            im_list.append(f'SA({cur_period!s})')  # noqa: PERF401
    if 'PGV' in im_info.keys():  # noqa: SIM118
        im_list.append('PGV')
    # Stations
    station_list = [
        {
            'Location': {
                'Latitude': stations[j]['lat'],
                'Longitude': stations[j]['lon'],
            }
        }
        for j in range(len(stations))
    ]
    for j in range(len(stations)):
        if stations[j].get('vs30'):
            station_list[j].update({'Vs30': int(stations[j]['vs30'])})
    station_info = {'Type': 'SiteList', 'SiteList': station_list}
    # hazard occurrent model
    if generator_info['method'] == 'Subsampling':  # noqa: SIM102
        # check if the period in the hazard curve is in the period list in the intensity measure
        if generator_info['Parameters'].get('IntensityMeasure') == 'SA':
            ho_period = generator_info['Parameters'].get('Period')
            if im_info['Type'] == 'Vector':
                if im_info.get('SA') is None:
                    sys.exit(
                        'SA is used in hazard downsampling but not defined in the intensity measure tab'
                    )
                elif ho_period in im_info['SA'].get('Periods'):
                    pass
                else:
                    tmp_periods = im_info['SA']['Periods'] + [ho_period]
                    tmp_periods.sort()
                    im_info['SA']['Periods'] = tmp_periods
            elif ho_period in im_info['SA'].get('Periods'):
                pass
            else:
                tmp_periods = im_info['SA']['Periods'] + [ho_period]
                tmp_periods.sort()
                im_info['SA']['Periods'] = tmp_periods
    # prepare gmpe list for intensity measure
    if gmpe_info['Type'] in ['Vector']:
        gmpe_dict, gmpe_weights_dict = get_gmpe_from_im_vector(im_info, gmpe_info)
    else:
        gmpe_dict, gmpe_weights_dict = get_gmpe_from_im_legency(im_info, gmpe_info)
    # prepare intensity measure dict
    im_dict = get_im_dict(im_info)

    t_start = time.time()
    # Loop over scenarios
    if mth_flag is False:
        # create a IM calculator
        im_calculator = IM_Calculator(
            im_dict=im_dict,
            gmpe_dict=gmpe_dict,
            gmpe_weights_dict=gmpe_weights_dict,
            site_info=stations,
        )
        if EqRupture_info['EqRupture']['Type'] in ['ERF']:
            im_calculator.erf = getERF(EqRupture_info)  # noqa: F405
        else:
            im_calculator.erf = None
        gmpe_set = set()
        for _, item in gmpe_dict.items():  # noqa: PERF102
            gmpe_set = gmpe_set.union(set(item))
        for gmpe in gmpe_set:
            if gmpe == 'Chiou & Youngs (2014)':
                im_calculator.CY = openSHAGMPE.chiou_youngs_2013()
            if gmpe == 'Abrahamson, Silva & Kamai (2014)':
                im_calculator.ASK = openSHAGMPE.abrahamson_silva_kamai_2014()
            if gmpe == 'Boore, Stewart, Seyhan & Atkinson (2014)':
                im_calculator.BSSA = openSHAGMPE.boore_etal_2014()
            if gmpe == 'Campbell & Bozorgnia (2014)':
                im_calculator.CB = openSHAGMPE.campbell_bozorgnia_2014()
        # for i in tqdm(range(len(scenarios.keys())), desc=f"Evaluate GMPEs for {len(scenarios.keys())} scenarios"):
        # Initialize an hdf5 file for IMmeanStd
        if os.path.exists(filename):  # noqa: PTH110
            os.remove(filename)  # noqa: PTH107
        for i in tqdm(
            range(len(scenarios.keys())),
            desc=f'Evaluate GMPEs for {len(scenarios.keys())} scenarios',
        ):
            # for i, key in enumerate(scenarios.keys()):
            # print('ComputeIntensityMeasure: Scenario #{}/{}'.format(i+1,len(scenarios)))
            # Rupture
            key = int(list(scenarios.keys())[i])
            source_info = scenarios[key]
            im_calculator.set_source(source_info)
            # Computing IM
            res_list = dict()  # noqa: C408
            for cur_im_type in list(im_dict.keys()):
                im_calculator.set_im_type(cur_im_type)
                res_list.update({cur_im_type: im_calculator.calculate_im()})
            # Collecting outputs
            # collectedResult.update({'SourceIndex':source_info['SourceIndex'], 'RuptureIndex':source_info['RuptureIndex']})
            if saveInJson:
                collectedResult = collect_multi_im_res(res_list)  # noqa: N806
                im_raw.update({key: collectedResult})
            else:
                collectedResult = collect_multi_im_res_hdf5(res_list, im_list)  # noqa: N806
                with h5py.File(filename, 'a') as f:
                    # Add a group named by the scenario index and has four dataset
                    # mean, totalSTd, interStd,itrastd
                    grp = f.create_group(str(i))
                    grp.create_dataset('Mean', data=collectedResult['Mean'])
                    grp.create_dataset(
                        'InterEvStdDev', data=collectedResult['InterEvStdDev']
                    )
                    grp.create_dataset(
                        'IntraEvStdDev', data=collectedResult['IntraEvStdDev']
                    )
            # if (i % 250 == 0):
            # 	if saveInJson:
            # 		print(f"Size of im_raw for {i} scenario is {sys.getsizeof(im_raw)}")
            # 	else:
            # 		print(f"Another 250 scenarios computed")

    if mth_flag:
        res_dict = {}
        sub_ths = []
        num_bins = 200
        bin_size = int(np.ceil(len(scenarios) / num_bins))
        ids_list = []
        scen_list = []
        for k in range(0, len(scenarios), bin_size):
            ids_list.append(list(scenarios.keys())[k : k + bin_size])
            scen_list.append(
                [scenarios[x] for x in list(scenarios.keys())[k : k + bin_size]]
            )
        # print(ids_list)
        for i in range(len(ids_list)):
            th = threading.Thread(
                target=compute_im_para,
                args=(
                    ids_list[i],
                    scen_list[i],
                    im_dict,
                    gmpe_dict,
                    gmpe_weights_dict,
                    station_info,
                    res_dict,
                ),
            )
            sub_ths.append(th)
            th.start()

        for th in sub_ths:
            th.join()

        # order the res_dict by id
        res_ordered = collections.OrderedDict(sorted(res_dict.items()))
        for i, cur_res in res_ordered.items():  # noqa: B007
            im_raw.append(cur_res)

    print(  # noqa: T201
        f'ComputeIntensityMeasure: mean and standard deviation of intensity measures {time.time() - t_start} sec'
    )

    if saveInJson:
        with open(filename, 'w') as f:  # noqa: PTH123
            ujson.dump(im_raw, f, indent=1)
    # return
    return filename, im_list


def compute_im_para(  # noqa: ANN201, D103, PLR0913
    ids,  # noqa: ANN001
    scenario_infos,  # noqa: ANN001
    im_dict,  # noqa: ANN001
    gmpe_dict,  # noqa: ANN001
    gmpe_weights_dict,  # noqa: ANN001
    station_info,  # noqa: ANN001
    res_dict,  # noqa: ANN001
):
    for i, id in enumerate(ids):  # noqa: A001
        print(f'ComputeIntensityMeasure: Scenario #{id + 1}.')  # noqa: T201
        scenario_info = scenario_infos[i]
        # create a IM calculator
        im_calculator = IM_Calculator(
            im_dict=im_dict,
            gmpe_dict=gmpe_dict,
            gmpe_weights_dict=gmpe_weights_dict,
            site_info=station_info,
        )
        # set scenario information
        im_calculator.set_source(scenario_info)
        # computing IM
        res_list = []
        for cur_im_type in list(im_dict.keys()):
            im_calculator.set_im_type(cur_im_type)
            res_list.append(im_calculator.calculate_im())
        # clean
        del im_calculator
        # collect multiple ims
        res = collect_multi_im_res(res_list)
        # append res to res_dcit
        res_dict[id] = res
    # return


def export_im(  # noqa: ANN201, C901, D103, PLR0912, PLR0913, PLR0915
    stations,  # noqa: ANN001
    im_list,  # noqa: ANN001
    im_data,  # noqa: ANN001
    eq_data,  # noqa: ANN001
    output_dir,  # noqa: ANN001
    filename,  # noqa: ANN001
    csv_flag,  # noqa: ANN001
    gf_im_list,  # noqa: ANN001
    scenario_ids,  # noqa: ANN001
):
    # Rename SA(xxx) to SA_xxx
    for i, im in enumerate(im_list):
        if im.startswith('SA'):
            im_list[i] = (
                im_list[i].split('(')[0] + '_' + im_list[i].split('(')[1][:-1]
            )
    # try:
    # Station number
    num_stations = len(stations)
    # Scenario number
    num_scenarios = len(eq_data)
    eq_data = np.array(eq_data)
    # Saving large files to HDF while small files to JSON
    if num_scenarios > 100000:  # noqa: PLR2004
        # Pandas DataFrame
        h_scenarios = ['Scenario-' + str(x) for x in range(1, num_scenarios + 1)]
        h_eq = [
            'Latitude',
            'Longitude',
            'Vs30',
            'Magnitude',
            'MeanAnnualRate',
            'SiteSourceDistance',
            'SiteRuptureDistance',
        ]
        for x in range(1, im_data[0][0, :, :].shape[1] + 1):
            for y in im_list:
                h_eq.append('Record-' + str(x) + f'-{y}')  # noqa: PERF401
        index = pd.MultiIndex.from_product([h_scenarios, h_eq])
        columns = ['Site-' + str(x) for x in range(1, num_stations + 1)]
        df = pd.DataFrame(index=index, columns=columns, dtype=float)  # noqa: PD901
        # Data
        for i in range(num_stations):
            tmp = []
            for j in range(num_scenarios):
                tmp.append(stations[i]['lat'])
                tmp.append(stations[i]['lon'])
                tmp.append(int(stations[i]['vs30']))
                tmp.append(eq_data[j][0])
                tmp.append(eq_data[j][1])
                tmp.append(eq_data[j][2])
                tmp.append(eq_data[j][3])
                for x in np.ndarray.tolist(im_data[j][i, :, :].T):
                    for y in x:
                        tmp.append(y)  # noqa: PERF402
            df['Site-' + str(i + 1)] = tmp
        # HDF output
        try:  # noqa: SIM105
            os.remove(os.path.join(output_dir, filename.replace('.json', '.h5')))  # noqa: PTH107, PTH118
        except:  # noqa: S110, E722
            pass
        hdf = pd.HDFStore(os.path.join(output_dir, filename.replace('.json', '.h5')))  # noqa: PTH118
        hdf.put('SiteIM', df, format='table', complib='zlib')
        hdf.close()
    else:
        res = []
        for i in range(num_stations):
            tmp = {
                'Location': {
                    'Latitude': stations[i]['lat'],
                    'Longitude': stations[i]['lon'],
                },
                'Vs30': int(stations[i]['vs30']),
            }
            tmp.update({'IMS': im_list})
            tmp_im = []
            for j in range(num_scenarios):
                tmp_im.append(np.ndarray.tolist(im_data[j][i, :, :]))  # noqa: PERF401
            if len(tmp_im) == 1:
                # Simplifying the data structure if only one scenario exists
                tmp_im = tmp_im[0]
            tmp.update({'lnIM': tmp_im})
            res.append(tmp)
        maf_out = []
        for ind, cur_eq in enumerate(eq_data):
            if cur_eq[1]:  # noqa: SIM108
                mar = cur_eq[1]
            else:
                mar = 'N/A'
            if cur_eq[2]:  # noqa: SIM108
                ssd = cur_eq[2]
            else:
                ssd = 'N/A'
            if len(cur_eq) > 3 and cur_eq[3]:  # noqa: SIM108, PLR2004
                srd = cur_eq[3]
            else:
                srd = 'N/A'
            tmp = {
                'Magnitude': float(cur_eq[0]),
                'MeanAnnualRate': mar,
                'SiteSourceDistance': ssd,
                'SiteRuputureDistance': srd,
                'ScenarioIndex': int(scenario_ids[ind]),
            }
            maf_out.append(tmp)
        res = {'Station_lnIM': res, 'Earthquake_MAF': maf_out}
        # save SiteIM.json
        with open(os.path.join(output_dir, filename), 'w') as f:  # noqa: PTH118, PTH123
            json.dump(res, f, indent=2)
    # export the event grid and station csv files
    if csv_flag:
        # output EventGrid.csv
        station_name = [
            'site' + str(stations[j]['ID']) + '.csv' for j in range(len(stations))
        ]
        lat = [stations[j]['lat'] for j in range(len(stations))]
        lon = [stations[j]['lon'] for j in range(len(stations))]
        # vs30 = [stations[j]['vs30'] for j in range(len(stations))]
        # zTR = [stations[j]['DepthToRock'] for j in range(len(stations))]
        df = pd.DataFrame(  # noqa: PD901
            {
                'GP_file': station_name,
                'Longitude': lon,
                'Latitude': lat,
                # 'Vs30': vs30,
                # 'DepthToRock': zTR
            }
        )
        # if cur_eq[2]:
        # 	df['SiteSourceDistance'] = cur_eq[2]
        output_dir = os.path.join(  # noqa: PTH118
            os.path.dirname(Path(output_dir)),  # noqa: PTH120
            os.path.basename(Path(output_dir)),  # noqa: PTH119
        )
        # separate directory for IM
        output_dir = os.path.join(output_dir, 'IMs')  # noqa: PTH118
        try:
            os.makedirs(output_dir)  # noqa: PTH103
        except:  # noqa: E722
            print('HazardSimulation: output folder already exists.')  # noqa: T201
        # save the csv
        df.to_csv(os.path.join(output_dir, 'EventGrid.csv'), index=False)  # noqa: PTH118
        # output station#.csv
        # csv header
        csvHeader = im_list  # noqa: N806
        for cur_scen in range(len(im_data)):
            if len(im_data) > 1:
                # IMPORTANT: the scenario index starts with 1 in the front end.
                cur_scen_folder = 'scenario' + str(int(scenario_ids[cur_scen]) + 1)
                try:  # noqa: SIM105
                    os.mkdir(os.path.join(output_dir, cur_scen_folder))  # noqa: PTH102, PTH118
                except:  # noqa: S110, E722
                    pass
                    # print('ComputeIntensityMeasure: scenario folder already exists.')
                cur_output_dir = os.path.join(output_dir, cur_scen_folder)  # noqa: PTH118
            else:
                cur_output_dir = output_dir
            # current IM data
            cur_im_data = im_data[cur_scen]
            for i, site_id in enumerate(station_name):
                df = dict()  # noqa: C408, PD901
                # Loop over all intensity measures
                for cur_im_tag in range(len(csvHeader)):
                    if (csvHeader[cur_im_tag].startswith('SA')) or (
                        csvHeader[cur_im_tag] in ['PGA', 'PGV']
                    ):
                        df.update(
                            {
                                csvHeader[cur_im_tag]: np.exp(
                                    cur_im_data[i, cur_im_tag, :]
                                )
                            }
                        )
                    else:
                        df.update(
                            {csvHeader[cur_im_tag]: cur_im_data[i, cur_im_tag, :]}
                        )
                df = pd.DataFrame(df)  # noqa: PD901
                # Combine PGD from liquefaction, landslide and fault
                if (
                    'liq_PGD_h' in df.columns
                    or 'ls_PGD_h' in df.columns
                    or 'fd_PGD_h' in df.columns
                ):
                    PGD_h = np.zeros(df.shape[0])  # noqa: N806
                    if 'liq_PGD_h' in df.columns:
                        PGD_h += df['liq_PGD_h'].to_numpy()  # noqa: N806
                    if 'ls_PGD_h' in df.columns:
                        PGD_h += df['ls_PGD_h'].to_numpy()  # noqa: N806
                    if 'fd_PGD_h' in df.columns:
                        PGD_h += df['fd_PGD_h'].to_numpy()  # noqa: N806
                    df['PGD_h'] = PGD_h
                if (
                    'liq_PGD_v' in df.columns
                    or 'ls_PGD_v' in df.columns
                    or 'fd_PGD_v' in df.columns
                ):
                    PGD_v = np.zeros(df.shape[0])  # noqa: N806
                    if 'liq_PGD_v' in df.columns:
                        PGD_v += df['liq_PGD_v'].to_numpy()  # noqa: N806
                    if 'ls_PGD_v' in df.columns:
                        PGD_v += df['ls_PGD_v'].to_numpy()  # noqa: N806
                    if 'fd_PGD_v' in df.columns:
                        PGD_v += df['fd_PGD_v'].to_numpy()  # noqa: N806
                    df['PGD_v'] = PGD_v
                colToDrop = []  # noqa: N806
                for col in df.columns:
                    if (
                        (not col.startswith('SA'))
                        and (col not in ['PGA', 'PGV', 'PGD_h', 'PGD_v'])
                        and (col not in gf_im_list)
                    ):
                        colToDrop.append(col)  # noqa: PERF401
                df.drop(columns=colToDrop, inplace=True)  # noqa: PD002
                # if 'liq_prob' in df.columns:
                # 	df.drop(columns=['liq_prob'], inplace=True)
                # if 'liq_susc' in df.columns:
                # 	df.drop(columns=['liq_susc'], inplace=True)
                df.fillna('NaN', inplace=True)  # noqa: PD002
                df.to_csv(os.path.join(cur_output_dir, site_id), index=False)  # noqa: PTH118

        # output the site#.csv file including all scenarios
        if len(im_data) > 1:
            print('ComputeIntensityMeasure: saving all selected scenarios.')  # noqa: T201
            # lopp over sites
            for i, site_id in enumerate(station_name):
                df = dict()  # noqa: C408, PD901
                for cur_im_tag in range(len(csvHeader)):
                    tmp_list = []
                    # loop over all scenarios
                    for cur_scen in range(len(im_data)):
                        tmp_list = (
                            tmp_list + im_data[cur_scen][i, cur_im_tag, :].tolist()
                        )
                    if (csvHeader[cur_im_tag].startswith('SA')) or (
                        csvHeader[cur_im_tag] in ['PGA', 'PGV']
                    ):
                        df.update({csvHeader[cur_im_tag]: np.exp(tmp_list)})
                    else:
                        df.update({csvHeader[cur_im_tag]: tmp_list})
                df = pd.DataFrame(df)  # noqa: PD901
                # Combine PGD from liquefaction, landslide and fault
                if (
                    'liq_PGD_h' in df.columns
                    or 'ls_PGD_h' in df.columns
                    or 'fd_PGD_h' in df.columns
                ):
                    PGD_h = np.zeros(df.shape[0])  # noqa: N806
                    if 'liq_PGD_h' in df.columns:
                        PGD_h += df['liq_PGD_h'].to_numpy()  # noqa: N806
                    if 'ls_PGD_h' in df.columns:
                        PGD_h += df['ls_PGD_h'].to_numpy()  # noqa: N806
                    if 'fd_PGD_h' in df.columns:
                        PGD_h += df['fd_PGD_h'].to_numpy()  # noqa: N806
                    df['PGD_h'] = PGD_h
                if (
                    'liq_PGD_v' in df.columns
                    or 'ls_PGD_v' in df.columns
                    or 'fd_PGD_v' in df.columns
                ):
                    PGD_v = np.zeros(df.shape[0])  # noqa: N806
                    if 'liq_PGD_v' in df.columns:
                        PGD_v += df['liq_PGD_v'].to_numpy()  # noqa: N806
                    if 'ls_PGD_v' in df.columns:
                        PGD_v += df['ls_PGD_v'].to_numpy()  # noqa: N806
                    if 'fd_PGD_v' in df.columns:
                        PGD_v += df['fd_PGD_v'].to_numpy()  # noqa: N806
                    df['PGD_v'] = PGD_v
                colToDrop = []  # noqa: N806
                for col in df.columns:
                    if (
                        (not col.startswith('SA'))
                        and (col not in ['PGA', 'PGV', 'PGD_h', 'PGD_v'])
                        and (col not in gf_im_list)
                    ):
                        colToDrop.append(col)
                df.drop(columns=colToDrop, inplace=True)  # noqa: PD002
                df.fillna('NaN', inplace=True)  # noqa: PD002
                df.to_csv(os.path.join(output_dir, site_id), index=False)  # noqa: PTH118
    # return
    return 0
    # except:
    # return
    # return 1


def compute_weighted_res(res_list, gmpe_weights):  # noqa: ANN001, ANN201, C901, D103, PLR0912
    # compute weighted average of gmpe results
    # initialize the return res (these three attributes are identical in different gmpe results)
    res = {
        'Magnitude': res_list[0]['Magnitude'],
        'MeanAnnualRate': res_list[0]['MeanAnnualRate'],
        'SiteSourceDistance': res_list[0].get('SiteSourceDistance', None),
        'Periods': res_list[0]['Periods'],
        'IM': res_list[0]['IM'],
    }
    # number of gmpe
    num_gmpe = len(res_list)
    # check number of weights
    if num_gmpe != len(gmpe_weights):
        print(  # noqa: T201
            'ComputeIntensityMeasure: please check the weights of different GMPEs.'
        )
        return 1
    # site number
    num_site = len(res_list[0]['GroundMotions'])
    # loop over different sites
    gm_collector = []
    for site_tag in range(num_site):
        # loop over different GMPE
        tmp_res = {}
        for i, cur_res in enumerate(res_list):
            cur_gmResults = cur_res['GroundMotions'][site_tag]  # noqa: N806
            # get keys
            im_keys = list(cur_gmResults.keys())
            for cur_im in im_keys:
                if cur_im not in list(tmp_res.keys()):
                    if cur_im in ['Location', 'SiteData']:
                        tmp_res.update({cur_im: cur_gmResults[cur_im]})
                    else:
                        tmp_res.update({cur_im: {}})
                if cur_im not in ['Location', 'SiteData']:
                    # get components
                    comp_keys = list(cur_gmResults[cur_im].keys())
                    # loop over different components
                    for cur_comp in comp_keys:
                        if cur_comp not in list(tmp_res[cur_im].keys()):
                            tmp_res[cur_im].update({cur_comp: []})
                            for cur_value in cur_gmResults[cur_im][cur_comp]:
                                if 'StdDev' in cur_comp:
                                    # standard deviation
                                    tmp_res[cur_im][cur_comp].append(
                                        np.sqrt(cur_value**2.0 * gmpe_weights[i])
                                    )
                                else:
                                    # mean
                                    tmp_res[cur_im][cur_comp].append(
                                        cur_value * gmpe_weights[i]
                                    )
                        else:
                            for j, cur_value in enumerate(
                                cur_gmResults[cur_im][cur_comp]
                            ):
                                if 'StdDev' in cur_comp:
                                    # standard deviation
                                    tmp_res[cur_im][cur_comp][j] = np.sqrt(
                                        tmp_res[cur_im][cur_comp][j] ** 2.0
                                        + cur_value**2.0 * gmpe_weights[i]
                                    )
                                else:
                                    # mean
                                    tmp_res[cur_im][cur_comp][j] = (
                                        tmp_res[cur_im][cur_comp][j]
                                        + cur_value * gmpe_weights[i]
                                    )
        # collector
        gm_collector.append(tmp_res)
    # res
    res.update({'GroundMotions': gm_collector})
    # return
    return res
