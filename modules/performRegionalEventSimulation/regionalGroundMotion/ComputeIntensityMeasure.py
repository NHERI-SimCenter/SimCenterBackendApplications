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
# The computation method of ground motion intensity map using Markhivida et al. and 
# the Baker-Jayaram correlation models is contributed by Dr. Anne Husley's
# seaturtles package (https://github.com/annehulsey/seaturtles). 
#
# Contributors:
# Anne Husley
# Kuanshi Zhong
# Jinyan Zhao

import warnings

LOCAL_IM_GMPE = {"DS575H": ["Bommer, Stafford & Alarcon (2009)", "Afshari & Stewart (2016)"],
                 "DS595H": ["Bommer, Stafford & Alarcon (2009)", "Afshari & Stewart (2016)"],
				 "DS2080H": ["Afshari & Stewart (2016)"],
				 "SA":["Chiou & Youngs (2014)", "Abrahamson, Silva & Kamai (2014)",\
		   "Boore, Stewart, Seyhan & Atkinson (2014)", "Campbell & Bozorgnia (2014)"],
				 "PGA":["Chiou & Youngs (2014)", "Abrahamson, Silva & Kamai (2014)",\
		   "Boore, Stewart, Seyhan & Atkinson (2014)", "Campbell & Bozorgnia (2014)"],
		   		 "PGV":["Chiou & Youngs (2014)", "Abrahamson, Silva & Kamai (2014)",\
		   "Boore, Stewart, Seyhan & Atkinson (2014)", "Campbell & Bozorgnia (2014)"]}

OPENSHA_IM_GMPE = {"SA": ["Abrahamson, Silva & Kamai (2014)", "Boore, Stewart, Seyhan & Atkinson (2014)", 
                          "Campbell & Bozorgnia (2014)", "Chiou & Youngs (2014)"],
                   "PGA": ["Abrahamson, Silva & Kamai (2014)", "Boore, Stewart, Seyhan & Atkinson (2014)", 
				           "Campbell & Bozorgnia (2014)", "Chiou & Youngs (2014)"],
				   "PGV": ["Abrahamson, Silva & Kamai (2014)", "Boore, Stewart, Seyhan & Atkinson (2014)", 
				           "Campbell & Bozorgnia (2014)", "Chiou & Youngs (2014)"]}

IM_GMPE = {"LOCAL": LOCAL_IM_GMPE,
           "OPENSHA": OPENSHA_IM_GMPE}

IM_CORR_INTER = {"Baker & Jayaram (2008)": ["SA"], 
                 "Baker & Bradley (2017)": ["SA", "PGA", "PGV", "DS575H", "DS595H"]}

IM_CORR_INTRA = {"Jayaram & Baker (2009)": ["SA"],
                 "Loth & Baker (2013)": ["SA"],
				 "Markhvida et al. (2017)": ["SA"],
				 "Du & Ning (2021)": ["SA", "PGA", "PGV", "Ia", "CAV", "DS575H", "DS595H"]}

IM_CORR = {"INTER": IM_CORR_INTER,
           "INTRA": IM_CORR_INTRA}

import os
import subprocess
import sys
import json
import numpy as np
from numpy.lib.utils import source
import pandas as pd
from gmpe import CorrelationModel, SignificantDurationModel, openSHAGMPE
from tqdm import tqdm
import time
from pathlib import Path
import copy
import socket
import collections
if 'stampede2' not in socket.gethostname():
	from FetchOpenSHA import *
	from FetchOpenQuake import get_site_rup_info_oq
import threading
import ujson

class IM_Calculator:
	# Chiou & Youngs (2014) GMPE class
	CY = None
	# Abrahamson, Silvar, & Kamai (2014)
	ASK = None
	# Boore, Stewart, Seyhan & Atkinson (2014)
	BSSA = None
	# Campbell & Bozorgnia (2014)
	CB = None

	# profile
	timeGetRuptureInfo = 0
	timeGetIM = 0
	def __init__(self, source_info=dict(), im_dict=dict(), gmpe_dict=dict(), 
	             gmpe_weights_dict=dict(), im_type=None, site_info=dict()):

		# basic set-ups
		self.set_source(source_info)
		self.set_im_gmpe(im_dict, gmpe_dict, gmpe_weights_dict)
		self.set_im_type(im_type)
		self.set_sites(site_info)
	
	def set_source(self, source_info):
		# set seismic source
		self.source_info = source_info.copy()
		# earthquake rupture forecast model (if any)
		# self.erf = None
		# if source_info.get('RuptureForecast', None):
		# 	self.erf = getERF(source_info['RuptureForecast'], True)

	def set_im_gmpe(self, im_dict, gmpe_dict, gmpe_weights_dict):
		# set im and gmpe information
		self.im_dict = im_dict.copy()
		self.gmpe_dict = gmpe_dict.copy()
		self.gmpe_weights_dict = gmpe_weights_dict.copy()

	def set_im_type(self, im_type):
		# set im type
		if im_type is None:
			self.im_type = None
		elif list(self.im_dict.keys()) and (im_type not in list(self.im_dict.keys())):
			print('IM_Calculator.set_im_type: warning - {} is not in the defined IM lists.'.format(im_type))
			self.im_type = None
		else:
			self.im_type = im_type
			
	def set_sites(self, site_info):
		# set sites
		self.site_info = site_info

	def calculate_im(self):
		# set up intensity measure calculations
		# current im type
		im_type = self.im_type
		if im_type is None:
			print('IM_Calculator.calculate_im: error - no IM type found.')
			return
		# get current im dict
		cur_im_dict = self.im_dict.get(im_type)
		# get gmpe list
		gmpe_list = self.gmpe_dict.get(im_type, None)
		if gmpe_list is None:
			print('IM_Calculator.calculate_im: error - no GMPE list found for {}.'.format(im_type))
			return
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
				print('IM_Calculator.calculate_im: error - {} is not supported.'.format(cur_gmpe))
				return
		# now compute im values
		if len(gmpe_list_local) > 0:
			res_local = self.get_im_from_local(self.source_info, gmpe_list_local, im_type, cur_im_dict, self.erf, 
			                                   self.site_info, gmpe_weights=gmpe_weights_list_local)
		else:
			res_local = dict()
		if len(gmpe_list_opensha) > 0:
			res_opensha = self.get_im_from_opensha(self.source_info, gmpe_list_opensha, self.gmpe_dict.get('Parameters'), self.erf, 
			                                       self.site_info, im_type, cur_im_dict, gmpe_weights=gmpe_weights_list_opensha)
		else:
			res_opensha = dict()
		
		# collect/combine im results
		if len(res_local)+len(res_opensha) == 0:
			print('IM_Calculator.calculate_im: error - no results available... please check GMPE availability')
			return dict()
		if len(res_local) == 0:
			res = res_opensha
		elif len(res_opensha) == 0:
			res = res_local
		else:
			res = compute_weighted_res([res_local, res_opensha], 
			                           [np.sum(gmpe_weights_list_local, np.sum(gmpe_weights_list_opensha))])

		# return
		return res
	def get_im_from_opensha(self, source_info, gmpe_list, gmpe_para, erf, station_info, im_type, im_info, gmpe_weights=None):

		# Computing IM
		res_list = []
		res = dict()
		curgmpe_info = {}
		station_list = station_info.get('SiteList')
		im_info.update({"Type": im_type})
		for cur_gmpe in gmpe_list:
			# set up site properties
			siteSpec, sites, site_prop = get_site_prop(cur_gmpe, station_list)
			curgmpe_info['Type'] = cur_gmpe
			curgmpe_info['Parameters'] = gmpe_para
			cur_res, station_info = get_IM(curgmpe_info, erf, sites, siteSpec, site_prop, source_info, station_info, im_info)
			cur_res.update({'IM': im_type})
			res_list.append(cur_res)
		# weighting if any
		if gmpe_weights is not None:
			res = compute_weighted_res(res_list, gmpe_weights)
		else:
			res = res_list[0]
		# return
		return res

	def get_im_from_local(self, source_info, gmpe_list, im_type, im_info, erf, station_info, gmpe_weights=None):

		# initiate
		res_list = []
		res = dict()
		# check IM type
		if im_type not in list(LOCAL_IM_GMPE.keys()):
			print('ComputeIntensityMeasure.get_im_from_local: error - IM type {} not supported'.format(im_type))
			return res
		# get availabel gmpe list
		avail_gmpe = LOCAL_IM_GMPE.get(im_type)
		# back compatibility for now (useful if other local GMPEs for SA is included)
		cur_T = im_info.get('Periods', None)
		# source and rupture
		if source_info['Type'] == 'PointSource':
			# magnitude
			eq_magnitude = source_info['Magnitude']
			eq_loc = [source_info['Location']['Latitude'],
					source_info['Location']['Longitude'],
					source_info['Location']['Depth']]
			# maf
			meanAnnualRate = None
		elif source_info['Type'] == 'ERF':
			source_index = source_info.get('SourceIndex', None)
			rupture_index = source_info.get('RuptureIndex', None)
			if None in [source_index, rupture_index]:
				print('ComputeIntensityMeasure.get_im_from_local: error - source/rupture index not given.')
				return res
			# magnitude
			# eq_magnitude = erf.getSource(source_index).getRupture(rupture_index).getMag()
			eq_magnitude = source_info["Magnitude"]
			# maf
			# timeSpan = erf.getTimeSpan()
			# meanAnnualRate = erf.getSource(source_index).getRupture(rupture_index).getMeanAnnualRate(timeSpan.getDuration())
			meanAnnualRate = source_info["MeanAnnualRate"]
		elif source_info['Type'] == 'oqSourceXML':
			source_index = source_info.get('SourceIndex', None)
			rupture_index = source_info.get('RuptureIndex', None)
			if None in [source_index, rupture_index]:
				print('ComputeIntensityMeasure.get_im_from_local: error - source/rupture index not given.')
				return res
			# magnitude
			eq_magnitude = source_info["Magnitude"]
			# maf
			meanAnnualRate = source_info["MeanAnnualRate"]
		else:
			print('ComputeIntensityMeasure.get_im_from_local: error - source type {} not supported'.format(source_info['Type']))
			return res
		# sites
		site_rup_dist = []
		for cur_site in station_info:
			cur_lat = cur_site['lat']
			cur_lon = cur_site['lon']
			# get distance
			if source_info['Type'] == 'PointSource':
				# no earth curvature is considered
				site_rup_dist.append(np.sqrt((eq_loc[0]-cur_lat)**2+(eq_loc[1]-cur_lon)**2+eq_loc[2]**2))
			elif source_info['Type'] == 'ERF':
				site_rup_dist.append(get_rupture_distance(erf, source_index, rupture_index, [cur_lat], [cur_lon])[0])
			elif source_info['Type'] == 'oqSourceXML':
				pass
		# evaluate gmpe
		site_rup_dist = []
		if source_info['Type']=='ERF':
			if 'Chiou & Youngs (2014)' in gmpe_list or 'Abrahamson, Silva & Kamai (2014)' in gmpe_list or\
				'Boore, Stewart, Seyhan & Atkinson (2014)' in gmpe_list or\
				'Campbell & Bozorgnia (2014)' in gmpe_list:
				start = time.process_time_ns()
				site_rup_dict, station_info = get_rupture_info_CY2014(erf, source_index, rupture_index, station_info)
				self.timeGetRuptureInfo += time.process_time_ns() - start
		elif source_info['Type']=='PointSource':
			if 'Chiou & Youngs (2014)' in gmpe_list or 'Abrahamson, Silva & Kamai (2014)' in gmpe_list or\
				'Boore, Stewart, Seyhan & Atkinson (2014)' in gmpe_list or\
				'Campbell & Bozorgnia (2014)' in gmpe_list:
				start = time.process_time_ns()
				site_rup_dict, station_info = get_PointSource_info_CY2014(source_info, station_info)
				self.timeGetRuptureInfo += time.process_time_ns() - start
		elif source_info['Type'] == 'oqSourceXML':
			if 'Chiou & Youngs (2014)' in gmpe_list or 'Abrahamson, Silva & Kamai (2014)' in gmpe_list or\
				'Boore, Stewart, Seyhan & Atkinson (2014)' in gmpe_list or\
				'Campbell & Bozorgnia (2014)' in gmpe_list:
				start = time.process_time_ns()
				site_rup_dict, station_info = get_site_rup_info_oq(source_info, station_info)
				self.timeGetRuptureInfo += time.process_time_ns() - start
		for cur_gmpe in gmpe_list:
			gm_collector = []
			if cur_gmpe not in avail_gmpe:
				print('ComputeIntensityMeasure.get_im_from_local: warning - {} is not available.'.format(cur_gmpe))
				continue
			for cur_site in station_info:
				# current site-rupture distance
				cur_dist = cur_site["rRup"]
				cur_vs30 = cur_site['vs30']
				tmpResult = {'Mean': [],
				             'TotalStdDev': [],
							 'InterEvStdDev': [],
							 'IntraEvStdDev': []}
				if cur_gmpe == 'Bommer, Stafford & Alarcon (2009)':
					mean, stdDev, interEvStdDev, intraEvStdDev = SignificantDurationModel.bommer_stafford_alarcon_ds_2009(magnitude=eq_magnitude, 
						distance=cur_dist, vs30=cur_vs30,duration_type=im_type)
					tmpResult['Mean'].append(float(mean))
					tmpResult['TotalStdDev'].append(float(stdDev))
					tmpResult['InterEvStdDev'].append(float(interEvStdDev))
					tmpResult['IntraEvStdDev'].append(float(intraEvStdDev))
				elif cur_gmpe == 'Afshari & Stewart (2016)':
					mean, stdDev, interEvStdDev, intraEvStdDev = SignificantDurationModel.afshari_stewart_ds_2016(magnitude=eq_magnitude, 
						distance=cur_dist, vs30=cur_vs30, duration_type=im_type)
					tmpResult['Mean'].append(float(mean))
					tmpResult['TotalStdDev'].append(float(stdDev))
					tmpResult['InterEvStdDev'].append(float(interEvStdDev))
					tmpResult['IntraEvStdDev'].append(float(intraEvStdDev))
				elif cur_gmpe == 'Chiou & Youngs (2014)':
					start = time.process_time_ns()
					tmpResult = self.CY.get_IM(eq_magnitude, site_rup_dict, cur_site, im_info)
					self.timeGetIM += time.process_time_ns() - start
				elif cur_gmpe == 'Abrahamson, Silva & Kamai (2014)':
					start = time.process_time_ns()
					tmpResult = self.ASK.get_IM(eq_magnitude, site_rup_dict, cur_site, im_info)
					self.timeGetIM += time.process_time_ns() - start
				elif cur_gmpe == 'Boore, Stewart, Seyhan & Atkinson (2014)':
					start = time.process_time_ns()
					tmpResult = self.BSSA.get_IM(eq_magnitude, site_rup_dict, cur_site, im_info)
					self.timeGetIM += time.process_time_ns() - start
				elif cur_gmpe == 'Campbell & Bozorgnia (2014)':
					start = time.process_time_ns()
					tmpResult = self.CB.get_IM(eq_magnitude, site_rup_dict, cur_site, im_info)
					self.timeGetIM += time.process_time_ns() - start
				else:
					print('ComputeIntensityMeasure.get_im_from_local: gmpe_name {} is not supported.'.format(cur_gmpe))
				# collect sites
				# gm_collector.append({
				# 	"Location": {'Latitude':cur_site['lat'], 'Longitude':cur_site['lon']},
				#              "SiteData": {key: cur_site[key] for key in cur_site if key not in ['lat','lon']},
				# 			 'ln'+im_type: tmpResult
				# 			 })
				gm_collector.append({
							 'ln'+im_type: tmpResult
							 })

			# Final results
			cur_res = {'Magnitude': eq_magnitude,
			           'MeanAnnualRate': meanAnnualRate,
					   'SiteSourceDistance': source_info.get('SiteSourceDistance',None),
					   'SiteRuptureDistance': source_info.get('SiteRuptureDistance',None),
					   'Periods': cur_T,
					   'IM': im_type,
					   'GroundMotions': gm_collector}
			# collect gmpes
			res_list.append(cur_res)

		# weighting if any 
		if gmpe_weights is not None:
			res = compute_weighted_res(res_list, gmpe_weights)
		else:
			res = res_list[0]
		# return
		return res

def collect_multi_im_res(res_list):

	res = dict()
	num_res = len(res_list)
	if num_res == 0:
		print('IM_Calculator._collect_res: error - the res_list is empty')
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
				res['Periods'] = res['Periods']+[None]
			elif type(cur_res.get('Periods')) in [float, int]:
				res['Periods'] = res['Periods']+[cur_res.get('Periods')]
			else:
				res['Periods'] = res['Periods']+cur_res.get('Periods')
			# combine ground motion characteristics
			for j in range(len(cur_res['GroundMotions'])):
				tmp_res = cur_res['GroundMotions'][j].get('ln{}'.format(cur_res['IM']))
				res['GroundMotions'][j].update({'ln{}'.format(cur_res['IM']): tmp_res})

	# return
	return res


def get_im_dict(im_info):
	if im_info.get("Type", None) == "Vector":
		im_dict = im_info.copy()
		im_dict.pop('Type')
		if ("PGV" in im_dict.keys()):
			PGV_dict = im_dict.pop('PGV')
			im_dict.update({'PGV':PGV_dict})
	else:
		# back compatibility
		im_dict = {im_info.get("Type"): im_info.copy()}

	# return
	return im_dict


def get_gmpe_from_im_vector(im_info, gmpe_info):

	gmpe_dict = dict()
	gmpe_weights_dict = dict()
	# check IM info type
	if not (im_info.get("Type", None) == "Vector"):
		print('ComputeIntensityMeasure.get_gmpe_from_im_vector: error: IntensityMeasure Type should be Vector.')
		return gmpe_dict, gmpe_weights_dict
	else:
		im_keys = list(im_info.keys())
		im_keys.remove('Type')
		for cur_im in im_keys:
			cur_gmpe = im_info[cur_im].get("GMPE", None)
			cur_weights = im_info[cur_im].get("GMPEWeights", None)
			if cur_gmpe is None:
				print('ComputeIntensityMeasure.get_gmpe_from_im_vector: warning: GMPE not found for {}'.format(cur_im))
			else:
				# back compatibility
				if type(cur_gmpe) == str:
					if cur_gmpe == 'NGAWest2 2014 Averaged':
						cur_gmpe = ["Abrahamson, Silva & Kamai (2014)", "Boore, Stewart, Seyhan & Atkinson (2014)", 
						            "Campbell & Bozorgnia (2014)", "Chiou & Youngs (2014)"]
						cur_weights = [0.25, 0.25, 0.25, 0.25]
					else:
						cur_gmpe = [cur_gmpe]
						cur_weights = None
			gmpe_dict.update({cur_im: cur_gmpe})
			gmpe_weights_dict.update({cur_im: cur_weights})
	# global parameters if any
	gmpe_dict.update({'Parameters': gmpe_info.get('Parameters',dict())})	
	# return
	return gmpe_dict, gmpe_weights_dict


def get_gmpe_from_im_legency(im_info, gmpe_info, gmpe_weights=None):

	# back compatibility for getting ims and gmpes
	gmpe_dict = dict()
	gmpe_weights_dict = dict()
	if gmpe_info['Type'] == 'NGAWest2 2014 Averaged':
		gmpe_list = ["Abrahamson, Silva & Kamai (2014)", "Boore, Stewart, Seyhan & Atkinson (2014)", 
					 "Campbell & Bozorgnia (2014)", "Chiou & Youngs (2014)"]
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
	gmpe_dict.update({'Parameters': gmpe_info.get('Parameters',dict())})
	# return
	return gmpe_dict, gmpe_weights_dict


def compute_im(scenarios, stations, EqRupture_info, gmpe_info, im_info, generator_info, output_dir, filename='IntensityMeasureMeanStd.json', mth_flag=True):

	# Calling OpenSHA to compute median PSA
	im_raw = {}
	# Loading ERF model (if exists)
	# erf = None
	# if scenarios[0].get('RuptureForecast', None):
	# 	erf = getERF(scenarios[0]['RuptureForecast'], True)
	# Stations
	station_list = [{
		'Location': {
			'Latitude': stations[j]['lat'],
			'Longitude': stations[j]['lon']
		}
	} for j in range(len(stations))]
	for j in range(len(stations)):
		if stations[j].get('vs30'):
			station_list[j].update({'Vs30': int(stations[j]['vs30'])})
	station_info = {'Type': 'SiteList',
					'SiteList': station_list}
	# hazard occurrent model
	if generator_info['method']=='Subsampling':
		# check if the period in the hazard curve is in the period list in the intensity measure
		if generator_info['Parameters'].get('IntensityMeasure')=='SA':
			ho_period = generator_info['Parameters'].get('Period')
			if im_info['Type'] == 'Vector':
				if im_info.get('SA') is None:
					sys.exit('SA is used in hazard downsampling but not defined in the intensity measure tab')
				else:
					if ho_period in im_info['SA'].get('Periods'):
						pass
					else:
						tmp_periods = im_info['SA']['Periods']+[ho_period]
						tmp_periods.sort()
						im_info['SA']['Periods'] = tmp_periods
			else:
				if ho_period in im_info['SA'].get('Periods'):
						pass
				else:
					tmp_periods = im_info['SA']['Periods']+[ho_period]
					tmp_periods.sort()
					im_info['SA']['Periods'] = tmp_periods
	# Configuring site properties
	siteSpec = []
	sites = []
	site_prop = []
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
		im_calculator = IM_Calculator(im_dict=im_dict, gmpe_dict=gmpe_dict, 
									gmpe_weights_dict=gmpe_weights_dict, site_info=stations)
		if EqRupture_info['EqRupture']['Type'] in ['ERF']:
			im_calculator.erf = getERF(EqRupture_info)
		else:
			im_calculator.erf = None
		for key in im_dict.keys():
			for gmpe in gmpe_dict[key]:
				if gmpe == "Chiou & Youngs (2014)":
					im_calculator.CY = openSHAGMPE.chiou_youngs_2013()
				if gmpe == 'Abrahamson, Silva & Kamai (2014)':
					im_calculator.ASK = openSHAGMPE.abrahamson_silva_kamai_2014()
				if gmpe == 'Boore, Stewart, Seyhan & Atkinson (2014)':
					im_calculator.BSSA = openSHAGMPE.boore_etal_2014()
				if gmpe == 'Campbell & Bozorgnia (2014)':
					im_calculator.CB = openSHAGMPE.campbell_bozorgnia_2014()
		# im_calculator.erf = getERF([elem for elem in scenarios.values()][0]['RuptureForecast'], True)
		for i in tqdm(range(len(scenarios.keys())), desc=f"Evaluate GMPEs for {len(scenarios.keys())} scenarios"):
		# for i, key in enumerate(scenarios.keys()):
			# print('ComputeIntensityMeasure: Scenario #{}/{}'.format(i+1,len(scenarios)))
			# Rupture
			key = int(list(scenarios.keys())[i])
			source_info = scenarios[key]
			im_calculator.set_source(source_info)
			# Computing IM
			res_list = []
			for cur_im_type in list(im_dict.keys()):
				im_calculator.set_im_type(cur_im_type)
				res_list.append(im_calculator.calculate_im())
			# Collecting outputs
			collectedResult = collect_multi_im_res(res_list)
			collectedResult.update({'SourceIndex':source_info['SourceIndex'], 'RuptureIndex':source_info['RuptureIndex']})
			# im_raw.append(collectedResult)
			im_raw.update({key:collectedResult})
			if (i % 250 == 0):
				print(f"Size of im_raw for {i} scenario is {sys.getsizeof(im_raw)}")
			# im_raw.append(copy.deepcopy(collect_multi_im_res(res_list)))

	if mth_flag:
		res_dict = {}
		sub_ths = []
		num_bins = 200
		bin_size = int(np.ceil(len(scenarios)/num_bins))
		ids_list = []
		scen_list = []
		for k in range(0, len(scenarios), bin_size):
			ids_list.append(list(scenarios.keys())[k:k+bin_size])
			scen_list.append([scenarios[x] for x in list(scenarios.keys())[k:k+bin_size]])
		#print(ids_list)
		for i in range(len(ids_list)):
			th = threading.Thread(target=compute_im_para, args=(ids_list[i], scen_list[i], im_dict, gmpe_dict, gmpe_weights_dict, station_info, res_dict))
			sub_ths.append(th)
			th.start()

		for th in sub_ths:
			th.join()

		# order the res_dict by id
		res_ordered = collections.OrderedDict(sorted(res_dict.items()))
		for i, cur_res in res_ordered.items():
			im_raw.append(cur_res)

	print('ComputeIntensityMeasure: mean and standard deviation of intensity measures {0} sec'.format(time.time() - t_start))

	# save im_raw comment out for debug
	# im_raw_preview = {"IntensityMeasures": im_raw}
	with open(os.path.join(output_dir, filename), "w") as f:
		ujson.dump(im_raw, f)

	# return
	return im_raw, im_info


def compute_im_para(ids, scenario_infos, im_dict, gmpe_dict, gmpe_weights_dict, station_info, res_dict):
	
	for i, id in enumerate(ids):
		print('ComputeIntensityMeasure: Scenario #{}.'.format(id+1))
		scenario_info = scenario_infos[i]
		# create a IM calculator
		im_calculator = IM_Calculator(im_dict=im_dict, gmpe_dict=gmpe_dict, 
									gmpe_weights_dict=gmpe_weights_dict, site_info=station_info)
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
	return


class GM_Simulator:

	def __init__(self, site_info = [], im_raw=dict(), num_simu=0,\
			  correlation_info=None,im_info = None):

		self.set_sites(site_info)
		self.set_num_simu(num_simu)
		self.parse_correlation_info(correlation_info, im_info)
		self.set_im_raw(im_raw)
		self.cross_check_im_correlation()

	def set_sites(self, site_info):
		# set sites
		self.sites = site_info.copy()
		self.num_sites = len(self.sites)
		if self.num_sites < 2:
			self.stn_dist = None
			print('GM_Simulator: Only one site is defined, spatial correlation models ignored.')
			return
		self._compute_distance_matrix()

	def _compute_distance_matrix(self):

		# site number check
		if self.num_sites < 2:
			print('GM_Simulator: error - please give at least two sites.')
			self.stn_dist = None
			return
		# compute the distance matrix
		tmp = np.zeros((self.num_sites, self.num_sites))
		for i in range(self.num_sites):
			loc_i = np.array([self.sites[i]['lat'],
							  self.sites[i]['lon']])
			for j in range(self.num_sites):
				loc_j = np.array([self.sites[j]['lat'],
								  self.sites[j]['lon']])
				# Computing station-wise distances
				tmp[i,j] = CorrelationModel.get_distance_from_lat_lon(loc_i, loc_j)
		self.stn_dist = tmp
	
	def set_num_simu(self, num_simu):
		# set simulation number
		self.num_simu = num_simu

	def set_im_raw(self, im_raw):
		# set up raw intensity measure data (mean/standard devs)
		self.im_raw = im_raw
		# get IM type list
		self.im_type_list = self.im_raw.get('IM',[])
		# get im_data
		self.im_data = self.im_raw.get('GroundMotions',[])
		# get period
		self.periods = [x for x in self.im_raw.get('Periods',[]) if x is not None]
		# im name list
		tmp_name = []
		for i in range(len(self.im_type_list)):
			if self.im_type_list[i] == 'SA':
				for cur_period in self.periods:
					tmp_name.append('SA({})'.format(str(cur_period)))
			else:
				tmp_name.append(self.im_type_list[i])
		self.im_name_list = tmp_name
		# set IM size
		self.num_im = len(self.im_name_list)

	def get_ln_im(self):
		ln_im = []
		for i in range(self.num_sites):
			tmp_im_data = []
			for cur_im_type in self.im_type_list:
				tmp_im_data = tmp_im_data+self.im_data[i]['ln{}'.format(cur_im_type)]['Mean']
			ln_im.append(tmp_im_data)
		return ln_im

	def get_inter_sigma_im(self):
		inter_sigma_im = []
		for i in range(self.num_sites):
			tmp_im_data = []
			for cur_im_type in self.im_type_list:
				tmp_im_data = tmp_im_data+self.im_data[i]['ln{}'.format(cur_im_type)]['InterEvStdDev']
			inter_sigma_im.append(tmp_im_data)
		return inter_sigma_im

	def get_intra_sigma_im(self):
		intra_sigma_im = []
		for i in range(self.num_sites):
			tmp_im_data = []
			for cur_im_type in self.im_type_list:
				tmp_im_data = tmp_im_data+self.im_data[i]['ln{}'.format(cur_im_type)]['IntraEvStdDev']
			intra_sigma_im.append(tmp_im_data)
		return intra_sigma_im

	def parse_correlation_info(self, correlation_info, im_info):

		# default is no correlation model and uncorrelated motions if generated
		self.inter_cm = None
		self.intra_cm = None
		# parse correlation infomation if any
		if correlation_info is None:
			print('GM_Simulator: warning - correlation information not found - results will be uncorrelated motions.')
			return
		if correlation_info.get('Type', None) == 'Vector':
			inter_cm = dict()
			im_info.pop('Type')
			for im, item in im_info.items():
			# for im in self.im_type_list:	
				inter_cm.update({im:item['InterEventCorr']})
			inter_cm_unique = list(set([item for _, item in inter_cm.items()]))
			if len(inter_cm_unique) == 1:
				inter_cm = inter_cm_unique[0]
			self.inter_cm = inter_cm
			intra_cm = dict()
			for im, item in im_info.items():
			# for im in self.im_type_list:
				intra_cm.update({im:item['IntraEventCorr']})
			intra_cm_unique = list(set([item for _, item in intra_cm.items()]))
			if len(intra_cm_unique) == 1:
				intra_cm = intra_cm_unique[0]
			self.intra_cm = intra_cm
			return
		
		# inter-event model
		if correlation_info.get('InterEvent', None):
			self.inter_cm = correlation_info['InterEvent']
		elif correlation_info.get('SaInterEvent', None):
			# back compatibility
			self.inter_cm = correlation_info['SaInterEvent']
		else:
			print('GM_Simulator: no inter-event correlation information not found - results will be uncorrelated motions.')
		# intra-event model
		if correlation_info.get('IntraEvent', None):
			self.intra_cm = correlation_info['IntraEvent']
		if correlation_info.get('SaIntraEvent', None):
			# back compatibility
			self.intra_cm = correlation_info['SaIntraEvent']
		else:
			print('GM_Simulator: no intra-event correlation information not found - results will be uncorrelated motions.')	

	def cross_check_im_correlation(self):
		# because each correlation model only applies to certain intensity measure
		# so hear we check if the correlation models are applicable for the required intensity measures
		self.im_cm_inter_flag = True
		self.im_cm_intra_flag = True
		if type(self.inter_cm)==dict:
			for cur_im in self.im_type_list:
				avail_im_inter_cm = IM_CORR_INTER.get(self.inter_cm[cur_im])
				if cur_im not in avail_im_inter_cm:
					print('GM_Simulator.cross_check_im_correlation: warning - {} is not available in {}'.format(cur_im, self.inter_cm))
					self.im_cm_inter_flag = False
					continue
		else:
			avail_im_inter_cm = IM_CORR_INTER.get(self.inter_cm)
			if avail_im_inter_cm is not None:
				for cur_im in self.im_type_list:
					if cur_im not in avail_im_inter_cm:
						print('GM_Simulator.cross_check_im_correlation: warning - {} is not available in {}'.format(cur_im, self.inter_cm))
						self.im_cm_inter_flag = False
						continue
		if type(self.intra_cm) ==dict:			
			for cur_im in self.im_type_list:
				avail_im_intra_cm = IM_CORR_INTRA.get(self.intra_cm[cur_im])
				if cur_im not in avail_im_intra_cm:
					print('GM_Simulator.cross_check_im_correlation: warning - {} is not available in {}'.format(cur_im, self.intra_cm))
					self.im_cm_intra_flag = False
					continue
		else:
			avail_im_intra_cm = IM_CORR_INTRA.get(self.intra_cm)
			if avail_im_intra_cm is not None:
				for cur_im in self.im_type_list:
					if cur_im not in avail_im_intra_cm:
						print('GM_Simulator.cross_check_im_correlation: warning - {} is not available in {}'.format(cur_im, self.intra_cm))
						self.im_cm_intra_flag = False
						continue

	def compute_inter_event_residual_ij(self, cm, im_name_list_1, im_name_list_2):
		if cm == 'Baker & Jayaram (2008)':
			rho = np.array([CorrelationModel.baker_jayaram_correlation_2008(im1, im2)
								for im1 in im_name_list_1 for im2 in im_name_list_2]).\
									reshape([len(im_name_list_1),\
				  							len(im_name_list_2)])
		elif cm == 'Baker & Bradley (2017)':
			rho = np.array([CorrelationModel.baker_bradley_correlation_2017(im1, im2)
								for im1 in im_name_list_1 for im2 in im_name_list_2]).\
									reshape([len(im_name_list_1),\
				  							len(im_name_list_2)])
		else:
				# TODO: extending this to more inter-event correlation models
			sys.exit('GM_Simulator.compute_inter_event_residual: currently supporting Baker & Jayaram (2008), Baker & Bradley (2017)')
		return rho

	def replace_submatrix(self, mat, ind1, ind2, mat_replace):
		for i, index in enumerate(ind1):
			mat[index, ind2] = mat_replace[i, :]
		return mat
	def compute_inter_event_residual(self):
		if type(self.inter_cm) == dict:
			rho = np.zeros([self.num_im, self.num_im])
			im_types = list(self.inter_cm.keys())
			for i in range(len(im_types)):
				for j in range(0,i+1):
					im_type_i = im_types[i]
					im_type_j = im_types[j]
					im_name_list_i = [im_name for im_name in self.im_name_list\
					   if im_name.startswith(im_type_i)]
					im_indices_i = [index for index, element in enumerate(self.im_name_list)\
					   if element.startswith(im_type_i)]
					im_name_list_j = [im_name for im_name in self.im_name_list\
					   if im_name.startswith(im_type_j)]
					im_indices_j = [index for index, element in enumerate(self.im_name_list)\
					   if element.startswith(im_type_j)]
					# In R2D, use SA(0.01) to approximate PGA
					im_name_list_i = ['SA(0.01)' if x == 'PGA' else x\
					    for x in im_name_list_i]
					im_name_list_j = ['SA(0.01)' if x == 'PGA' else x\
					    for x in im_name_list_j]
					rho_ij = self.compute_inter_event_residual_ij(\
						self.inter_cm[im_types[i]],im_name_list_i, im_name_list_j)
					rho = self.replace_submatrix(rho, im_indices_i, im_indices_j,\
									rho_ij)
					if i!=j:
						rho = self.replace_submatrix(rho, im_indices_j, im_indices_i,\
									rho_ij.T)
		else:
			rho = self.compute_inter_event_residual_ij(\
				self.inter_cm, self.im_name_list, self.im_name_list)
		# Simulating residuals
		with warnings.catch_warnings():
			# The intra-event models produce rho with tiny negative eigen values
			# This warning is suppressed
			warnings.filterwarnings("ignore", message="covariance is not symmetric positive-semidefinite.")
			residuals = np.random.multivariate_normal(np.zeros(self.num_im), rho, self.num_simu).T
		# return
		return residuals
	def compute_intra_event_residual_i(self, cm, im_name_list, num_simu):
		if cm == 'Jayaram & Baker (2009)':
			rho = np.zeros((self.num_sites, self.num_sites, len(im_name_list)))
			for i in range(self.num_sites):
				for j in range(self.num_sites):
					cur_stn_dist = self.stn_dist[i, j]
					for k in range(len(im_name_list)):
						rho[i, j, k] = CorrelationModel.jayaram_baker_correlation_2009(im_name_list[k], cur_stn_dist, 
						                                                               flag_clustering = False)
			# Simulating residuals
			residuals = np.zeros((self.num_sites, len(im_name_list), num_simu))
			for k in range(self.num_im):
				residuals[:, k, :] = np.random.multivariate_normal(np.zeros(self.num_sites), rho[:, :, k], num_simu).T
		elif cm == 'Loth & Baker (2013)':
			residuals = CorrelationModel.loth_baker_correlation_2013(self.sites,\
															im_name_list, num_simu)
		elif cm == 'Markhvida et al. (2017)':
			num_pc = 19
			residuals = CorrelationModel.markhvida_ceferino_baker_correlation_2017(\
				self.sites, im_name_list, num_simu, num_pc)
		elif cm == 'Du & Ning (2021)':
			num_pc = 23
			residuals = CorrelationModel.du_ning_correlation_2021(self.sites,\
												im_name_list, num_simu, num_pc)
		else:
				# TODO: extending this to more inter-event correlation models
			sys.exit('GM_Simulator.compute_intra_event_residual: currently supporting Jayaram & Baker (2009), Loth & Baker (2013),Markhvida et al. (2017), Du & Ning (2021)')
		return residuals

	def compute_intra_event_residual(self):
		if type(self.intra_cm) == dict:
			cm_groups = dict()
			# Group the IMs using the same cm
			for key, item in self.intra_cm.items():
				if item not in cm_groups.keys():
					cm_groups.update({item:[key]})
				else:
					cm_groups[item].append(key)	
			residuals = np.zeros((self.num_sites, self.num_im, self.num_simu))
			for cm, im_types in cm_groups.items():
				# im_type_list = [im_name.split('(')[0] for im_name in self.im_name_list]
				im_name_list = [im_name for im_name in self.im_name_list\
				   if im_name.split('(')[0] in im_types]
				im_indices = [index for index, element in enumerate(self.im_name_list)\
				   if element.split('(')[0] in im_types]
				residuals_i = self.compute_intra_event_residual_i(cm,\
													im_name_list, self.num_simu)
				for i, ind in enumerate(im_indices):
					residuals[:,ind,:] = residuals_i[:,i,:]
		else:
			residuals = self.compute_intra_event_residual_i(self.intra_cm,
											self.im_name_list, self.num_simu)
		# return
		return residuals


def export_im(stations, im_list, im_data, eq_data, output_dir, filename, csv_flag,\
			  gf_im_list, scenario_ids):
	# Rename SA(xxx) to SA_xxx
	for i, im in enumerate(im_list):
		if im.startswith('SA'):
			im_list[i] = im_list[i].split('(')[0] + '_' + im_list[i].split('(')[1][:-1]
	#try:
	# Station number
	num_stations = len(stations)
	# Scenario number
	num_scenarios = len(eq_data)
	eq_data = np.array(eq_data)
	# Saving large files to HDF while small files to JSON
	if num_scenarios > 100000:
		# Pandas DataFrame
		h_scenarios = ['Scenario-'+str(x) for x in range(1, num_scenarios + 1)]
		h_eq = ['Latitude', 'Longitude', 'Vs30', 'Magnitude', 'MeanAnnualRate','SiteSourceDistance','SiteRuptureDistance']
		for x in range(1, im_data[0][0, :, :].shape[1]+1):
			for y in im_list:
				h_eq.append('Record-'+str(x)+'-{}'.format(y))
		index = pd.MultiIndex.from_product([h_scenarios, h_eq])
		columns = ['Site-'+str(x) for x in range(1, num_stations + 1)]
		df = pd.DataFrame(index=index, columns=columns, dtype=float)
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
						tmp.append(y)
			df['Site-'+str(i+1)] = tmp
		# HDF output
		try:
			os.remove(os.path.join(output_dir, filename.replace('.json', '.h5')))
		except:
			pass
		hdf = pd.HDFStore(os.path.join(output_dir, filename.replace('.json', '.h5')))
		hdf.put('SiteIM', df, format='table', complib='zlib')
		hdf.close()
	else:
		res = []
		for i in range(num_stations):
			tmp = {'Location': {
					   'Latitude': stations[i]['lat'],
					   'Longitude': stations[i]['lon']
					   },
				   'Vs30': int(stations[i]['vs30'])
				  }
			tmp.update({'IMS': im_list})
			tmp_im = []
			for j in range(num_scenarios):
				tmp_im.append(np.ndarray.tolist(im_data[j][i, :, :]))
			if len(tmp_im) == 1:
				# Simplifying the data structure if only one scenario exists
				tmp_im = tmp_im[0]
			tmp.update({'lnIM': tmp_im})
			res.append(tmp)
		maf_out = []
		for ind, cur_eq in enumerate(eq_data):
			if cur_eq[1]:
				mar = cur_eq[1]
			else:
				mar = 'N/A'
			if cur_eq[2]:
				ssd = cur_eq[2]
			else:
				ssd = 'N/A'
			if len(cur_eq)>3 and cur_eq[3]:
				srd = cur_eq[3]
			else:
				srd = 'N/A'
			tmp = {'Magnitude': float(cur_eq[0]),
				   'MeanAnnualRate': mar,
				   'SiteSourceDistance': ssd,
				   'SiteRuputureDistance': srd,
				   'ScenarioIndex': int(scenario_ids[ind])}
			maf_out.append(tmp)
		res = {'Station_lnIM': res,
			   'Earthquake_MAF': maf_out}
		# save SiteIM.json
		with open(os.path.join(output_dir, filename), "w") as f:
			json.dump(res, f, indent=2)
	# export the event grid and station csv files
	if csv_flag:
		# output EventGrid.csv
		station_name = ['site'+str(stations[j]['ID'])+'.csv' for\
				   j in range(len(stations))]
		lat = [stations[j]['lat'] for j in range(len(stations))]
		lon = [stations[j]['lon'] for j in range(len(stations))]
		# vs30 = [stations[j]['vs30'] for j in range(len(stations))]
		# zTR = [stations[j]['DepthToRock'] for j in range(len(stations))]
		df = pd.DataFrame({
			'GP_file': station_name,
			'Longitude': lon,
			'Latitude': lat,
			# 'Vs30': vs30,
			# 'DepthToRock': zTR
		})
		# if cur_eq[2]:
		# 	df['SiteSourceDistance'] = cur_eq[2]
		output_dir = os.path.join(os.path.dirname(Path(output_dir)),
								os.path.basename(Path(output_dir)))
		# seperate directory for IM
		output_dir = os.path.join(output_dir, 'IMs')
		try:
			os.makedirs(output_dir)
		except:
			print('HazardSimulation: output folder already exists.')
		# save the csv
		df.to_csv(os.path.join(output_dir, 'EventGrid.csv'), index = False)
		# output station#.csv
		# csv header
		csvHeader = im_list
		for cur_scen in range(len(im_data)):
			if len(im_data) > 1:
				# IMPORTANT: the scenario index starts with 1 in the front end.
				cur_scen_folder = 'scenario'+str(int(scenario_ids[cur_scen])+1)
				try:
					os.mkdir(os.path.join(output_dir, cur_scen_folder))
				except:
					pass
					# print('ComputeIntensityMeasure: scenario folder already exists.')
				cur_output_dir = os.path.join(output_dir, cur_scen_folder)
			else:
				cur_output_dir = output_dir
			# current IM data
			cur_im_data = im_data[cur_scen]
			for i, site_id in enumerate(station_name):
				df = dict()
				# Loop over all intensity measures
				for cur_im_tag in range(len(csvHeader)):
					if (csvHeader[cur_im_tag].startswith('SA')) or \
						(csvHeader[cur_im_tag] in ['PGA', 'PGV']):
						df.update({
							csvHeader[cur_im_tag]: np.exp(cur_im_data[i, cur_im_tag, :])
						})
					else:
						df.update({
							csvHeader[cur_im_tag]: cur_im_data[i, cur_im_tag, :]
						})
				df = pd.DataFrame(df)
				# Combine PGD from liquefaction, landslide and fault
				if 'liq_PGD_h' in df.columns or 'ls_PGD_h'in df.columns or 'fd_PGD_h' in df.columns:
					PGD_h = np.zeros(df.shape[0])
					if 'liq_PGD_h' in df.columns:
						PGD_h += df['liq_PGD_h'].to_numpy()
					if 'ls_PGD_h' in df.columns:
						PGD_h += df['ls_PGD_h'].to_numpy()
					if 'fd_PGD_h' in df.columns:
						PGD_h += df['fd_PGD_h'].to_numpy()
					df['PGD_h'] = PGD_h
				if 'liq_PGD_v' in df.columns or 'ls_PGD_v'in df.columns or 'fd_PGD_v' in df.columns:
					PGD_v = np.zeros(df.shape[0])
					if 'liq_PGD_v' in df.columns:
						PGD_v += df['liq_PGD_v'].to_numpy()
					if 'ls_PGD_v' in df.columns:
						PGD_v += df['ls_PGD_v'].to_numpy()
					if 'fd_PGD_v' in df.columns:
						PGD_v += df['fd_PGD_v'].to_numpy()
					df['PGD_v'] = PGD_v
				colToDrop = []
				for col in df.columns:
					if (not col.startswith('SA')) and (col not in ['PGA', 'PGV',\
						'PGD_h', 'PGD_v']) and (col not in gf_im_list):
						colToDrop.append(col)
				df.drop(columns=colToDrop, inplace=True)
				# if 'liq_prob' in df.columns:
				# 	df.drop(columns=['liq_prob'], inplace=True)
				# if 'liq_susc' in df.columns:
				# 	df.drop(columns=['liq_susc'], inplace=True)
				df.fillna('NaN', inplace=True)
				df.to_csv(os.path.join(cur_output_dir, site_id), index = False)

		
		# output the site#.csv file including all scenarios
		if len(im_data) > 1:
			print('ComputeIntensityMeasure: saving all selected scenarios.')
			# lopp over sites
			for i, site_id in enumerate(station_name):
				df = dict()
				for cur_im_tag in range(len(csvHeader)):
					tmp_list = []
					# loop over all scenarios
					for cur_scen in range(len(im_data)):
						tmp_list = tmp_list + im_data[cur_scen][i, cur_im_tag, :].tolist()
					if (csvHeader[cur_im_tag].startswith('SA')) or \
						(csvHeader[cur_im_tag] in ['PGA', 'PGV']):
						df.update({
							csvHeader[cur_im_tag]: np.exp(tmp_list)
						})
					else:
						df.update({
							csvHeader[cur_im_tag]: tmp_list
						})
				df = pd.DataFrame(df)
				# Combine PGD from liquefaction, landslide and fault
				if 'liq_PGD_h' in df.columns or 'ls_PGD_h'in df.columns or 'fd_PGD_h' in df.columns:
					PGD_h = np.zeros(df.shape[0])
					if 'liq_PGD_h' in df.columns:
						PGD_h += df['liq_PGD_h'].to_numpy()
					if 'ls_PGD_h' in df.columns:
						PGD_h += df['ls_PGD_h'].to_numpy()
					if 'fd_PGD_h' in df.columns:
						PGD_h += df['fd_PGD_h'].to_numpy()
					df['PGD_h'] = PGD_h
				if 'liq_PGD_v' in df.columns or 'ls_PGD_v'in df.columns or 'fd_PGD_v' in df.columns:
					PGD_v = np.zeros(df.shape[0])
					if 'liq_PGD_v' in df.columns:
						PGD_v += df['liq_PGD_v'].to_numpy()
					if 'ls_PGD_v' in df.columns:
						PGD_v += df['ls_PGD_v'].to_numpy()
					if 'fd_PGD_v' in df.columns:
						PGD_v += df['fd_PGD_v'].to_numpy()
					df['PGD_v'] = PGD_v
				colToDrop = []
				for col in df.columns:
					if (not col.startswith('SA')) and (col not in ['PGA', 'PGV',\
						'PGD_h', 'PGD_v']) and (col not in gf_im_list):
						colToDrop.append(col)
				df.drop(columns=colToDrop, inplace=True)
				df.fillna('NaN', inplace=True)
				df.to_csv(os.path.join(output_dir, site_id), index = False)
	# return
	return 0
	#except:
		# return
		#return 1


def simulate_ground_motion(stations, im_raw, num_simu, correlation_info, im_info, eq_ids):

	# create a ground motion simulator
	gm_simulator = GM_Simulator(site_info=stations, num_simu=num_simu,\
							 correlation_info=correlation_info, im_info=im_info)
	ln_im_mr = []
	mag_maf = []
	t_start = time.time()
	for scen_i in tqdm(range(len(eq_ids)), desc=f"ComputeIntensityMeasure for {len(eq_ids)} scenarios"):
	# for i, cur_im_raw in enumerate(im_raw):
		# print('ComputeIntensityMeasure: Scenario #{}/{}'.format(i+1,len(im_raw)))
		cur_im_raw = im_raw[eq_ids[scen_i]]
		# set im_raw
		gm_simulator.set_im_raw(cur_im_raw)
		# Computing inter event residuals
		#t_start = time.time()
		epsilon = gm_simulator.compute_inter_event_residual()
		#print('ComputeIntensityMeasure: inter-event correlation {0} sec'.format(time.time() - t_start))
		# Computing intra event residuals
		#t_start = time.time()
		eta = gm_simulator.compute_intra_event_residual()
		#print('ComputeIntensityMeasure: intra-event correlation {0} sec'.format(time.time() - t_start))
		ln_im_all = np.zeros((gm_simulator.num_sites, gm_simulator.num_im, num_simu))
		for i in range(num_simu):
			epsilon_m = np.array([epsilon[:, i] for j in range(gm_simulator.num_sites)])
			ln_im_all[:, :, i] = gm_simulator.get_ln_im() + \
			                     gm_simulator.get_inter_sigma_im() * epsilon_m + \
								 gm_simulator.get_intra_sigma_im() * eta[:, :, i]

		ln_im_mr.append(ln_im_all)
		mag_maf.append([cur_im_raw['Magnitude'], cur_im_raw.get('MeanAnnualRate',None), 
		                cur_im_raw.get('SiteSourceDistance',None), cur_im_raw.get('SiteRuptureDistance',None)])
	
	print('ComputeIntensityMeasure: all inter- and intra-event correlation {0} sec'.format(time.time() - t_start))
	im_list = copy.copy(gm_simulator.im_name_list)
	# return
	return ln_im_mr, mag_maf, im_list


def compute_weighted_res(res_list, gmpe_weights):

	# compute weighted average of gmpe results
	# initialize the return res (these three attributes are identical in different gmpe results)
	res = {'Magnitude': res_list[0]['Magnitude'],
		   'MeanAnnualRate': res_list[0]['MeanAnnualRate'],
		   'SiteSourceDistance': res_list[0].get('SiteSourceDistance',None),
		   'Periods': res_list[0]['Periods'],
		   'IM': res_list[0]['IM']}
	# number of gmpe
	num_gmpe = len(res_list)
	# check number of weights
	if not (num_gmpe == len(gmpe_weights)):
		print('ComputeIntensityMeasure: please check the weights of different GMPEs.')
		return 1
	# site number
	num_site = len(res_list[0]['GroundMotions'])
	# loop over different sites
	gm_collector = []
	for site_tag in range(num_site):
		# loop over different GMPE
		tmp_res = {}
		for i, cur_res in enumerate(res_list):
			cur_gmResults = cur_res['GroundMotions'][site_tag]
			# get keys
			im_keys = list(cur_gmResults.keys())
			for cur_im in im_keys:
				if not (cur_im in list(tmp_res.keys())):
					if cur_im in ['Location','SiteData']:
						tmp_res.update({cur_im: cur_gmResults[cur_im]})
					else:
						tmp_res.update({cur_im: {}})
				if not (cur_im in ['Location','SiteData']):
					# get components
					comp_keys = list(cur_gmResults[cur_im].keys())
					# loop over differen components
					for cur_comp in comp_keys:
						if not (cur_comp in list(tmp_res[cur_im].keys())):
							tmp_res[cur_im].update({cur_comp: []})
							for cur_value in cur_gmResults[cur_im][cur_comp]:
								if 'StdDev' in cur_comp:
									# standard deviation
									tmp_res[cur_im][cur_comp].append(np.sqrt(cur_value ** 2.0 * gmpe_weights[i]))
								else:
									# mean
									tmp_res[cur_im][cur_comp].append(cur_value * gmpe_weights[i])
						else:
							for j, cur_value in enumerate(cur_gmResults[cur_im][cur_comp]):
								if 'StdDev' in cur_comp:
									# standard deviation
									tmp_res[cur_im][cur_comp][j] = np.sqrt(tmp_res[cur_im][cur_comp][j] ** 2.0 + cur_value ** 2.0 * gmpe_weights[i])
								else:
									# mean
									tmp_res[cur_im][cur_comp][j] = tmp_res[cur_im][cur_comp][j] + cur_value * gmpe_weights[i]
		# collector
		gm_collector.append(tmp_res)
	# res
	res.update({'GroundMotions': gm_collector})
	# return
	return res