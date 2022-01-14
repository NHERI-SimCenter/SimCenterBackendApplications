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
#

LOCAL_IM_GMPE = {"DS575H": ["Bommer, Stafford & Alarcon (2009)", "Afshari & Stewart (2016)"],
                 "DS595H": ["Bommer, Stafford & Alarcon (2009)", "Afshari & Stewart (2016)"],
				 "DS2080H": ["Afshari & Stewart (2016)"]}

OPENSHA_IM_GMPE = {"SA": ["Abrahamson, Silva & Kamai (2014)", "Boore, Stewart, Seyhan & Atkinson (2014)", 
                          "Campbell & Bozorgnia (2014)", "Chiou & Youngs (2014)"],
                   "PGA": ["Abrahamson, Silva & Kamai (2014)", "Boore, Stewart, Seyhan & Atkinson (2014)", 
				           "Campbell & Bozorgnia (2014)", "Chiou & Youngs (2014)"],
				   "PGV": ["Abrahamson, Silva & Kamai (2014)", "Boore, Stewart, Seyhan & Atkinson (2014)", 
				           "Campbell & Bozorgnia (2014)", "Chiou & Youngs (2014)"]}

IM_GMPE = {"LOCAL": LOCAL_IM_GMPE,
           "OPENSHA": LOCAL_IM_GMPE}

import os
import subprocess
import sys
import json
import numpy as np
from numpy.lib.utils import source
import pandas as pd
from gmpe import CorrelationModel, SignificantDurationModel
from FetchOpenSHA import *
from tqdm import tqdm
import time
from pathlib import Path
import copy

class IM_Calculator:

	def __init__(self, source_info=dict(), im_dict=dict(), gmpe_dict=dict(), 
	             gmpe_weights_dict=dict(), im_type=None, site_info=dict()):

		# basic set-ups
		self.set_source(source_info)
		self.set_im_gmpe(im_dict, gmpe_dict, gmpe_weights_dict)
		self.set_im_type(im_type)
		self.set_sites(site_info)
	
	def set_source(self, source_info):
		# set seismic source
		self.source_info = source_info
		# earthquake rupture forecast model (if any)
		self.erf = None
		if source_info.get('RuptureForecast', None):
			self.erf = getERF(source_info['RuptureForecast'], True)

	def set_im_gmpe(self, im_dict, gmpe_dict, gmpe_weights_dict):
		# set im and gmpe information
		self.im_dict = im_dict
		self.gmpe_dict = gmpe_dict
		self.gmpe_weights_dict = gmpe_weights_dict

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
			res_local, site_info_local = self.get_im_from_local(gmpe_list_local, im_type, cur_im_dict, self.erf, 
			                                                    self.site_info, gmpe_weights=gmpe_weights_list_local)
		else:
			res_local = dict()
			site_info_local = dict()
		if len(gmpe_list_opensha) > 0:
			res_opensha, site_info_opensha = self.get_im_from_opensha(self.source_info, gmpe_list_opensha, self.gmpe_dict.get('Parameters'), self.erf, 
			                                                          self.site_info, im_type, cur_im_dict, gmpe_weights=gmpe_weights_list_opensha)
		else:
			res_local = dict()
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
		return res, station_info

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
			eq_magnitude = erf.getSource(source_index).getRupture(rupture_index).getMag()
			# maf
			timeSpan = erf.getTimeSpan()
			meanAnnualRate = erf.getSource(source_index).getRupture(rupture_index).getMeanAnnualRate(timeSpan.getDuration())
		else:
			print('ComputeIntensityMeasure.get_im_from_local: error - source type {} not supported'.format(source_info['Type']))
			return res
		# sites
		site_list = station_info.get('SiteList')
		site_rup_dist = []
		for cur_site in site_list:
			cur_lat = cur_site['Location']['Latitude']
			cur_lon = cur_site['Location']['Longitude']
			# get distance
			if source_info['Type'] == 'PointSource':
				# no earth curvature is considered
				site_rup_dist.append(np.sqrt((eq_loc[0]-cur_lat)**2+(eq_loc[1]-cur_lon)^2+eq_loc[2]**2))
			else:
				site_rup_dist.append(get_rupture_distance(erf, source_index, rupture_index, [cur_lat], [cur_lon]))
		# evaluate gmpe
		for cur_gmpe in gmpe_list:
			gm_collector = []
			if cur_gmpe not in avail_gmpe:
				print('ComputeIntensityMeasure.get_im_from_local: warning - {} is not available.'.format(cur_gmpe))
				continue
			for i, cur_site in enumerate(site_list):
				# current site-rupture distance
				cur_dist = site_rup_dist[i]
				cur_vs30 = cur_site['Vs30']
				tmpResult = {'Mean': [],
				             'TotalStdDev': [],
							 'InterEvStdDev': [],
							 'IntraEvStdDev': []}
				gmResults = {"Location": cur_site["Location"],
				             "SiteData": {
								"Type": "Vs30",
								"Value": cur_vs30
							}}
				if cur_gmpe == 'Bommer, Stafford & Alarcon (2009)':
					mean, stdDev, interEvStdDev, intraEvStdDev = SignificantDurationModel.bommer_stafford_alarcon_ds_2009(magnitude=eq_magnitude, 
						distance=cur_dist, vs30=cur_vs30,duration_type=im_type)
					tmpResult['Mean'].append(float(np.log(mean)))
					tmpResult['TotalStdDev'].append(float(stdDev))
					tmpResult['InterEvStdDev'].append(float(interEvStdDev))
					tmpResult['IntraEvStdDev'].append(float(intraEvStdDev))
				elif cur_gmpe == 'Afshari & Stewart (2016)':
					mean, stdDev, interEvStdDev, intraEvStdDev = SignificantDurationModel.afshari_stewart_ds_2016(magnitude=eq_magnitude, 
						distance=cur_dist, vs30=cur_vs30, duration_type=im_type)
					tmpResult['Mean'].append(float(np.log(mean)))
					tmpResult['TotalStdDev'].append(float(stdDev))
					tmpResult['InterEvStdDev'].append(float(interEvStdDev))
					tmpResult['IntraEvStdDev'].append(float(intraEvStdDev))
				else:
					print('FetchOpenSHA.get_IM: gmpe_name {} is not supported.'.format(cur_gmpe))
				gmResults.update({'ln'+im_type: tmpResult})
				# collect sites
				gm_collector.append(gmResults)

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
			res.update(cur_res)
			res['IM'] = [res['IM']]
			res['Periods'] = res['Periods']
		else:
			res['IM'].append(cur_res['IM'])
			res['Periods'] = res['Periods']+cur_res['Periods']
			res['GroundMotions'] = res['GroundMotions']+cur_res['GroundMotions']

	print(res)
	# return
	return res


def get_im_dict(im_info):
	if im_info.get("Type", None) == "Vector":
		im_dict = im_info
		im_dict.pop('Type')
	else:
		# back compatibility
		im_dict = {im_info.get("Type"): im_info}

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
		im_keys = list(im_info.keys()).remove('Type')
		for cur_im in im_keys:
			cur_gmpe = im_info[cur_im].get("GMPE", None)
			cur_weights = im_info[cur_im].get("GMPEWeights", None)
			if cur_gmpe is None:
				print('ComputeIntensityMeasure.get_gmpe_from_im_vector: warning: GMPE not found for {}'.format(cur_im))
			else:
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
		gmpe_dict = {im_type: gmpe_list}
		gmpe_weights_dict = {im_type: gmpe_weights}
	# global parameters if any
	gmpe_dict.update({'Parameters': gmpe_info.get('Parameters',dict())})
	# return
	return gmpe_dict, gmpe_weights_dict


def compute_im(scenarios, stations, gmpe_info, im_info):

	# Calling OpenSHA to compute median PSA
	im_raw = []
	# Loading ERF model (if exists)
	erf = None
	if scenarios[0].get('RuptureForecast', None):
		erf = getERF(scenarios[0]['RuptureForecast'], True)
	# Stations
	station_list = [{
		'Location': {
			'Latitude': stations[j]['Latitude'],
			'Longitude': stations[j]['Longitude']
		}
	} for j in range(len(stations))]
	for j in range(len(stations)):
		if stations[j].get('Vs30'):
			station_list[j].update({'Vs30': int(stations[j]['Vs30'])})
	station_info = {'Type': 'SiteList',
					'SiteList': station_list}
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

	# create a IM calculator
	im_calculator = IM_Calculator(im_dict=im_dict, gmpe_dict=gmpe_dict, 
	                              gmpe_weights_dict=gmpe_weights_dict, site_info=station_info)
	
	# Loop over scenarios
	for i, s in enumerate(tqdm(scenarios, desc='Scenarios')):
		# Rupture
		source_info = scenarios[i]
		im_calculator.set_source(source_info)
		# Computing IM
		res_list = []
		for cur_im_type in list(im_dict.keys()):
			im_calculator.set_im_type(cur_im_type)
			res_list.append(im_calculator.calculate_im())
		# Collecting outputs
		im_raw.append(copy.deepcopy(collect_multi_im_res(res_list)))

	# Collecting station_info updates to staitons
	#for j in range(len(stations)):
	#	stations[j]['Latitude'] = station_info['SiteList'][j]['Location']['Latitude']
	#	stations[j]['Longitude'] = station_info['SiteList'][j]['Location']['Longitude']
	#	stations[j]['Vs30'] = station_info['SiteList'][j]['Vs30']
	# return
	return im_raw


def compute_inter_event_residual(sa_inter_cm, periods, num_simu, ims_type=[]):

	num_periods = len(periods)
	if sa_inter_cm in ['Baker & Jayaram (2008)', 'Baker & Bradley (2017)']:
		rho = np.array([CorrelationModel.baker_jayaram_correlation_2008(T1, T2)
						for T1 in periods for T2 in periods]).reshape([num_periods, num_periods])
		nim = 0	
		for cur_im in ims_type:
			cur_rho1 = np.array([CorrelationModel.baker_bradley_correlation_2017(T, im_type=cur_im)
			                    for T in periods]).reshape([1,num_periods])
			cur_rho1 = np.append(cur_rho1, np.zeros(nim)).reshape([1,num_periods+nim])		
			nim = nim+1
			rho = np.append(np.append(rho,cur_rho1,0).T,np.append(cur_rho1,np.zeros(nim)).reshape([1,num_periods+nim]),0).T
		cur_rho2 = np.array([CorrelationModel.baker_bradley_correlation_2017(0,im_type=cur_im1, im2_type=cur_im2)
		                    for cur_im1 in ims_type for cur_im2 in ims_type]).reshape([len(ims_type),len(ims_type)])
		rho[num_periods:,num_periods:] = cur_rho2

	else:
		# TODO: extending this to more inter-event correlation models
		print('ComputeIntensityMeaure.compute_inter_event_residual: currently only supporting Baker & Bradley (2017)')

	# Simulating residuals
	residuals = np.random.multivariate_normal(np.zeros(num_periods+nim), rho, num_simu).T
	# return
	return residuals


def compute_intra_event_residual(intra_cm, periods, station_data, num_simu, ims_type=[]):

	# Computing correlation coefficients
	num_stations = len(station_data)
	num_periods = len(periods)
	if intra_cm == 'Jayaram & Baker (2009)':
		rho = np.zeros((num_stations, num_stations, num_periods))
		for i in range(num_stations):
			loc_i = np.array([station_data[i]['Latitude'],
							  station_data[i]['Longitude']])
			for j in range(num_stations):
				loc_j = np.array([station_data[j]['Latitude'],
								  station_data[j]['Longitude']])
				# Computing station-wise distances
				stn_dist = np.linalg.norm(loc_i - loc_j) * 111.0
				for k in range(num_periods):
					rho[i, j, k] = \
						CorrelationModel.jayaram_baker_correlation_2009(periods[k],
							stn_dist, flag_clustering = False)
		# Simulating residuals
		residuals = np.zeros((num_stations, num_periods, num_simu))
		for k in range(num_periods):
			residuals[:, k, :] = np.random.multivariate_normal(np.zeros(num_stations),
															   rho[:, :, k], num_simu).T
	elif intra_cm == 'Loth & Baker (2013)':
		residuals = CorrelationModel.loth_baker_correlation_2013(station_data, periods, num_simu)

	elif intra_cm == 'Markhvida et al. (2017)':
		num_pc = 19
		residuals = CorrelationModel.markhvida_ceferino_baker_correlation_2017(station_data, periods, num_simu, num_pc)

	elif intra_cm == 'Du & Ning (2021)':
		num_pc = 23
		# pseudo periods map (11-PGA, 12-PGV, 13-Ia, 14-CAV, 15-DS575, 16-DS595)
		ims_list = ['PGA','PGV','Ia','CAV','DS575H','DS595H']
		for cur_im in ims_type:
			if cur_im in ims_list:
				print('CommputeIntensityMeasure.compute_intra_event_residual: ims_type not found {}'.append(cur_im))
				return None
			periods.append(cur_im)
		residuals = CorrelationModel.du_ning_correlation_2021(station_data, periods, num_simu, num_pc)

	# return
	return residuals


def export_im(stations, im_info, im_data, eq_data, output_dir, filename, csv_flag):

	#try:
		imType = im_info['Type']
		T = im_info['Periods']
		ims_type = im_info.get('IMS', [])
		# Station number
		num_stations = len(stations)
		# Scenario number
		num_scenarios = len(eq_data)
		# Saving large files to HDF while small files to JSON
		if num_scenarios > 100000:
			# Pandas DataFrame
			h_scenarios = ['Scenario-'+str(x) for x in range(1, num_scenarios + 1)]
			h_eq = ['Latitude', 'Longitude', 'Vs30', 'Magnitude', 'MeanAnnualRate','SiteSourceDistance','SiteRuptureDistance']
			for x in range(len(T)):
				h_eq.append('Period-{0}'.format(x+1))
			for x in range(1, im_data[0][0, :, :].shape[1]+1):
				for y in T:
					h_eq.append('Record-'+str(x)+'-lnSa-{0}s'.format(y))
			index = pd.MultiIndex.from_product([h_scenarios, h_eq])
			columns = ['Site-'+str(x) for x in range(1, num_stations + 1)]
			df = pd.DataFrame(index=index, columns=columns, dtype=float)
			# Data
			for i in range(num_stations):
				tmp = []
				for j in range(num_scenarios):
					tmp.append(stations[i]['Latitude'])
					tmp.append(stations[i]['Longitude'])
					tmp.append(int(stations[i]['Vs30']))
					tmp.append(eq_data[j][0])
					tmp.append(eq_data[j][1])
					tmp.append(eq_data[j][2])
					tmp.append(eq_data[j][3])
					for x in T:
						tmp.append(x)
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
						   'Latitude': stations[i]['Latitude'],
						   'Longitude': stations[i]['Longitude']
						   },
					   'Vs30': int(stations[i]['Vs30'])
					  }
				tmp.update({'Periods': T})
				tmp.update({'IMS': ims_type})
				tmp_im = []
				for j in range(num_scenarios):
					print(i)
					print(j)
					print(im_data[j])
					tmp_im.append(np.ndarray.tolist(im_data[j][i, :, :]))
				if len(tmp_im) == 1:
					# Simplifying the data structure if only one scenario exists
					tmp_im = tmp_im[0]
				tmp.update({'lnSa': tmp_im})
				res.append(tmp)
			maf_out = []
			for cur_eq in eq_data:
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
				tmp = {'Magnitdue': float(cur_eq[0]),
					   'MeanAnnualRate': mar,
					   'SiteSourceDistance': ssd,
					   'SiteRuputureDistance': srd}
				maf_out.append(tmp)
			res = {'Station_lnSa': res,
				   'Earthquake_MAF': maf_out}
			# save
			with open(os.path.join(output_dir, filename), "w") as f:
				json.dump(res, f, indent=2)

		# export the event grid and station csv files
		if csv_flag:
			# output EventGrid.csv
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
			if cur_eq[2]:
				df['SiteSourceDistance'] = cur_eq[2]
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
			csvHeader = []
			if imType == 'SA':
				for cur_T in T:
					csvHeader.append(imType + '(' + str(cur_T) + ')')
			else:
				csvHeader = [imType]
			csvHeader = csvHeader+ims_type
			for cur_scen in range(len(im_data)):
				if len(im_data) > 1:
					cur_scen_folder = 'scenario'+str(cur_scen+1)
					try:
						os.mkdir(os.path.join(output_dir, cur_scen_folder))
					except:
						print('ComputeIntensityMeasure: scenario folder already exists.')
					cur_output_dir = os.path.join(output_dir, cur_scen_folder)
				else:
					cur_output_dir = output_dir
				# current IM data
				cur_im_data = im_data[cur_scen]
				for i, site_id in enumerate(station_name):
					df = dict()
					# Loop over all intensity measures
					for cur_im_tag in range(len(csvHeader)):
						df.update({
							csvHeader[cur_im_tag]: np.exp(cur_im_data[i, cur_im_tag, :])
						})
					df = pd.DataFrame(df)
					df.to_csv(os.path.join(cur_output_dir, site_id), index = False)
			
			# output the site#.csv file including all scenarios
			if len(im_data) > 1:
				print('ComputeIntensityMeasure: saving all scenarios.')
				# lopp over sites
				for i, site_id in enumerate(station_name):
					df = dict()
					for cur_im_tag in range(len(csvHeader)):
						tmp_list = []
						# loop over all scenarios
						for cur_scen in range(len(im_data)):
							tmp_list = tmp_list + im_data[cur_scen][i, cur_im_tag, :].tolist()
						df.update({
							csvHeader[cur_im_tag]: np.exp(tmp_list)
						})
					df = pd.DataFrame(df)
					df.to_csv(os.path.join(output_dir, site_id), index = False)

		# return
		return 0
	#except:
		# return
		#return 1


def simulate_ground_motion(stations, im_raw, num_simu, correlation_info, im_info):

	# Sa inter-event model
	sa_inter_cm = correlation_info['SaInterEvent']
	# Sa intra-event model
	sa_intra_cm = correlation_info['SaIntraEvent']
	# Other IMs
	ims_type = im_info.get('IMS', [])
	# Periods
	periods = im_raw[0]['Periods']
	# Computing inter event residuals
	t_start = time.time()
	epsilon = compute_inter_event_residual(sa_inter_cm, periods, num_simu, ims_type=ims_type)
	print('ComputeIntensityMeasure: inter-event correlation {0} sec'.format(time.time() - t_start))
	# Computing intra event residuals
	t_start = time.time()
	eta = compute_intra_event_residual(sa_intra_cm, periods, stations, num_simu, ims_type=ims_type)
	print('ComputeIntensityMeasure: intra-event correlation {0} sec'.format(time.time() - t_start))
	ln_im_mr = []
	mag_maf = []
	for cur_im_raw in tqdm(im_raw, desc='Scenarios'):
		# Spectral data (median and dispersions)
		im_data = cur_im_raw['GroundMotions']
		# Combining inter- and intra-event residuals
		if 'SA' in im_info['Type']:
			ln_im = [im_data[i]['lnSA']['Mean'] for i in range(len(im_data))]
			inter_sigma_im = [im_data[i]['lnSA']['InterEvStdDev'] for i in range(len(im_data))]
			intra_sigma_im = [im_data[i]['lnSA']['IntraEvStdDev'] for i in range(len(im_data))]
		elif 'PGA' in im_info['Type']:
			ln_im = [im_data[i]['lnPGA']['Mean'] for i in range(len(im_data))]
			inter_sigma_im = [im_data[i]['lnPGA']['InterEvStdDev'] for i in range(len(im_data))]
			intra_sigma_im = [im_data[i]['lnPGA']['IntraEvStdDev'] for i in range(len(im_data))]
		elif 'PGV' in im_info['Type']:
			ln_im = [im_data[i]['lnPGV']['Mean'] for i in range(len(im_data))]
			inter_sigma_im = [im_data[i]['lnPGV']['InterEvStdDev'] for i in range(len(im_data))]
			intra_sigma_im = [im_data[i]['lnPGV']['IntraEvStdDev'] for i in range(len(im_data))]
		else:
			print('ComputeInensityMeasure: currently supporing spatial correlated SA and PGA.')
		ln_im_all = np.zeros((len(im_data), len(periods), num_simu))
		for i in range(num_simu):
			epsilon_m = np.array([epsilon[:, i] for j in range(len(im_data))])
			ln_im_all[:, :, i] = ln_im + inter_sigma_im * epsilon_m + intra_sigma_im * eta[:, :, i]

		ln_im_mr.append(ln_im_all)
		mag_maf.append([cur_im_raw['Magnitude'], cur_im_raw.get('MeanAnnualRate',None), 
		                cur_im_raw.get('SiteSourceDistance',None), cur_im_raw.get('SiteRuptureDistance',None)])
	# return
	return ln_im_mr, mag_maf


def simulate_storm(app_dir, input_dir, output_dir):

	file_list = os.listdir(app_dir)
	if sys.platform.startswith('win'):
		if any([i.endswith('exe') for i in file_list]):
			app = file_list[[i.endswith('exe') for i in file_list].index(True)]
		else:
			print('ComputeIntensityMeasure: please check the app for wind simulation.')
	elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
		if any([i.startswith('WindApp') for i in file_list]):
			app = file_list[[i.startswith('WindApp') for i in file_list].index(True)]
		else:
			print('ComputeIntensityMeasure: please check the app for wind simulation.')
	else:
		print('ComputeIntensityMeasure: system error.')
	try:
		os.mkdir(output_dir)
	except:
		print('ComputeIntensityMeasure: output directory already exists.')

	print([app_dir + app, '--input_dir', input_dir, '--output_dir', output_dir])
	_ = subprocess.call([app_dir + app, '--input_dir', input_dir,
						 '--output_dir', output_dir])

	return output_dir


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