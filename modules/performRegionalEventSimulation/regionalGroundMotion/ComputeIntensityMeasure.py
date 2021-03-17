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
import numpy as np
import pandas as pd
from gmpe import CorrelationModel
from FetchOpenSHA import *
from tqdm import tqdm
import time


def compute_spectra(scenarios, stations, gmpe_info, im_info):

	# Calling OpenSHA to compute median PSA
	psa_raw = []
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
	siteSpec, sites, site_prop = get_site_prop(gmpe_info['Type'], station_list)
	# Loop over scenarios
	for i, s in enumerate(tqdm(scenarios, desc='Scenarios')):
		# Rupture
		source_info = scenarios[i]
		# Computing IM
		res, station_info = get_IM(gmpe_info, erf, sites, siteSpec, site_prop, source_info, station_info, im_info)
		# Collecting outputs
		psa_raw.append(res)

	# Collecting station_info updates to staitons
	for j in range(len(stations)):
		stations[j]['Latitude'] = station_info['SiteList'][j]['Location']['Latitude']
		stations[j]['Longitude'] = station_info['SiteList'][j]['Location']['Longitude']
		stations[j]['Vs30'] = station_info['SiteList'][j]['Vs30']
	# return
	return psa_raw, stations


def compute_inter_event_residual(sa_inter_cm, periods, num_simu):

	num_periods = len(periods)
	if sa_inter_cm == 'Baker & Jayaram (2008)':
		rho = np.array([CorrelationModel.baker_jayaram_correlation_2008(T1, T2)
		                for T1 in periods for T2 in periods]).reshape([num_periods, num_periods])
	else:
		# TODO: extending this to more inter-event correlation models
		print('ComputeIntensityMeaure: currently only supporting Baker & Jayaram (2008)')

	# Simulating residuals
	residuals = np.random.multivariate_normal(np.zeros(num_periods), rho, num_simu).T
	# return
	return residuals


def compute_intra_event_residual(sa_intra_cm, periods, station_data, num_simu):

	# Computing correlation coefficients
	num_stations = len(station_data)
	num_periods = len(periods)
	if sa_intra_cm == 'Jayaram & Baker (2009)':
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
	elif sa_intra_cm == 'Loth & Baker (2013)':
		residuals = CorrelationModel.loth_baker_correlation_2013(station_data, periods, num_simu)

	elif sa_intra_cm == 'Markhvida et al. (2017)':
		num_pc = 19
		residuals = CorrelationModel.markhvida_ceferino_baker_correlation_2017(station_data, periods, num_simu, num_pc)

    # return
	return residuals


def export_im(stations, T, im_data, eq_data, output_dir, filename):

	#try:
	    # Station number
		num_stations = len(stations)
		# Scenario number
		num_scenarios = len(eq_data)
		# Saving large files to HDF while small files to JSON
		if num_scenarios > 10:
			# Pandas DataFrame
			h_scenarios = ['Scenario-'+str(x) for x in range(1, num_scenarios + 1)]
			h_eq = ['Latitude', 'Longitude', 'Vs30', 'Magnitude', 'MeanAnnualRate']
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
				tmp_im = []
				for j in range(num_scenarios):
					tmp_im.append(np.ndarray.tolist(im_data[j][i, :, :]))
				if len(tmp_im) == 1:
					# Simplifying the data structure if only one scenario exists
					tmp_im = tmp_im[0]
				tmp.update({'lnSa': tmp_im})
				res.append(tmp)
			maf_out = []
			for cur_eq in eq_data:
				tmp = {'Magnitdue': cur_eq[0],
				       'MeanAnnualRate': cur_eq[1]}
				maf_out.append(tmp)
			res = {'Station_lnSa': res,
			       'Earthquake_MAF': maf_out}
			# save
			with open(os.path.join(output_dir, filename), "w") as f:
				json.dump(res, f, indent=2)
		# return
		return 0
	#except:
		# return
		#return 1


def simulate_ground_motion(stations, psa_raw, num_simu, correlation_info, im_info):

    # Sa inter-event model
	sa_inter_cm = correlation_info['SaInterEvent']
	# Sa intra-event model
	sa_intra_cm = correlation_info['SaIntraEvent']
	# Periods
	periods = psa_raw[0]['Periods']
	# Computing inter event residuals
	t_start = time.time()
	epsilon = compute_inter_event_residual(sa_inter_cm, periods, num_simu)
	print('ComputeIntensityMeasure: inter-event correlation {0} sec'.format(time.time() - t_start))
	# Computing intra event residuals
	t_start = time.time()
	eta = compute_intra_event_residual(sa_intra_cm, periods, stations, num_simu)
	print('ComputeIntensityMeasure: intra-event correlation {0} sec'.format(time.time() - t_start))
	ln_psa_mr = []
	mag_maf = []
	for cur_psa_raw in tqdm(psa_raw, desc='Scenarios'):
		# Spectral data (median and dispersions)
		sa_data = cur_psa_raw['GroundMotions']
	    # Combining inter- and intra-event residuals
		if 'SA' in im_info['Type']:
			ln_sa = [sa_data[i]['lnSA']['Mean'] for i in range(len(sa_data))]
			ln_sa = [sa_data[i]['lnSA']['Mean'] for i in range(len(sa_data))]
			inter_sigma_sa = [sa_data[i]['lnSA']['InterEvStdDev'] for i in range(len(sa_data))]
			intra_sigma_sa = [sa_data[i]['lnSA']['IntraEvStdDev'] for i in range(len(sa_data))]
		elif 'PGA' in im_info['Type']:
			ln_sa = [sa_data[i]['lnPGA']['Mean'] for i in range(len(sa_data))]
			ln_sa = [sa_data[i]['lnPGA']['Mean'] for i in range(len(sa_data))]
			inter_sigma_sa = [sa_data[i]['lnPGA']['InterEvStdDev'] for i in range(len(sa_data))]
			intra_sigma_sa = [sa_data[i]['lnPGA']['IntraEvStdDev'] for i in range(len(sa_data))]
		else:
			print('ComputeInensityMeasure: currently supporing spatial correlated SA and PGA.')
		ln_psa = np.zeros((len(sa_data), len(periods), num_simu))
		for i in range(num_simu):
			epsilon_m = np.array([epsilon[:, i] for j in range(len(sa_data))])
			ln_psa[:, :, i] = ln_sa + inter_sigma_sa * epsilon_m + intra_sigma_sa * eta[:, :, i]

		ln_psa_mr.append(ln_psa)
		mag_maf.append([cur_psa_raw['Magnitude'], cur_psa_raw['MeanAnnualRate']])
    # return
	return ln_psa_mr, mag_maf


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
