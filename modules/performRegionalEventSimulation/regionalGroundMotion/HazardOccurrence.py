# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Leland Stanford Junior University
# Copyright (c) 2022 The Regents of the University of California
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

import numpy as np
import pulp
import pandas as pd
import json, os, itertools, threading, collections, time
from scipy.stats import norm
from USGS_API import *
from tqdm import tqdm


def configure_hazard_occurrence(input_dir, 
                                output_dir,
                                hzo_config=None,
                                site_config=None,
                                mth_flag=True):

    if hzo_config is None or site_config is None:
        # no model is defined
        return {}
    # model type
    model_type = hzo_config.get('Model')
    # number of earthquake in the subset
    num_target_eqs = hzo_config.get('EarthquakeSampleSize',10)
    # number of ground motion maps
    num_target_gmms = hzo_config.get('GroundMotionMapSize',num_target_eqs*10)
    # return periods
    return_periods = hzo_config.get('ReturnPeriods',None)
    if return_periods is None:
        return {}
    # im type
    im_type = hzo_config.get('IntensityMeasure',None)
    if im_type is None:
        return {}
    # period (if any)
    period = hzo_config.get('Period',0.0)
    if im_type == 'SA':
        cur_imt = im_type+"{:.1f}".format(period).replace('.','P')
    else:
        cur_imt = im_type
    # get hazard curve input
    hc_input = hzo_config.get('HazardCurveInput',None)
    # return periods
    cur_edition = hzo_config.get('Edition','E2014')
    if hc_input is None:
        return {}
    elif hc_input == 'Inferred':
        # fecthing hazard curve from usgs
        hazard_curve_collector = []
        for site_id in range(len(site_config)):
            cur_site = site_config[site_id]
            cur_lon = cur_site.get('Longitude')
            cur_lat = cur_site.get('Latitude')
            cur_vs30 = cur_site.get('Vs30',760)
            hazard_curve_collector.append(USGS_HazardCurve(longitude=cur_lon,
                                                           latitude=cur_lat,
                                                           vs30=cur_vs30,
                                                           edition=cur_edition,
                                                           imt=cur_imt,
                                                           tag=site_id))
        hc_data = []
        print('HazardOCcurrence: fetching USGS hazard curve for individual sites - this may take a while.')
        t_start = time.time()
        if mth_flag:
            num_bins = 100
            bin_size = int(np.ceil(len(hazard_curve_collector)/num_bins))
            ids_list = []
            collector_list = []
            sub_ths = []
            hc_dict = {}
            for k in range(0, len(hazard_curve_collector), bin_size):
                ids_list.append(list(range(k,k+bin_size)))
                collector_list.append(hazard_curve_collector[k:k+bin_size])
            #print(ids_list)
            for i in range(len(ids_list)):
                th = threading.Thread(target=fetch_usgs_hazard_curve_para, args=(ids_list[i], collector_list[i], hc_dict))
                sub_ths.append(th)
                th.start()
            for th in sub_ths:
                th.join()
            # order the res_dict by id
            res_ordered = collections.OrderedDict(sorted(hc_dict.items()))
            for i, cur_res in res_ordered.items():
                hc_data.append(cur_res)
        else:
            for i in range(len(hazard_curve_collector)):
                cur_collector = hazard_curve_collector[i]
                if cur_collector.fetch_url():
                    hc_data.append(cur_collector.get_hazard_curve())
                else:
                    print('HazardOCcurrence: error in fetching hazard curve for site {}.'.format(i))
                    return

        print('HazardOCcurrence: all hazard curves fetched {0} sec.'.format(time.time()-t_start))
    else:
        hc_input = os.path.join(input_dir,hc_input)
        if hc_input.endswith('.csv'):
            hc_data = get_hazard_curves(input_csv=hc_input)
        elif hc_input.endswith('.json'):
            hc_data = get_hazard_curves(input_json=hc_input)
        else:
            hc_data = get_hazard_curves(input_dir=hc_input)
    # interpolate the hazard curve with the return periods
    num_sites = len(hc_data)
    num_rps = len(return_periods)
    hc_interp = np.zeros((num_sites,num_rps))
    ln_maf = [np.log(x) for x in return_periods]
    for i in range(num_sites):
        ln_cur_maf = [np.log(x) for x in hc_data[i].get('ReturnPeriod')]
        ln_cur_sa = np.log(hc_data[i].get('IM')).tolist()
        hc_interp[i,:] = np.exp(np.interp(ln_maf,ln_cur_maf,ln_cur_sa,left=ln_cur_sa[0],right=ln_cur_sa[-1]))
    hc_interp_list = hc_interp.tolist()
    # summary
    occ_dict = {
        "Model": model_type,
        "NumTargetEQs": num_target_eqs,
        "NumTargetGMMs": num_target_gmms,
        "ReturnPeriods": return_periods,
        "IntensityMeasure": im_type,
        "Period": period,
        "HazardCurves": hc_interp_list
    }
    # output the hazard occurrence information file
    with open(os.path.join(output_dir,"HazardCurves.json"),"w") as f:
        json.dump(occ_dict, f, indent=2)
    occ_dict = {
        "Model": model_type,
        "NumTargetEQs": num_target_eqs,
        "NumTargetGMMs": num_target_gmms,
        "ReturnPeriods": return_periods,
        "IntensityMeasure": im_type,
        "Period": period,
        "HazardCurves": hc_interp
    }
    # return
    return occ_dict


def fetch_usgs_hazard_curve_para(ids, hc_collectors, hc_dict):

    for cur_id, cur_collector in zip(ids, hc_collectors):
        if cur_collector.fetch_url():
            hc_dict[cur_id] = cur_collector.get_hazard_curve()
        else:
            print('HazardOCcurrence: error in fetching hazard curve for site {}.'.format(cur_id))
    # return
    return


def get_hazard_curves(input_dir=None, 
                      input_csv=None, 
                      input_json=None):

    if input_dir is not None:
        return

    if input_csv is not None:
        df_hc = pd.read_csv(input_csv,header=None)
        num_sites = df_hc.shape[0]-1
        return_periods = df_hc.iloc[0,1:].to_numpy().tolist()
        hc_data = []
        for i in range(num_sites):
            hc_data.append({
                "SiteID": i,
                "ReturnPeriod": return_periods,
                "IM": df_hc.iloc[i+1,1:].to_numpy().tolist()
            })
        return hc_data

    if input_json is not None:
        with open(input_json, 'r') as f:
            hc_data = json.load(f)
        return hc_data


# KZ-08/23/22: adding a function for computing exceeding probability at an im level
def get_im_exceedance_probility(im_raw, 
                                im_type, 
                                period, 
                                im_level):

    # number of scenarios
    num_scen = len(im_raw)

    # number of intensity levels
    num_rps = im_level.shape[1]
    
    # initialize output
    num_sites = len(im_raw[0].get('GroundMotions'))
    im_exceedance_prob = np.zeros((num_sites,num_scen,num_rps))

    # check IM type first
    if im_type not in im_raw[0].get('IM'):
        print('IM_Calculator.get_im_exceedance_probility: error - intensity measure {} type does not match to {}.'.format(im_type,im_raw[0].get('IM')))
        return im_exceedance_prob
        
    # check period
    if period not in im_raw[0].get('Periods'):
        print('IM_Calculator.get_im_exceedance_probility: error - period {} does not match to {}.'.format(period,im_raw[0].get('Periods')))
        return im_exceedance_prob
    else:
        periodID = im_raw[0].get('Periods').index(period)

    # start to compute the exceedance probability
    for k in range(num_scen):
        allGM = im_raw[k].get('GroundMotions')
        for i in range(num_sites):
            curIM = allGM[i].get('ln{}'.format(im_type))
            curMean = curIM.get('Mean')[periodID]
            curStd = curIM.get('TotalStdDev')[periodID]
            im_exceedance_prob[i,k,:] = 1.0-norm.cdf(np.log(im_level[i,:]),loc=curMean,scale=curStd)
    # return
    return im_exceedance_prob


def get_im_exceedance_probability_gm(im_raw,
                                     im_list,
                                     im_type,
                                     period, 
                                     im_level):

    # get periodID
    for i in range(len(im_list)):
        if im_type in im_list[i]:
            if im_type == 'SA' and float(im_list[i].split('(')[1][:-1]) == period:
                periodID = i
                break
            else:
                periodID = i

    # number of intensity levels
    num_rps = im_level.shape[1]

    # get the exceedance probability table now
    num_scen = len(im_raw)
    num_site = im_raw[0].shape[0]
    num_simu = im_raw[0].shape[-1]
    im_exceedance_prob = np.zeros((num_site,num_simu*num_scen,num_rps))
    #print('im_exceedance_prob_gm.shape=',im_exceedance_prob)
    for i in range(num_scen):
        for j in range(num_site):
            curIM = im_raw[i][j,periodID,:]
            for k in range(num_simu):
                im_exceedance_prob[j,i*num_simu+k,:] = [int(x) for x in curIM[k]>im_level[j,:]]

    # return
    return im_exceedance_prob


def sample_earthquake_occurrence(model_type,
                                 num_target_eqs,
                                 return_periods,
                                 im_exceedance_prob):

    # model type
    if model_type == 'Manzour & Davidson (2016)':
        # create occurrence model
        om = OccurrenceModel_ManzourDavidson2016(return_periods=return_periods,
                                                 im_exceedance_probs=im_exceedance_prob,
                                                 num_scenarios=num_target_eqs)
        # solve the optimiation
        om.solve_opt()
    else:
        print('HazardOccurrence.get_im_exceedance_probility: {} is not available yet.')
        return None

    return om


def export_sampled_earthquakes(id_selected_eqs, eqdata, P, output_dir=None):

    selected_eqs = []
    for i in id_selected_eqs:
        selected_eqs.append(eqdata[i])
    dict_selected_eqs = {
        'EarthquakeNumber': len(id_selected_eqs),
        'EarthquakeID': id_selected_eqs,
        'EarthquakeInfo': selected_eqs,
        'ProbabilityWeight': [P[x] for x in id_selected_eqs],
    }

    if output_dir is not None:
        with open(os.path.join(output_dir,'RupSampled.json'), 'w') as f:
            json.dump(dict_selected_eqs, f, indent=2)

def export_sampled_gmms(id_selected_gmms, id_selected_scens, P, output_dir=None):

    dict_selected_gmms = {
        'EarthquakeID': id_selected_scens,
        'ProbabilityWeight': [P[x] for x in id_selected_gmms]
    }

    if output_dir is not None:
        with open(os.path.join(output_dir,'InfoSampledGM.json'), 'w') as f:
            json.dump(dict_selected_gmms, f, indent=2)
    

class OccurrenceModel_ManzourDavidson2016:

    def __init__(self, 
                 return_periods = [],
                 im_exceedance_probs = [],
                 num_scenarios = -1):
        """
        __init__: initialization a hazard occurrence optimizer
        :param return_periods: 1-D array of return periods, RP(r)
        :param earthquake_mafs: 1-D array of annual occurrence probability, MAF(j)
        :param im_exceedance_probs: 3-D array of exceedance probability of Sa, EP(i,j,r) for site #i, earthquake #j, return period #r
        :param num_scenarios: integer for number of target scenarios
        """
        # read input parameters
        self.return_periods = return_periods
        self.im_exceedance_probs = im_exceedance_probs
        self.num_eqs = self.im_exceedance_probs.shape[1]
        self.num_scenarios = num_scenarios
        # check input parameters
        self.input_valid = self._input_check()
        if not self.input_valid:
            print('OccurrenceModel_ManzourDavidson2016.__init__: at least one input parameter invalid.')
            return

    def _input_check(self):
        """
        _input_check: check of input parameters
        """
        # number of return periods
        if len(self.return_periods) > 0:
            self.num_return_periods = len(self.return_periods)
            print('OccurrenceModel_ManzourDavidson2016._input_check: number of return periods = {}.'.format(self.num_return_periods))
        else:
            print('OccurrenceModel_ManzourDavidson2016._input_check: no return period is defined.')
            return False
        # shape of exceedance probability
        if len(self.im_exceedance_probs.shape) != 3:
            print('OccurrenceModel_ManzourDavidson2016._input_check: exceedance probability array should be 3-D.')
            return False
        else:
            if self.im_exceedance_probs.shape[-1] != len(self.return_periods):
                print('OccurrenceModel_ManzourDavidson2016._input_check: exceedance probability array should have dimensions of (#site, #eq, #return_period).')
                return False
            else:
                self.num_sites = self.im_exceedance_probs.shape[0]
                print('OccurrenceModel_ManzourDavidson2016._input_check: number of sites = {}.'.format(self.num_sites))
        # number of target scenarios
        if self.num_scenarios <= 0:
            print('OccurrenceModel_ManzourDavidson2016._input_check: number of target scenarios should be positive.')
            return False
        else:
            # initialize outputs
            init_flag = False
            init_flag = self._opt_initialization()
            if init_flag:
                print('OccurrenceModel_ManzourDavidson2016._input_check: initialization completed.')
                return True
            else:
                print('OccurrenceModel_ManzourDavidson2016._input_check: initialization errors.')
                return False

    def _opt_initialization(self):
        """
        _opt_initialization: intialization of optimization problem
        """
        # the problem is mixed integer program
        self.prob = pulp.LpProblem("MIP", pulp.LpMinimize)

        # variables
        self.e_plus = {}
        self.e_minus = {}
        self.e_plus_name = {}
        self.e_minus_name = {}
        for i in range(self.num_sites):
            for j in range(self.num_return_periods):
                self.e_plus_name[i,j] = 'ep-{}-{}'.format(i,j)
                self.e_minus_name[i,j] = 'en-{}-{}'.format(i,j)
                self.e_plus[i,j] = pulp.LpVariable(self.e_plus_name[i,j], 0, None)
                self.e_minus[i,j] = pulp.LpVariable(self.e_minus_name[i,j], 0, None)
        self.P = {}
        self.Z = {}
        self.P_name = {}
        self.Z_name = {}
        for i in range(self.num_eqs):
            self.P_name[i] = 'p-{}'.format(i)
            self.Z_name[i] = 'z-{}'.format(i)
            self.P[i] = pulp.LpVariable(self.P_name[i], 0, 1)
            self.Z[i] = pulp.LpVariable(self.Z_name[i], 0, 1, pulp.LpBinary)

        # objective function
        comb_sites_rps = list(itertools.product(range(self.num_sites),range(self.num_return_periods)))
        self.prob += pulp.lpSum(self.return_periods[j]*self.e_plus[(i,j)]+self.return_periods[j]*self.e_minus[(i,j)] for (i,j) in comb_sites_rps)

        # constraints
        for i in range(self.num_sites):
            for j in range(self.num_return_periods):
                self.prob += pulp.lpSum(self.P[k]*self.im_exceedance_probs[i,k,j] for k in range(self.num_eqs))+self.e_minus[i,j]-self.e_plus[i,j] == 1.0/self.return_periods[j]

        for i in range(self.num_eqs):
            self.prob += self.P[i]-self.Z[i] <= 0

        self.prob += pulp.lpSum(self.Z[i] for i in range(self.num_eqs)) == self.num_scenarios

        return True


    def solve_opt(self):
        """
        target_function: compute the target function to be minimized
        :param X: 2-D array of annual occurrence probability of earthquakes and corresponding binary variables (many values are reduced to zeros)
        """
        maximum_runtime = 1*60*60 # 1 hours maximum
        self.prob.solve(pulp.PULP_CBC_CMD(timeLimit=maximum_runtime, gapRel=0.001))
        print("Status:", pulp.LpStatus[self.prob.status])

    def get_selected_earthquake(self):

        P_selected = [self.P[i].varValue for i in range(self.num_eqs)]
        Z_selected = [self.Z[i].varValue for i in range(self.num_eqs)]

        return P_selected, Z_selected

    def get_error_vector(self):

        e_plus_selected = {}
        e_minus_selected = {}
        for i in range(self.num_sites):
            for j in range(self.num_return_periods):
                e_plus_selected[i,j] = self.e_plus[i,j].varValue
                e_minus_selected[i,j] = self.e_minus[i,j].varValue