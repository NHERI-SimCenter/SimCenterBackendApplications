#  # noqa: D100, INP001
# Copyright (c) 2023 Leland Stanford Junior University
# Copyright (c) 2023 The Regents of the University of California
#
# This file is part of pelicun.
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
# pelicun. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Jinyan Zhao
# Tamika Bassman
# Adam Zsarnóczay

import argparse
import json
import logging
import os
import sys
from pathlib import Path, PurePath

import CapacityModels
import DampingModels
import DemandModels
import numpy as np

# import the common constants and methods
this_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()  # noqa: PTH100, PTH120
main_dir = this_dir.parents[1]

sys.path.insert(0, str(main_dir / 'common'))
sys.path.insert(0, str(main_dir / 'common/groundMotionIM'))

import simcenter_common  # noqa: E402
from IntensityMeasureComputer  import IntensityMeasureComputer, load_records # noqa: E402

def find_performance_point(cap_x, cap_y, dem_x, dem_y, dd=0.001):
    """Interpolate to have matching discretization for cap/demand curves.

    Created by: Tamika Bassman.
    """
    # Interpolate to have matching discretization for cap/demand curves
    x_interp = np.arange(0, min(cap_x[-1], dem_x[-1]) + dd, dd)
    dem_y_interp = np.interp(x_interp, dem_x, dem_y)
    cap_y_interp = np.interp(x_interp, cap_x, cap_y)

    # # Enforce capacity curve to have same final length as spectrum
    # cap_y = cap_y[:min(len(cap_x),len(spec_x))]

    # Find sign changes in the difference between the two curves - these are
    # effectively intersections between the two curves
    curves_diff = dem_y_interp - cap_y_interp
    # adapted from https://stackoverflow.com/questions/4111412/how-do-i-get-a-list-of-indices-of-non-zero-elements-in-a-list
    id_sign_changes = [
        n
        for n, (i, j) in enumerate(zip(curves_diff[:-1], curves_diff[1:]))
        if i * j <= 0
    ]

    # id_sign_changes = []
    # for i,sign in enumerate(curves_diff[:-1]):
    #   if curves_diff[i]*curves_diff[i+1]<=0:
    #     # print(i)
    #     id_sign_changes += [i]

    # If sign changes detected, return the first (smallest abscissa) as the PP
    if len(id_sign_changes) > 0:
        ix = id_sign_changes[0]
        perf_x = x_interp[ix]
        perf_y = np.average([cap_y_interp[ix], dem_y_interp[ix]])
    elif dem_y_interp[0] > cap_y_interp[0]:
        perf_x = x_interp[-1]
        perf_y = cap_y_interp[-1]
    elif dem_y_interp[0] < cap_y_interp[0]:
        perf_x = 0.001  # x_interp[0]
        perf_y = 0.001  # cap_y_interp[0]
    else:
      print('No performance point found; curves do not intersect.')

    return perf_x, perf_y


def find_unit_scale_factor(aim):
    """Find the unit scale factor based on the AIM file.

    Args:
        AIM (dict): The AIM file content as a dictionary.

    Returns
    -------
        dict: A dictionary with the scale factors for different units.

    Raises
    ------
        KeyError: If 'units' or 'RegionalEvent' are not defined in the AIM file.
    """
    general_info = aim['GeneralInformation']
    if general_info.get('units', None) is None:
        msg = 'No units defined in the AIM file'
        raise KeyError(msg)
    units = general_info['units']
    length_units = units.get('length', None)
    time_units = units.get('time', None)
    if length_units == 'in':
        length_units='inch'

    f_scale_im_user_to_cms = {}
    f_time_in = getattr(simcenter_common, time_units, None)
    f_length_in = getattr(simcenter_common, length_units, None)
    
    gm_units={"SA_0.3":"g","SA_1.0":"g"}
    
    for name, unit in gm_units.items():
        unit_type = None
        print(name, unit)
        for base_unit_type, unit_set in simcenter_common.unit_types.items():
            if unit in unit_set:
                unit_type = base_unit_type
        # If the input event unit is acceleration, convert to g
        if unit_type == 'acceleration':
            f_in = f_length_in / f_time_in**2.0
            f_out = 1 / simcenter_common.g
            f_scale_im_user_to_cms[name] = f_in * f_out
        else:
            f_scale_im_user_to_cms[name] = 1
    f_scale_edp_cms_to_user = {}
    f_scale_edp_cms_to_user['1-SA-1-1'] = simcenter_common.g / (
        f_length_in / f_time_in**2.0
    )
    f_scale_edp_cms_to_user['1-PRD-1-1'] = simcenter_common.inch / f_length_in

    return f_scale_im_user_to_cms, f_scale_edp_cms_to_user


def run_csm(demand_model, capacity_model, damping_model, tol, max_iter, im_i):
    """Run the Capacity Spectrum Method (CSM) analysis.

    Args:
        demand_model (object): The demand model used in the analysis.
        capacity_model (object): The capacity model used in the analysis.
        damping_model (object): The damping model used in the analysis.
        tol (float): The tolerance for convergence.
        max_iter (int): The maximum number of iterations.
        im_i (int): The intensity measure index.

    Returns
    -------
        tuple: A tuple containing the effective damping ratio and the performance point.

    Raises
    ------
        ValueError: If the analysis does not converge within the maximum number of iterations.
    """
    beta_eff = damping_model.get_beta_elastic()
    beta_d = beta_eff

    # Track convergence
    iter_sd = []  # intermediate predictions of Sd @ PP
    iter_sa = []  # intermediate predictions of Sa @ PP
    # Iterate to find converged PP
    for i in range(max_iter):
        # Calc demand spectrum
        dem_sd, dem_sa = demand_model.get_reduced_demand(beta_eff)
        # create capacity curve
        cap_sd, cap_sa = capacity_model.get_capacity_curve(dem_sd[-1])
        # Calc intersection (PP)
        perf_sd, perf_sa = find_performance_point(cap_sd, cap_sa, dem_sd, dem_sa)
        iter_sd.append(perf_sd)
        iter_sa.append(perf_sa)

        # Calc effective damping at this point on the capacity curve
        beta_eff = damping_model.get_beta(perf_sd, perf_sa)

        # Check if tolerance met on damping ratios of capacity, demand cueves at this point
        if abs(beta_d - beta_eff) <= tol:
            # print('Final Iteration #%d' % (i+1))
            # print('IM realization  #%d' % (im_i))
            # print('Performance Point: (%.3f,%.3f)' % (perf_sd, perf_sa))
            # # print('Final Demand Spectrum Damping: %.3f' % beta_d)
            # print('Final Elastic Damping: %.3f' % damping_model.get_beta_elastic())
            # print('Final Capacity Curve Eff. Damping: %.3f' % beta_eff)
            # print('\n')
            break
        # If not met, adjust the demand spectrum accordingly and reiterate
        # print('Iteration #%d' % (i+1))
        # print('Performance Point: (%.3f,%.3f)' % (perf_sd, perf_sa))
        # print('Demand Spectrum Damping: %.3f' % beta_d)
        # print('Capacity Curve Eff. Damping: %.3f' % beta_eff)
        # print('\n')
        dem_sd, dem_sa = demand_model.get_reduced_demand(beta_eff)
        beta_d = beta_eff
        if i == max_iter - 1:
            logging.warning(
                f'The capacity spectrum method did not converge for the {im_i}th IM realization.'
            )

    return perf_sd, perf_sa


def determine_response(AIM_input_path, EVENT_input_path, EDP_input_path):  # noqa: C901, N802, N803, D103
    
    # open the AIM file
    with open(AIM_input_path, encoding='utf-8') as f:  # noqa: PTH123
        AIM_in = json.load(f)  # noqa: N806
    applications = AIM_in['Applications']

    # get the simulation application
    SIM_input = applications['Simulation']  # noqa: N806
    if SIM_input['Application'] != 'CapacitySpectrumMethod2':
        msg = 'Wrong simulation application is called'
        raise ValueError(msg)
    SIM_input_data = SIM_input['ApplicationData']  # noqa: N806
    tol = SIM_input_data.get('tolerance', 0.05)
    max_iter = SIM_input_data.get('max_iter', 100)

    # open the event file and get the list of events
    with open(EVENT_input_path, encoding='utf-8') as f:  # noqa: PTH123
        EVENT_in = json.load(f)  # noqa: N806

    evt = EVENT_in['Events'][0]

    # check the type of event
    evt_type = evt['type']

    if evt_type  != 'Seismic':
        msg = 'Wrong Event Type, need Seismic NOT ' + evt_type
        raise ValueError(msg)        

    # get magnitude, use event file one over interface one
    if "magnitude" in evt:
        Mw = evt["magnitude"]
    else:
        Mw=SIM_input_data['DemandModel']['Parameters']['EarthquakeMagnitude']

    #
    # determine SA .3 and SA 1.0
    #

    im_units=dict()
    ampScaled=False
    
    units = AIM_in['GeneralInformation'].get('units',None)

    time_series_dict = load_records(EVENT_in, ampScaled)
    im_computer = IntensityMeasureComputer(time_hist_dict=time_series_dict, units=units, ampScaled=ampScaled)
    periods=[0.3, 1.0]
    im_computer.compute_response_spectrum(periods=periods, im_units=im_units)
    print(im_computer.intensity_measures)
    
    pga = .2
    sa_03 = .2
    sa_10 = .21
        
    # open EDP file
    # open the event file and get the list of events
    with open(EDP_input_path, encoding='utf-8') as f:  # noqa: PTH123
        EDP_in = json.load(f)  # noqa: N806
    
    # convert to in/sec
    
    f_scale_im_user_to_cms, f_scale_edp_cms_to_user = find_unit_scale_factor(AIM_in)

    # the first column is Spectrum Acceleration, the second column is Spectrum Displacement
    EDP_output = np.zeros([1,2])

    # demand model
    demand_model_name = SIM_input_data['DemandModel']['Name']
    if demand_model_name in ['HAZUS', 'HAZUS_lin_chang_2003']:
        demand_model = getattr(DemandModels, demand_model_name)(Mw)

    # capacity model
    capacity_model_name = SIM_input_data['CapacityModel']['Name']
    if capacity_model_name == 'HAZUS_cao_peterson_2006':
        capacity_model = CapacityModels.HAZUS_cao_peterson_2006(
            general_info=AIM_in['GeneralInformation']
        )

    # damping model
    damping_model_name = SIM_input_data['DampingModel']['Name']
    if damping_model_name == 'HAZUS_cao_peterson_2006':
        damping_model = DampingModels.HAZUS_cao_peterson_2006(
            demand_model, capacity_model
        )

    demand_model.set_IMs(sa_03, sa_10)
    demand_model.set_Tavb(damping_model)
    demand_model.set_beta_tvd(damping_model)
    
    # if (damping_model_name == 'HAZUS_cao_peterson_2006'
    #     and capacity_model_name == 'HAZUS_cao_peterson_2006'):
    #     damping_model.set_HAZUS_bldg_type(capacity_model.get_hazus_bldg_type())

    # iterate to get sd and sa
    perf_sd, perf_sa = run_csm(
        demand_model, capacity_model, damping_model, tol, max_iter, 0
    )
    
    EDP_output[0, 0] = perf_sa

    # Table 5-1 in Hazus, convert to inches
    general_info = AIM_in['GeneralInformation']
    if general_info.get('RoofHeight', None) is not None:
        roof_height = general_info['RoofHeight']
    elif general_info.get('height', None) is not None:
        roof_height = general_info['height']        
    else:
        roof_height = capacity_model.get_hazus_roof_height() * 12

    num_stories = general_info['NumberOfStories']

    drift_ratio = perf_sd / capacity_model.get_hazus_alpha2() / roof_height
    EDP_output[0, 1] = drift_ratio

    roof_sa = perf_sa
    roof_disp = drift_ratio*roof_height
    
    ### Convert EDPs to the units defined in the AIM file
    print(EDP_in)
    print(EDP_output)

    #
    # store the IM(s) in the EDPs
    #

    print(num_stories)
    for edp_item in EDP_in['EngineeringDemandParameters'][0]['responses']:
        
        if edp_item['type'] == 'max_abs_acceleration':
            floor = edp_item['floor']
            if isinstance(floor,str):
                floor = int(floor)
                
            if floor == num_stories:            
                edp_item['scalar_data'] = [roof_sa]
            elif floor == 0:
                edp_item['scalar_data'] = [pga]            
            else:
                edp_item['scalar_data'] = [0.0]                

        elif edp_item['type'] == 'max_roof_drift':
            edp_item['scalar_data'] = [drift_ratio]

        elif edp_item['type'] == 'max_drift':
            edp_item['scalar_data'] = [drift_ratio]

        else:
            edp_item['scalar_data'] = [0.0]            

    #
    # write EDP file and results.out
    #
    
    with open(EDP_input_path, 'w', encoding='utf-8') as f:  # noqa: PTH123
        json.dump(EDP_in, f, indent=2)

    #
    # write results file .. done by extract method in driver file
    # with open("results.out", 'w', encoding='utf-8') as f:  # noqa: PTH123
    #    f.write(", ".join(map(str, results)))


        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameAIM', default=None)
    parser.add_argument('--filenameSAM', default=None)
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameEDP')
    parser.add_argument('--filenameSIM', default=None)
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.getRV:
        sys.exit(0)
    else:
        sys.exit(determine_response(args.filenameAIM, args.filenameEVENT, args.filenameEDP))
