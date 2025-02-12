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
import pandas as pd

# import the common constants and methods
this_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()  # noqa: PTH100, PTH120
main_dir = this_dir.parents[1]

sys.path.insert(0, str(main_dir / 'common'))
sys.path.insert(0, str(main_dir / 'common/groundMotionIM'))

import simcenter_common  # noqa: E402
from IntensityMeasureComputer import (  # noqa: E402
    IntensityMeasureComputer,
    load_records,
)


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
        print('No performance point found; curves do not intersect.')  # noqa: T201

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
    if aim['GeneralInformation'].get('units', None) is None:
        msg = 'No units defined in the AIM file'
        raise KeyError(msg)

    length_unit = aim['GeneralInformation']['units'].get('length', None)
    time_unit = aim['GeneralInformation']['units'].get('time', None)

    if length_unit is None:
        msg = 'No length units defined in the AIM file'
        raise KeyError(msg)

    if time_unit is None:
        msg = 'No time units defined in the AIM file'
        raise KeyError(msg)

    if length_unit == 'in':
        length_unit = 'inch'

    f_scale_im_user_to_cms = {}
    f_time_in = getattr(simcenter_common, time_unit, None)
    f_length_in = getattr(simcenter_common, length_unit, None)

    # ground motion accelerations are expected to use "g" unit
    gm_units = {'SA_0.3': 'g', 'SA_1.0': 'g'}

    for name, unit in gm_units.items():
        unit_type = None
        # print(name, unit)

        for base_unit_type, unit_set in simcenter_common.unit_types.items():
            if unit in unit_set:
                unit_type = base_unit_type

        if unit_type == 'acceleration':
            if unit == 'g':
                f_in = f_length_in / f_time_in**2.0
                f_out = 1 / simcenter_common.g
                f_scale_im_user_to_cms[name] = f_in * f_out
            else:
                msg = f'Unexpected internal unit: {unit}'
                raise ValueError(msg)
        else:
            f_scale_im_user_to_cms[name] = 1

    f_scale_edp_cms_to_user = {}
    f_scale_edp_cms_to_user['1-SA-1-1'] = simcenter_common.g / (
        f_length_in / f_time_in**2.0
    )
    f_scale_edp_cms_to_user['1-PRD-1-1'] = simcenter_common.inch / f_length_in

    # print(f'length:{length_unit}, time:{time_unit}, scale_in:{f_in}, scale_out:{f_out}')
    # print(f'scale_total:{f_scale_im_user_to_cms}')
    # print(f'scale_back:{f_scale_edp_cms_to_user}')

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
    beta_eff = 5
    beta_d = beta_eff

    capacity_data = {'iterations': []}

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

        iteration_data = {
            'capacity_spectrum': {'Sd': cap_sd.tolist(), 'Sa': cap_sa.tolist()},
            'demand_spectrum': {'Sd': dem_sd.tolist(), 'Sa': dem_sa.tolist()},
            'beta_eff': beta_eff,
            'performance_point': [perf_sd, perf_sa],
        }
        capacity_data['iterations'].append(iteration_data)

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

    with open('capacity_data.json', 'w') as f:  # noqa: PTH123
        json.dump(capacity_data, f, indent=2)

    return perf_sd, perf_sa


def determine_response(AIM_input_path, EVENT_input_path, EDP_input_path):  # noqa: C901, N803, D103
    # open the AIM file
    with open(AIM_input_path, encoding='utf-8') as f:  # noqa: PTH123
        AIM_in = json.load(f)  # noqa: N806
    applications = AIM_in['Applications']

    # open the event file and get the list of events
    with open(EVENT_input_path, encoding='utf-8') as f:  # noqa: PTH123
        EVENT_in = json.load(f)  # noqa: N806

    # use the first event
    evt = EVENT_in['Events'][0]

    # make sure we are working with a Seismic event
    if evt['type'] != 'Seismic':
        msg = f'Wrong Event Type, need Seismic NOT {evt['type']}'
        raise ValueError(msg)

    # get the simulation application
    SIM_input = applications['Simulation']  # noqa: N806
    if SIM_input['Application'] != 'CapacitySpectrumMethod2':
        msg = 'Wrong simulation application is called'
        raise ValueError(msg)
    SIM_input_data = SIM_input['ApplicationData']  # noqa: N806
    tol = SIM_input_data.get('tolerance', 0.05)
    max_iter = SIM_input_data.get('max_iter', 100)

    # get magnitude, use event file one over interface one
    if 'magnitude' in evt:
        event_magnitude = evt['magnitude']
    else:
        event_magnitude = SIM_input_data['DemandModel']['Parameters'][
            'EarthquakeMagnitude'
        ]

    # Identify the models to use
    # demand model
    demand_model_name = SIM_input_data['DemandModel']['Name']
    if demand_model_name not in ['HAZUS', 'HAZUS_lin_chang_2003']:
        msg = f'Unknown Demand Model: {demand_model_name}'
        raise ValueError(msg)

    # capacity model
    capacity_model_name = SIM_input_data['CapacityModel']['Name']
    if capacity_model_name != 'HAZUS_cao_peterson_2006':
        msg = f'Unknown Capacity Model: {capacity_model_name}'
        raise ValueError(msg)

    # damping model
    damping_model_name = SIM_input_data['DampingModel']['Name']
    if damping_model_name != 'HAZUS_cao_peterson_2006':
        msg = f'Unknown Damping Model: {damping_model_name}'
        raise ValueError(msg)

    general_info = AIM_in['GeneralInformation']

    roof_height = None
    if not pd.isna(general_info.get('RoofHeight')):
        roof_height = general_info['RoofHeight']
    elif not pd.isna(general_info.get('height')):
        roof_height = general_info['height']

    # we need to scale from input units to internal units
    units = general_info['units']
    length_unit_in = units['length']
    f_length_in = getattr(simcenter_common, length_unit_in, None)
    f_length_internal = getattr(simcenter_common, 'inch', None)
    roof_height *= f_length_in / f_length_internal

    # If roof height is not defined as an input, use Hazus estimates
    if roof_height is None:
        capacity_model = getattr(CapacityModels, capacity_model_name)(
            general_info=general_info
        )

        roof_height = capacity_model.get_hazus_roof_height()

        # Table 5-1 in Hazus provides roof height in feet, we need to convert to inches
        # because that unit is used by this code internally
        roof_height *= 12  # KUANSHI UNITS

    if not pd.isna(general_info.get('NumberOfStories')):
        num_stories = general_info['NumberOfStories']
    else:
        msg = 'Required feature NumberOfStories is not defined.'
        raise ValueError(msg)

    # TODO(TBD): This is a temporary bugfix because the MultiplePEER app only
    # provides event accelerations in "g. We need to fix the MultiPEER app
    # and then remove this fix from here
    if evt['subtype'] == 'MultiplePEER_Event':
        evt['timeSeries'][0]['factor'] *= simcenter_common.g / f_length_in

    #
    # determine SA 0.3 and SA 1.0
    #
    amp_scaled = False

    # convert to in/sec
    f_scale_im_user_to_cms, f_scale_edp_cms_to_user = find_unit_scale_factor(AIM_in)

    #
    # get spectrum values
    #

    # This takes care of scaling the event data with the factor in the event file
    time_series_dict = load_records(EVENT_in, amp_scaled)
    # print(len(time_series_dict['dirn1']))

    # This converts the time series from the provided units to cm/s2
    # for internal calcs in the IntensityMeasureComputer
    im_computer = IntensityMeasureComputer(
        time_hist_dict=time_series_dict, units=units, ampScaled=amp_scaled
    )

    periods = [0.0001, 0.3, 1.0]
    # This returns SA values in "g" by default. We should provide the desired units as im_units
    im_computer.compute_response_spectrum(periods=periods, im_units={})

    # print(json.dumps(im_computer.intensity_measures, indent=2))

    response_accel = im_computer.intensity_measures.get('AccelerationSpectrum', None)
    # After the various conversions above, the response_accel is coming out in "g"
    # as long as the original EVENT_in had the accelerations in the AIM units

    # set initial responses to 0, in case accel is not provided for all directions
    #  - dimension 2: 1 = x dir, 2 = y dir
    pga = [0, 0]
    drift_ratio1 = [0, 0]
    roof_sa1 = [0, 0]
    roof_disp1 = [0, 0]

    #
    # compute X dirn response
    #

    accel_x = None
    if 'accel_x' in response_accel:
        accel_x = response_accel['accel_x']
    elif 'dirn1' in response_accel:
        accel_x = response_accel['dirn1']

    if accel_x is not None:
        pga[0], sa_x_03, sa_x_10 = accel_x

        demand_model = getattr(DemandModels, demand_model_name)(event_magnitude)

        capacity_model = getattr(CapacityModels, capacity_model_name)(
            general_info=general_info
        )

        damping_model = getattr(DampingModels, damping_model_name)(
            demand_model, capacity_model
        )

        demand_model.set_IMs(
            sa_x_03, sa_x_10
        )  # this function almost surely expects SA in "g"
        demand_model.set_Tavb(damping_model)
        demand_model.set_beta_tvd(damping_model)

        # iterate to get sd and sa
        perf_sd_x, perf_sa_x = run_csm(
            demand_model, capacity_model, damping_model, tol, max_iter, 0
        )

        average_drift_x = perf_sd_x / (
            capacity_model.get_hazus_alpha2() * roof_height
        )
        drift_ratio1[0] = average_drift_x
        roof_sa1[0] = perf_sa_x
        roof_disp1[0] = average_drift_x * roof_height

    #
    # now y (or 2) dirn response
    #

    accel_y = None
    if 'accel_y' in response_accel:
        accel_y = response_accel['accel_y']
    elif 'dirn2' in response_accel:
        accel_y = response_accel['dirn2']

    if accel_y is not None:
        pga[1], sa_y_03, sa_y_10 = accel_y

        demand_model = getattr(DemandModels, demand_model_name)(event_magnitude)

        capacity_model = getattr(CapacityModels, capacity_model_name)(
            general_info=general_info
        )

        damping_model = getattr(DampingModels, damping_model_name)(
            demand_model, capacity_model
        )

        demand_model.set_IMs(sa_y_03, sa_y_10)
        demand_model.set_Tavb(damping_model)
        demand_model.set_beta_tvd(damping_model)

        # iterate to get sd and sa
        perf_sd_y, perf_sa_y = run_csm(
            demand_model, capacity_model, damping_model, tol, max_iter, 0
        )

        average_drift_y = perf_sd_y / (
            capacity_model.get_hazus_alpha2() * roof_height
        )
        drift_ratio1[1] = average_drift_y
        roof_sa1[1] = perf_sa_y
        roof_disp1[1] = average_drift_y * roof_height

    #
    # store the EDPs
    #

    # open EDP file

    with open(EDP_input_path, encoding='utf-8') as f:  # noqa: PTH123
        EDP_in = json.load(f)  # noqa: N806

    # calculate scale factor to convert accelerations back to output units
    sa_factor = simcenter_common.g / f_length_in

    # update EDP

    for edp_item in EDP_in['EngineeringDemandParameters'][0]['responses']:
        dofs = edp_item['dofs']

        if edp_item['type'] == 'max_abs_acceleration':
            floor = edp_item['floor']
            if isinstance(floor, str):
                floor = int(floor)

            if floor == num_stories:
                edp_item['scalar_data'] = [roof_sa1[i - 1] * sa_factor for i in dofs]

            elif floor == 0:
                edp_item['scalar_data'] = [pga[i - 1] * sa_factor for i in dofs]
            else:
                edp_item['scalar_data'] = [0.0 for i in dofs]

        elif edp_item['type'] == 'max_roof_drift' or edp_item['type'] == 'max_drift':
            edp_item['scalar_data'] = [drift_ratio1[i - 1] for i in dofs]

        else:
            edp_item['scalar_data'] = [0.0 for i in dofs]

    #
    # write EDP file and results.out
    #

    with open(EDP_input_path, 'w', encoding='utf-8') as f:  # noqa: PTH123
        json.dump(EDP_in, f, indent=2)

    #
    # write results file .. done by extract method in driver file
    # with open("results.out", 'w', encoding='utf-8') as f:
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
        sys.exit(
            determine_response(
                args.filenameAIM, args.filenameEVENT, args.filenameEDP
            )
        )
