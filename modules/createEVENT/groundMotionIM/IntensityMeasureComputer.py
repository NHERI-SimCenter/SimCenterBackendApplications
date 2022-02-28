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

import argparse, json, sys, os, bisect
import numpy as np
from pathlib import Path
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.stats.mstats import gmean
import pandas as pd

IM_TYPES = ['PeakGroundResponse','PseudoSpectrum','AriasIntensity','Duration','SpectralShape']
IM_MAP = {'PeakGroundResponse': ['PGA', 'PGV', 'PGD'],
          'PseudoSpectrum': ['PSA', 'PSV', 'PSD'],
          'AriasIntensity': ['Ia'],
          'Duration': ['DS575', 'DS595'],
          'SpectralShape': ['SaRatio']}

class IntensityMeasureComputer:

    def __init__(self, time_hist_dict=dict(), units=dict(), ampScaled=False):

        self.time_hist_dict = time_hist_dict
        self.units = units
        self._define_constants()

        # convert acc
        if 'acceleration' in list(units.keys()):
            from_acc_unit = units.get('acceleration')
        else:
            from_acc_unit = '{}/{}^2'.format(units['length'], units['time'])
        for cur_hist_name, cur_hist in self.time_hist_dict.items():
            cur_hist[2] = self.convert_accel_units(cur_hist[2], from_acc_unit).tolist()

        # initialize intensity measure dict
        self._init_intensity_measures()

    def _define_constants(self):

        self.km_sec_square = ("km/sec/sec", "km/sec**2", "km/sec^2")
        self.m_sec_square = ("m/sec/sec", "m/sec**2", "m/sec^2")
        self.cm_sec_square = ("cm/sec/sec", "cm/sec**2", "cm/sec^2")
        self.mm_sec_square = ("mm/sec/sec", "mm/sec**2", "mm/sec^2")
        self.in_sec_square = ("inch/sec/sec", "inch/sec**2", "inch/sec^2","in/sec/sec", "in/sec**2", "in/sec^2")
        self.ft_sec_square = ("ft/sec/sec", "ft/sec**2", "ft/sec^2")
        self.mile_sec_square = ("mile/sec/sec", "mile/sec**2", "mile/sec^2")
        self.g = 9.80665
        self.inch = 0.0254

    def _init_intensity_measures(self):

        # response spectra
        self.periods = dict()
        self.disp_spectrum = dict()
        self.vel_spectrum = dict()
        self.acc_spectrum = dict()
        self.psv = dict()
        self.psa = dict()
        # peak ground responses
        self.pga = dict()
        self.pgv = dict()
        self.pgd = dict()
        # arias intensity
        self.i_a = dict()
        # significant duration
        self.ds575 = dict()
        self.ds595 = dict()
        # saratio
        self.saratio = dict()

        # all
        self.intensity_measures = {
            'Periods': self.periods,
            'PSD': self.disp_spectrum,
            'VelocitySpectrum': self.vel_spectrum,
            'AccelerationSpectrum': self.acc_spectrum,
            'PSV': self.psv,
            'PSA': self.psa,
            'PGA': self.pga,
            'PGV': self.pgv,
            'PGD': self.pgd,
            'Ia': self.i_a,
            'DS575': self.ds575,
            'DS595': self.ds595,
            'SaRatio': self.saratio
        }

    def convert_accel_units(self, acceleration, from_, to_='cm/sec/sec'):
        """
        Converts acceleration from/to different units
        """      
        acceleration = np.asarray(acceleration)
        if from_ == 'g':
            if to_ == 'g':
                return acceleration
            if to_ in self.km_sec_square:
                return acceleration * self.g / 1000.0
            if to_ in self.m_sec_square:
                return acceleration * self.g
            if to_ in self.cm_sec_square:
                return acceleration * (100 * self.g)
            if to_ in self.mm_sec_square:
                return acceleration * (1000 * self.g)
            if to_ in self.in_sec_square:
                return acceleration * self.g / self.inch
            if to_ in self.ft_sec_square:
                return acceleration * self.g / (12.0 * self.inch)
            if to_ in self.mile_sec_square:
                return acceleration * self.g / (5280.0 * 12.0 * self.inch)

        elif from_ in self.km_sec_square:
            if to_ == 'g':
                return acceleration * 1000.0 / self.g
            if to_ in self.km_sec_square:
                return acceleration
            if to_ in self.m_sec_square:
                return acceleration * 1000.0
            if to_ in self.cm_sec_square:
                return acceleration * 1000.0 * 100.0
            if to_ in self.mm_sec_square:
                return acceleration * 1000.0 * 1000.0
            if to_ in self.in_sec_square:
                return acceleration * 1000.0 / self.inch
            if to_ in self.ft_sec_square:
                return acceleration * 1000.0 / (12.0 * self.inch)
            if to_ in self.mile_sec_square:
                return acceleration * 1000.0 / (5280.0 * 12.0 * self.inch)
            
        elif from_ in self.m_sec_square:
            if to_ == 'g':
                return acceleration / self.g
            if to_ in self.km_sec_square:
                return acceleration / 1000.0
            if to_ in self.m_sec_square:
                return acceleration
            if to_ in self.cm_sec_square:
                return acceleration * 100.0
            if to_ in self.mm_sec_square:
                return acceleration * 1000.0
            if to_ in self.in_sec_square:
                return acceleration / self.inch
            if to_ in self.ft_sec_square:
                return acceleration / (12.0 * self.inch)
            if to_ in self.mile_sec_square:
                return acceleration / (5280.0 * 12.0 * self.inch)
            
        elif from_ in self.cm_sec_square:
            if to_ == 'g':
                return acceleration / 100.0 / self.g
            if to_ in self.km_sec_square:
                return acceleration / 100.0 / 1000.0
            if to_ in self.m_sec_square:
                return acceleration / 100.0
            if to_ in self.cm_sec_square:
                return acceleration
            if to_ in self.mm_sec_square:
                return acceleration / 100.0 * 1000.0
            if to_ in self.in_sec_square:
                return acceleration / 100.0 / self.inch
            if to_ in self.ft_sec_square:
                return acceleration / 100.0 / (12.0 * self.inch)
            if to_ in self.mile_sec_square:
                return acceleration / 100.0 / (5280.0 * 12.0 * self.inch)

        elif from_ in self.mm_sec_square:
            if to_ == 'g':
                return acceleration / 1000.0 / self.g
            if to_ in self.km_sec_square:
                return acceleration / 1000.0 / 1000.0
            if to_ in self.m_sec_square:
                return acceleration / 1000.0
            if to_ in self.cm_sec_square:
                return acceleration / 10.0
            if to_ in self.mm_sec_square:
                return acceleration
            if to_ in self.in_sec_square:
                return acceleration / 1000.0 / self.inch
            if to_ in self.ft_sec_square:
                return acceleration / 1000.0 / (12.0 * self.inch)
            if to_ in self.mile_sec_square:
                return acceleration / 1000.0 / (5280.0 * 12.0 * self.inch)

        elif from_ in self.in_sec_square:
            if to_ == 'g':
                return acceleration * self.inch / self.g
            if to_ in self.km_sec_square:
                return acceleration * self.inch / 1000.0
            if to_ in self.m_sec_square:
                return acceleration * self.inch
            if to_ in self.cm_sec_square:
                return acceleration * self.inch * 100.0
            if to_ in self.mm_sec_square:
                return acceleration * self.inch * 1000.0
            if to_ in self.in_sec_square:
                return acceleration
            if to_ in self.ft_sec_square:
                return acceleration / 12.0
            if to_ in self.mile_sec_square:
                return acceleration / (5280.0 * 12.0)

        elif from_ in self.ft_sec_square:
            if to_ == 'g':
                return acceleration * 12.0 * self.inch / self.g
            if to_ in self.km_sec_square:
                return acceleration * 12.0 * self.inch / 1000.0
            if to_ in self.m_sec_square:
                return acceleration * 12.0 * self.inch
            if to_ in self.cm_sec_square:
                return acceleration * 12.0 * self.inch * 100.0
            if to_ in self.mm_sec_square:
                return acceleration * 12.0 * self.inch * 1000.0
            if to_ in self.in_sec_square:
                return acceleration * 12.0
            if to_ in self.ft_sec_square:
                return acceleration
            if to_ in self.mile_sec_square:
                return acceleration / 5280.0

        elif from_ in self.mile_sec_square:
            if to_ == 'g':
                return acceleration * 5280 * 12.0 * self.inch / self.g
            if to_ in self.km_sec_square:
                return acceleration * 5280 * 12.0 * self.inch / 1000.0
            if to_ in self.m_sec_square:
                return acceleration * 5280 * 12.0 * self.inch
            if to_ in self.cm_sec_square:
                return acceleration * 5280 * 12.0 * self.inch * 100.0
            if to_ in self.mm_sec_square:
                return acceleration * 5280 * 12.0 * self.inch * 1000.0
            if to_ in self.in_sec_square:
                return acceleration * 5280 * 12.0
            if to_ in self.ft_sec_square:
                return acceleration * 5280
            if to_ in self.mile_sec_square:
                return acceleration

        raise ValueError(f"Unrecognized unit {from_}")

    def compute_response_spectrum(self, periods=[], damping=0.05):

        # note this function assumes acceleration in cm/sec/sec
        # psa is in g, psv in cm/sec
        if len(periods)==0:
            return
        elif type(periods)==list:
            periods = np.array(periods)
        num_periods = len(periods)

        for cur_hist_name, cur_hist in self.time_hist_dict.items():
            dt = cur_hist[1]
            ground_acc = cur_hist[2]
            num_steps = len(ground_acc)
            # discritize
            dt_disc = 0.005
            num_steps_disc = int(np.floor(num_steps*dt/dt_disc))
            f = interp1d([dt*x for x in range(num_steps)], ground_acc, bounds_error=False, fill_value=(ground_acc[0], ground_acc[-1]))
            tmp_time = [dt_disc*x for x in range(num_steps_disc)]
            ground_acc = f(tmp_time)
            # circular frequency, damping, and stiffness terms
            omega = (2*np.pi)/periods
            cval = damping*2*omega
            kval = ((2*np.pi)/periods)**2
            # Newmark-Beta
            accel = np.zeros([num_steps_disc, num_periods])
            vel = np.zeros([num_steps_disc, num_periods])
            disp = np.zeros([num_steps_disc, num_periods])
            a_t = np.zeros([num_steps_disc, num_periods])
            accel[0, :] =(-ground_acc[0] - (cval * vel[0, :])) - (kval * disp[0, :])
            for j in range(1, num_steps_disc):
                delta_acc = ground_acc[j]-ground_acc[j-1]
                delta_d2u = (-delta_acc-dt_disc*cval*accel[j-1,:]-dt_disc*kval*(vel[j-1,:]+0.5*dt_disc*accel[j-1,:]))/ \
                    (1.0+0.5*dt_disc*cval+0.25*dt_disc**2*kval)
                delta_du = dt_disc*accel[j-1,:]+0.5*dt_disc*delta_d2u
                delta_u = dt_disc*vel[j-1,:]+0.5*dt_disc**2*accel[j-1,:]+0.25*dt_disc**2*delta_d2u
                accel[j,:] = delta_d2u+accel[j-1,:]
                vel[j,:] = delta_du+vel[j-1,:]
                disp[j,:] = delta_u+disp[j-1,:]
                a_t[j, :] = ground_acc[j] + accel[j, :]
            # collect data
            self.disp_spectrum.update({cur_hist_name: np.ndarray.tolist(np.max(np.fabs(disp), axis=0))})
            self.vel_spectrum.update({cur_hist_name: np.ndarray.tolist(np.max(np.fabs(vel), axis=0))})
            self.acc_spectrum.update({cur_hist_name: np.ndarray.tolist(np.max(np.fabs(a_t), axis=0)/100.0/self.g)})
            self.psv.update({cur_hist_name: np.ndarray.tolist(omega*np.max(np.fabs(disp), axis=0))})
            self.psa.update({cur_hist_name: np.ndarray.tolist(omega**2*np.max(np.fabs(disp), axis=0)/100.0/self.g)})
            self.periods.update({cur_hist_name: periods.tolist()})

    def compute_peak_ground_responses(self):

        # note this function assumes acceleration in cm/sec/sec
        # pga is in g, pgv in cm/sec, pgd in cm
        for cur_hist_name, cur_hist in self.time_hist_dict.items():
            dt = cur_hist[1]
            ground_acc = cur_hist[2]
            num_steps = len(ground_acc)
            # integral
            velocity = dt * cumtrapz(ground_acc, initial=0.)
            displacement = dt * cumtrapz(velocity, initial=0.)
            # collect data
            self.pga.update({cur_hist_name: np.max(np.fabs(ground_acc))/self.g/100.0})
            self.pgv.update({cur_hist_name: np.max(np.fabs(velocity))})
            self.pgd.update({cur_hist_name: np.max(np.fabs(displacement))})

    def compute_arias_intensity(self):
        
        # note this function assumes acceleration in cm/sec/sec and return Arias Intensity in m/sec
        for cur_hist_name, cur_hist in self.time_hist_dict.items():
            dt = cur_hist[1]
            ground_acc = cur_hist[2]
            num_steps = len(ground_acc)
            tmp = [x**2/100/100 for x in ground_acc]
            # integral
            I_A = np.pi / 2 / self.g * dt * cumtrapz(tmp, initial=0.)
            # collect data
            self.i_a.update({cur_hist_name: np.max(np.fabs(I_A))})
            # compute significant duration
            ds575, ds595 = self._compute_significant_duration(I_A, dt)
            self.ds575.update({cur_hist_name: ds575})
            self.ds595.update({cur_hist_name: ds595})

    def _compute_significant_duration(self, I_A, dt):

        # note this function return duration in sec
        ds575 = 0.0
        ds595 = 0.0
        # normalize
        I_A_n = I_A / np.max(I_A)
        # find 5%, 75%, 95%
        id5 = next(x for x, val in enumerate(I_A_n) if val > 0.05)
        id75 = next(x for x, val in enumerate(I_A_n) if val > 0.75)
        id95 = next(x for x, val in enumerate(I_A_n) if val > 0.95)
        # compute ds
        ds575 = dt*(id75-id5)
        ds595 = dt*(id95-id5)
        # return
        return ds575, ds595

    def compute_saratio(self, T1 = 1.0, Ta = 0.02, Tb = 3.0):

        if len(self.psa) == 0:
            return

        # period list for SaRatio calculations
        period_list = [0.01*x for x in range(1500)]
        period_list = [x for x in period_list if x <= Tb and x >= Ta]

        for cur_hist_name, cur_hist in self.time_hist_dict.items():
            cur_psa = self.psa.get(cur_hist_name, None)
            cur_periods = self.periods.get(cur_hist_name, None)
            if (cur_psa is None) or (cur_periods is None):
                # put zero if the psa is empty
                self.saratio.update({cur_hist_name: 0.0})
            else:
                f = interp1d(cur_periods, cur_psa)
                self.saratio.update({cur_hist_name: f(T1)/gmean(f(period_list))})               


def load_records(event_file, ampScaled):

    event_data = event_file.get('Events', None)
    if event_data is None:
        raise ValueError(f"IntensityMeasureComputer: 'Events' attribute is not found in EVENT.json")
    else:
        event_data = event_data[0]
    
    # check type
    if event_data.get('type', None) != 'Seismic':
        return dict()
    
    # get time series attribute
    time_series = event_data.get('timeSeries', None)
    if time_series is None:
        return dict()
    ts_names = [x['name'] for x in time_series]
    
    # collect time series tags
    pattern = event_data.get('pattern', None)
    if pattern is None:
        raise ValueError(f"IntensityMeasureComputer: 'pattern' is not found in EVENT.json")
    dict_ts = dict()
    for cur_pat in pattern:
        dict_ts.update({cur_pat['timeSeries']: [cur_pat['dof']]})
    
    # get time series (currently only for horizontal directions)
    for cur_ts in list(dict_ts.keys()):
        try:
            cur_id = ts_names.index(cur_ts)
        except:
            raise ValueError(f"IntensityMeasureComputer: {cur_ts} is not found in 'timeSeries' in EVENT.json")
        # get amplitude scaling (if the record is raw, i.e., ampScaled is false)
        if not ampScaled:
            scalingFactor = time_series[cur_id].get('factor',1.0)
        else:
            scalingFactor = 1.0
        # append that record
        dict_ts[cur_ts].append(time_series[cur_id]['dT'])
        dict_ts[cur_ts].append([x*scalingFactor for x in time_series[cur_id]['data']])

    # return
    return dict_ts


def main(BIM_file, EVENT_file, IM_file, unitScaled, ampScaled):

    # load BIM file
    try:
        with open(BIM_file, 'r') as f:
            bim_file = json.load(f)
    except:
        raise ValueError(f"IntensityMeasureComputer: cannot load BIM file {BIM_file}")
    
    # load EVENT file
    try:
        with open(EVENT_file, 'r') as f:
            event_file = json.load(f)
    except:
        raise ValueError(f"IntensityMeasureComputer: cannot load EVENT file {EVENT_file}")

    # get periods
    bim_event = bim_file['Events']
    if type(bim_event)==list:
        bim_event = bim_event[0]
    """
    periods = bim_event.get('SpectrumPeriod',[0.01,0.02,0.03,0.04,0.05,0.075,
                                              0.1,0.2,0.3,0.4,0.5,0.75,1.0,
                                              2.0,3.0,4.0,5.0,7.5,10.0])
    """

    # get units
    if unitScaled:
        # corresponding to records after SimCenterEvent.py
        units = bim_file['GeneralInformation'].get('units', None)
        if units is None:
            raise ValueError(f"IntensityMeasureComputer: units is not found in {BIM_file}")
    else:
        # corresponding to raw records (e.g., EE-UQ)
        units = {"acceleration": "g"}

    # get IM list (will be user-defined)
    im_types = [] # IM type
    im_names = ['Periods'] # IM name
    bim_im = bim_file.get('IntensityMeasure', None)
    output_periods = []
    if bim_im is None:
        # search it again under UQ_Method/surrogateMethodInfo
        bim_im = bim_file['UQ_Method']['surrogateMethodInfo'].get('IntensityMeasure',None)
    if bim_im is None or len(bim_im)==0:
        # no intensity measure calculation requested
        return
    else:
        for cur_im in list(bim_im.keys()):
            for ref_type in IM_TYPES:
                if cur_im in IM_MAP.get(ref_type):
                    im_names.append(cur_im)
                    if ref_type not in im_types:
                        im_types.append(ref_type)
                    if cur_im.startswith('PS'):
                        periods = bim_im[cur_im].get('Periods',[0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 0.3, 
                                                                0.4, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0])
                        output_periods = periods
                    if cur_im=='SaRatio':
                        tmp = bim_im[cur_im].get('Periods',[0.02, 1.0, 3.0])
                        Ta, Tb = [np.min(tmp), np.max(tmp)]
                        tmp.pop(tmp.index(Ta))
                        tmp.pop(tmp.index(Tb))
                        T1 = tmp[0]
                        periods = [Ta+0.01*(x-1) for x in range(int(np.ceil((Tb-Ta)/0.01))+3)]
                    break
        for Ti in output_periods:
            if Ti not in periods:
                bisect.insort(periods,Ti)

    for cur_type in im_types:
        if cur_type not in IM_TYPES:
            # pop the non-supported IMs
            im_types.pop(cur_type)
    
    # load records
    dict_time_series = load_records(event_file, ampScaled)

    # intensity measure computer
    im_computer = IntensityMeasureComputer(time_hist_dict=dict_time_series, units=units, ampScaled=ampScaled)

    # compute intensity measures
    if 'PeakGroundResponse' in im_types:
        im_computer.compute_peak_ground_responses()
    if 'PseudoSpectrum' in im_types or 'SpectralShape' in im_types:
        im_computer.compute_response_spectrum(periods=periods)
    if 'AriasIntensity' in im_types or 'Duration' in im_types:
        im_computer.compute_arias_intensity()
    if 'SpectralShape' in im_types:
        im_computer.compute_saratio(T1=T1, Ta=Ta, Tb=Tb)

    # pop not requested IMs
    for cur_im in list(im_computer.intensity_measures.keys()):
        if cur_im not in im_names:
            im_computer.intensity_measures.pop(cur_im)

    # save a IM.json
    out_data = {'IntensityMeasure': im_computer.intensity_measures}
    with open(IM_file, 'w') as f:
        json.dump(out_data, f, indent=2)

    # save a csv file
    csv_dict = dict()
    colname = []
    for cur_im in im_types:
        colname = colname+IM_MAP.get(cur_im, [])
    im_dict = im_computer.intensity_measures
    for cur_hist_name, cur_hist in dict_time_series.items():
        cur_colname = []
        cur_dof = cur_hist[0]
        cur_periods = im_dict['Periods'].get(cur_hist_name)
        for cur_im in im_names:
            if cur_im in IM_MAP.get('PseudoSpectrum'):
                if len(output_periods) > 0:
                    for Ti in output_periods:
                        cur_im_T = '{}({}s)'.format(cur_im, Ti)
                        tmp_key = '1-{}-0-{}'.format(cur_im_T,cur_dof)
                        # interp
                        f = interp1d(cur_periods, im_dict.get(cur_im).get(cur_hist_name))
                        if tmp_key in csv_dict.keys():
                            csv_dict[tmp_key].append(f(Ti))
                        else:
                            csv_dict.update({tmp_key: [f(Ti)]})
            elif cur_im == 'Periods':
                pass
            else:
                tmp_key = '1-{}-0-{}'.format(cur_im,cur_dof)
                if tmp_key in csv_dict.keys():
                    csv_dict[tmp_key].append(im_dict.get(cur_im).get(cur_hist_name))
                else:
                    csv_dict.update({tmp_key: [im_dict.get(cur_im).get(cur_hist_name)]})
    # create df
    csv_df = pd.DataFrame.from_dict(csv_dict)
    tmp_idx = IM_file.index('.')
    if tmp_idx:
        filenameCSV = IM_file[:tmp_idx]+'.csv'
    else:
        filenameCSV = IM_file+'.csv'
    csv_df.to_csv(filenameCSV,index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        "Read SimCenterEvent EVENT.json files and "
        "compute intensity measures for ground motions time histories and "
        "append intensity measures to the EVENT.json",
        allow_abbrev = False
    )

    # BIM file - getting units
    parser.add_argument('--filenameBIM', help = "Name of the BIM file")
    # Event file - getting time histories
    parser.add_argument('--filenameEVENT', help = "Name of the EVENT file")
    # IM file - getting time histories
    parser.add_argument('--filenameIM', help = "Name of the IM file")
    # unit scaled tag
    parser.add_argument('--unitScaled', default=False, help = "Records have been scaled in units")
    # amplitude scaled tag
    parser.add_argument('--ampScaled', default=False, help="Records have been scaled in amplitudes")
    
    # parse arguments
    args = parser.parse_args()

    # run and return
    sys.exit(main(args.filenameBIM, args.filenameEVENT, args.filenameIM, args.unitScaled, args.ampScaled))
