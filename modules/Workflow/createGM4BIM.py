#
# Copyright (c) 2019 The Regents of the University of California
#
# This file is part of the RDT Application.
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
# You should have received a copy of the BSD 3-Clause License along with the
# RDT Application. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Chaofeng Wang
# fmk

import argparse
import json
import os
import shutil
import sys
from glob import glob

# import the common constants and methods
from pathlib import Path

import numpy as np
import pandas as pd
from computeResponseSpectrum import *

this_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
main_dir = this_dir.parents[0]
sys.path.insert(0, str(main_dir / 'common'))
from simcenter_common import *


def get_scale_factors(input_units, output_units):
    """Determine the scale factor to convert input event to internal event data"""
    # special case: if the input unit is not specified then do not do any scaling
    if input_units is None:
        scale_factors = {'ALL': 1.0}

    else:
        # parse output units:

        # if no length unit is specified, 'inch' is assumed
        unit_length = output_units.get('length', 'inch')
        f_length = globals().get(unit_length, None)
        if f_length is None:
            raise ValueError(f'Specified length unit not recognized: {unit_length}')

        # if no time unit is specified, 'sec' is assumed
        unit_time = output_units.get('time', 'sec')
        f_time = globals().get(unit_time, None)
        if f_time is None:
            raise ValueError(f'Specified time unit not recognized: {unit_time}')

        scale_factors = {}

        for input_name, input_unit in input_units.items():
            # exceptions
            if input_name == 'factor':
                f_scale = 1.0

            else:
                # get the scale factor to standard units
                f_in = globals().get(input_unit, None)
                if f_in is None:
                    raise ValueError(
                        f'Input unit for event files not recognized: {input_unit}'
                    )

                unit_type = None
                for base_unit_type, unit_set in globals()['unit_types'].items():
                    if input_unit in unit_set:
                        unit_type = base_unit_type

                if unit_type is None:
                    raise ValueError(f'Failed to identify unit type: {input_unit}')

                # the output unit depends on the unit type
                if unit_type == 'acceleration':
                    f_out = f_time**2.0 / f_length

                elif unit_type == 'speed':
                    f_out = f_time / f_length

                elif unit_type == 'length':
                    f_out = 1.0 / f_length

                else:
                    raise ValueError(
                        f'Unexpected unit type in workflow: {unit_type}'
                    )

                # the scale factor is the product of input and output scaling
                f_scale = f_in * f_out

            scale_factors.update({input_name: f_scale})

    return scale_factors


def createFilesForEventGrid(inputDir, outputDir, removeInputDir):
    if not os.path.isdir(inputDir):
        print(f'input dir: {inputDir} does not exist')
        return 0

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    #
    # FMK bug fix - have to copy AIM files back to the inputDir dir as code below assumes they are there
    #

    extension = 'AIM.json'
    the_dir = os.path.abspath(inputDir)
    for item in os.listdir(the_dir):
        item_path = os.path.join(the_dir, item)
        if os.path.isdir(item_path):
            template_dir = os.path.join(item_path, 'templatedir')
            for the_file in os.listdir(template_dir):
                if the_file.endswith(extension):
                    bim_path = os.path.join(template_dir, the_file)
                    shutil.copy(bim_path, the_dir)

    # siteFiles = glob(f"{inputDir}/*BIM.json")
    # KZ: changing BIM to AIM
    siteFiles = glob(f'{inputDir}/*AIM.json')

    GP_file = []
    Longitude = []
    Latitude = []
    id = []
    sites = []
    # site im dictionary
    periods = np.array(
        [
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.075,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.75,
            1,
            2,
            3,
            4,
            5,
            7.5,
            10,
        ]
    )
    dict_im_all = {
        ('type', 'loc', 'dir', 'stat'): [],
        ('PGA', 0, 1, 'median'): [],
        ('PGA', 0, 1, 'beta'): [],
        ('PGA', 0, 2, 'median'): [],
        ('PGA', 0, 2, 'beta'): [],
        ('PGV', 0, 1, 'median'): [],
        ('PGV', 0, 1, 'beta'): [],
        ('PGV', 0, 2, 'median'): [],
        ('PGV', 0, 2, 'beta'): [],
        ('PGD', 0, 1, 'median'): [],
        ('PGD', 0, 1, 'beta'): [],
        ('PGD', 0, 2, 'median'): [],
        ('PGD', 0, 2, 'beta'): [],
    }
    dict_im_site = {
        '1-PGA-0-1': [],
        '1-PGA-0-2': [],
        '1-PGV-0-1': [],
        '1-PGV-0-2': [],
        '1-PGD-0-1': [],
        '1-PGD-0-2': [],
    }
    for Ti in periods:
        dict_im_all.update(
            {
                (f'SA({Ti}s)', 0, 1, 'median'): [],
                (f'SA({Ti}s)', 0, 1, 'beta'): [],
                (f'SA({Ti}s)', 0, 2, 'median'): [],
                (f'SA({Ti}s)', 0, 2, 'beta'): [],
            }
        )
        dict_im_site.update({f'1-SA({Ti}s)-0-1': [], f'1-SA({Ti}s)-0-2': []})

    for site in siteFiles:
        dict_im = {
            ('type', 'loc', 'dir', 'stat'): [],
            ('PGA', 0, 1, 'median'): [],
            ('PGA', 0, 1, 'beta'): [],
            ('PGA', 0, 2, 'median'): [],
            ('PGA', 0, 2, 'beta'): [],
            ('PGV', 0, 1, 'median'): [],
            ('PGV', 0, 1, 'beta'): [],
            ('PGV', 0, 2, 'median'): [],
            ('PGV', 0, 2, 'beta'): [],
            ('PGD', 0, 1, 'median'): [],
            ('PGD', 0, 1, 'beta'): [],
            ('PGD', 0, 2, 'median'): [],
            ('PGD', 0, 2, 'beta'): [],
        }
        dict_im_site = {
            '1-PGA-0-1': [],
            '1-PGA-0-2': [],
            '1-PGV-0-1': [],
            '1-PGV-0-2': [],
            '1-PGD-0-1': [],
            '1-PGD-0-2': [],
        }
        for Ti in periods:
            dict_im.update(
                {
                    (f'SA({Ti}s)', 0, 1, 'median'): [],
                    (f'SA({Ti}s)', 0, 1, 'beta'): [],
                    (f'SA({Ti}s)', 0, 2, 'median'): [],
                    (f'SA({Ti}s)', 0, 2, 'beta'): [],
                }
            )
            dict_im_site.update({f'1-SA({Ti}s)-0-1': [], f'1-SA({Ti}s)-0-2': []})

        with open(site) as f:
            All_json = json.load(f)
            generalInfo = All_json['GeneralInformation']
            Longitude.append(generalInfo['Longitude'])
            Latitude.append(generalInfo['Latitude'])
            # siteID = generalInfo['BIM_id']
            # KZ: changing BIM to AIM
            siteID = generalInfo['AIM_id']
            # get unit info (needed for determining the simulated acc unit)
            unitInfo = All_json['units']
            # get scaling factor for surface acceleration
            acc_unit = {'AccelerationEvent': 'g'}
            f_scale_units = get_scale_factors(acc_unit, unitInfo)

            # if f_scale_units is None
            if None in [acc_unit, f_scale_units]:
                f_scale = 1.0
            else:
                for cur_var in list(f_scale_units.keys()):
                    cur_unit = acc_unit.get(cur_var)
                    unit_type = None
                    for base_unit_type, unit_set in globals()['unit_types'].items():
                        if cur_unit in unit_set:
                            unit_type = base_unit_type
                    if unit_type == 'acceleration':
                        f_scale = f_scale_units.get(cur_var)

            id.append(int(siteID))

            siteFileName = f'Site_{siteID}.csv'
            sites.append(siteFileName)

            workdirs = glob(f'{inputDir}/{siteID}/workdir.*')
            siteEventFiles = []
            siteEventFactors = []

            # initialization
            psa_x = []
            psa_y = []
            pga_x = []
            pga_y = []
            pgv_x = []
            pgv_y = []
            pgd_x = []
            pgd_y = []

            for workdir in workdirs:
                head, sep, sampleID = workdir.partition('workdir.')
                # print(sampleID)

                eventName = f'Event_{siteID}_{sampleID}'
                # print(eventName)
                shutil.copy(f'{workdir}/fmkEVENT', f'{outputDir}/{eventName}.json')

                siteEventFiles.append(eventName)
                siteEventFactors.append(1.0)

                # compute ground motion intensity measures
                with open(f'{outputDir}/{eventName}.json') as f:
                    cur_gm = json.load(f)
                cur_seismograms = cur_gm['Events'][0]['timeSeries']
                num_seismograms = len(cur_seismograms)
                # im_X and im_Y
                for cur_time_series in cur_seismograms:
                    dt = cur_time_series.get('dT')
                    acc = [x / f_scale for x in cur_time_series.get('data')]
                    acc_hist = np.array([[dt * x for x in range(len(acc))], acc])
                    # get intensity measure
                    my_response_spectrum_calc = NewmarkBeta(
                        acc, dt, periods, damping=0.05, units='g'
                    )
                    tmp, time_series, accel, vel, disp = (
                        my_response_spectrum_calc.run()
                    )
                    psa = tmp.get('Pseudo-Acceleration')
                    pga = time_series.get('PGA', 0.0)
                    pgv = time_series.get('PGV', 0.0)
                    pgd = time_series.get('PGD', 0.0)
                    # append
                    if cur_time_series.get('name') == 'accel_X':
                        psa_x.append(psa)
                        pga_x.append(pga)
                        pgv_x.append(pgv)
                        pgd_x.append(pgd)
                    else:
                        psa_y.append(psa)
                        pga_y.append(pga)
                        pgv_y.append(pgv)
                        pgd_y.append(pgd)

            # individual
            dict_im_site['1-PGA-0-1'] = pga_x
            dict_im_site['1-PGA-0-2'] = pga_y
            dict_im_site['1-PGV-0-1'] = pgv_x
            dict_im_site['1-PGV-0-2'] = pgv_y
            dict_im_site['1-PGD-0-1'] = pgd_x
            dict_im_site['1-PGD-0-2'] = pgd_y
            for jj, Ti in enumerate(periods):
                cur_sa = f'1-SA({Ti}s)-0-1'
                dict_im_site[cur_sa] = [tmp[jj] for tmp in psa_x]
                cur_sa = f'1-SA({Ti}s)-0-2'
                dict_im_site[cur_sa] = [tmp[jj] for tmp in psa_y]

            # dump dict_im_site
            df_im_site = pd.DataFrame.from_dict(dict_im_site)
            site_im_file = f'{inputDir}/{siteID}/IM_realization.csv'
            df_im_site.to_csv(site_im_file, index=False)

            # median and dispersion
            # psa
            if len(psa_x) > 0:
                m_psa_x = np.exp(np.mean(np.log(psa_x), axis=0))
                s_psa_x = np.std(np.log(psa_x), axis=0)
            else:
                m_psa_x = np.zeros((len(periods),))
                s_psa_x = np.zeros((len(periods),))
            if len(psa_y) > 0:
                m_psa_y = np.exp(np.mean(np.log(psa_y), axis=0))
                s_psa_y = np.std(np.log(psa_y), axis=0)
            else:
                m_psa_y = np.zeros((len(periods),))
                s_psa_y = np.zeros((len(periods),))
            # pga
            if len(pga_x) > 0:
                m_pga_x = np.exp(np.mean(np.log(pga_x)))
                s_pga_x = np.std(np.log(pga_x))
            else:
                m_psa_x = 0.0
                s_pga_x = 0.0
            if len(pga_y) > 0:
                m_pga_y = np.exp(np.mean(np.log(pga_y)))
                s_pga_y = np.std(np.log(pga_y))
            else:
                m_psa_y = 0.0
                s_pga_y = 0.0
            # pgv
            if len(pgv_x) > 0:
                m_pgv_x = np.exp(np.mean(np.log(pgv_x)))
                s_pgv_x = np.std(np.log(pgv_x))
            else:
                m_pgv_x = 0.0
                s_pgv_x = 0.0
            if len(pgv_y) > 0:
                m_pgv_y = np.exp(np.mean(np.log(pgv_y)))
                s_pgv_y = np.std(np.log(pgv_y))
            else:
                m_pgv_y = 0.0
                s_pgv_y = 0.0
            # pgd
            if len(pgd_x) > 0:
                m_pgd_x = np.exp(np.mean(np.log(pgd_x)))
                s_pgd_x = np.std(np.log(pgd_x))
            else:
                m_pgd_x = 0.0
                s_pgd_x = 0.0
            if len(pgd_y) > 0:
                m_pgd_y = np.exp(np.mean(np.log(pgd_y)))
                s_pgd_y = np.std(np.log(pgd_y))
            else:
                m_pgd_y = 0.0
                s_pgd_y = 0.0
            # add to dictionary
            dict_im[('type', 'loc', 'dir', 'stat')].append(int(siteID))
            # pga
            dict_im[('PGA', 0, 1, 'median')].append(m_pga_x)
            dict_im[('PGA', 0, 1, 'beta')].append(s_pga_x)
            dict_im[('PGA', 0, 2, 'median')].append(m_pga_y)
            dict_im[('PGA', 0, 2, 'beta')].append(s_pga_y)
            # pgv
            dict_im[('PGV', 0, 1, 'median')].append(m_pgv_x)
            dict_im[('PGV', 0, 1, 'beta')].append(s_pgv_x)
            dict_im[('PGV', 0, 2, 'median')].append(m_pgv_y)
            dict_im[('PGV', 0, 2, 'beta')].append(s_pgv_y)
            # pgd
            dict_im[('PGD', 0, 1, 'median')].append(m_pgd_x)
            dict_im[('PGD', 0, 1, 'beta')].append(s_pgd_x)
            dict_im[('PGD', 0, 2, 'median')].append(m_pgd_y)
            dict_im[('PGD', 0, 2, 'beta')].append(s_pgd_y)
            for jj, Ti in enumerate(periods):
                cur_sa = f'SA({Ti}s)'
                dict_im[(cur_sa, 0, 1, 'median')].append(m_psa_x[jj])
                dict_im[(cur_sa, 0, 1, 'beta')].append(s_psa_x[jj])
                dict_im[(cur_sa, 0, 2, 'median')].append(m_psa_y[jj])
                dict_im[(cur_sa, 0, 2, 'beta')].append(s_psa_y[jj])

            # aggregate
            for cur_key, cur_value in dict_im.items():
                if isinstance(cur_value, list):
                    dict_im_all[cur_key].append(cur_value[0])
                else:
                    dict_im_all[cur_key].append(cur_value)

            # save median and standard deviation to IM.csv
            df_im = pd.DataFrame.from_dict(dict_im)
            df_im.to_csv(f'{inputDir}/{siteID}/IM.csv', index=False)

            # create site csv
            siteDF = pd.DataFrame(
                list(zip(siteEventFiles, siteEventFactors)),
                columns=['TH_file', 'factor'],
            )
            siteDF.to_csv(f'{outputDir}/{siteFileName}', index=False)

    # create the EventFile
    gridDF = pd.DataFrame(
        list(zip(sites, Longitude, Latitude)),
        columns=['GP_file', 'Longitude', 'Latitude'],
    )

    # change the writing mode to append for paralleling workflow
    if os.path.exists(f'{outputDir}/EventGrid.csv'):
        # EventGrid.csv has been created
        gridDF.to_csv(
            f'{outputDir}/EventGrid.csv', mode='a', index=False, header=False
        )
    else:
        # EventGrid.csv to be created
        gridDF.to_csv(f'{outputDir}/EventGrid.csv', index=False)
    # gridDF.to_csv(f"{outputDir}/EventGrid.csv", index=False)
    print(f'EventGrid.csv saved to {outputDir}')

    # create pandas
    im_csv_path = os.path.dirname(os.path.dirname(outputDir))
    df_im_all = pd.DataFrame.from_dict(dict_im_all)
    try:
        os.mkdir(os.path.join(im_csv_path, 'Results'))
    except:
        print('Results folder already exists')
    # KZ: 10/19/2022, minor patch for Buildings
    df_im_all.to_csv(
        os.path.join(
            im_csv_path,
            'Results',
            'Buildings',
            f'IM_{min(id)}-{max(id)}.csv',
        ),
        index=False,
    )
    df_im_all.to_csv(
        os.path.join(im_csv_path, f'IM_{min(id)}-{max(id)}.csv'),
        index=False,
    )

    # remove original files
    if removeInputDir:
        shutil.rmtree(inputDir)

    return 0


if __name__ == '__main__':
    # Defining the command line arguments

    workflowArgParser = argparse.ArgumentParser(
        'Create ground motions for BIM.', allow_abbrev=False
    )

    workflowArgParser.add_argument(
        '-i', '--inputDir', help='Dir containing results of siteResponseWhale.'
    )

    workflowArgParser.add_argument(
        '-o', '--outputDir', help='Dir where results to be stored.'
    )

    workflowArgParser.add_argument('--removeInput', action='store_true')

    # Parsing the command line arguments
    wfArgs = workflowArgParser.parse_args()

    print(wfArgs)
    # Calling the main function
    createFilesForEventGrid(wfArgs.inputDir, wfArgs.outputDir, wfArgs.removeInput)
