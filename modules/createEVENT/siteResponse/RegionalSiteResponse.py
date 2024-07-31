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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS'
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
# Long Chen


import argparse
import json
import os
import posixpath
import shutil
import subprocess
import sys
from math import pi

# import the common constants and methods
from pathlib import Path

import numpy as np
from scipy import integrate

this_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()  # noqa: PTH100, PTH120
main_dir = this_dir.parents[1]
sys.path.insert(0, str(main_dir / 'common'))
from simcenter_common import *  # noqa: E402, F403

convert_EDP = {'max_abs_acceleration': 'PGA'}  # noqa: N816

gravityG = 9.81  # m/s2  # noqa: N816
# default element size before wave length check
elementSize = 0.5  # m  # noqa: N816
# site class B, m/s
VsRock = 760
plotFlag = False  # noqa: N816


def get_scale_factors(input_units, output_units):  # noqa: ANN001, ANN201, C901, PLR0912
    """Determine the scale factor to convert input event to internal event data"""  # noqa: D400, D415
    # special case: if the input unit is not specified then do not do any scaling
    if input_units is None:
        scale_factors = {'ALL': 1.0}

    else:
        # parse output units:

        # if no length unit is specified, 'inch' is assumed
        unit_length = output_units.get('length', 'inch')
        f_length = globals().get(unit_length, None)
        if f_length is None:
            raise ValueError(f'Specified length unit not recognized: {unit_length}')  # noqa: EM102, TRY003

        # if no time unit is specified, 'sec' is assumed
        unit_time = output_units.get('time', 'sec')
        f_time = globals().get(unit_time, None)
        if f_time is None:
            raise ValueError(f'Specified time unit not recognized: {unit_time}')  # noqa: EM102, TRY003

        scale_factors = {}

        for input_name, input_unit in input_units.items():
            # exceptions
            if input_name in [
                'factor',
            ]:
                f_scale = 1.0

            else:
                # get the scale factor to standard units
                f_in = globals().get(input_unit, None)
                if f_in is None:
                    raise ValueError(  # noqa: TRY003
                        f'Input unit for event files not recognized: {input_unit}'  # noqa: EM102
                    )

                unit_type = None
                for base_unit_type, unit_set in globals()['unit_types'].items():
                    if input_unit in unit_set:
                        unit_type = base_unit_type

                if unit_type is None:
                    raise ValueError(f'Failed to identify unit type: {input_unit}')  # noqa: EM102, TRY003

                # the output unit depends on the unit type
                if unit_type == 'acceleration':
                    f_out = f_time**2.0 / f_length

                elif unit_type == 'speed':
                    f_out = f_time / f_length

                elif unit_type == 'length':
                    f_out = 1.0 / f_length

                else:
                    raise ValueError(  # noqa: TRY003
                        f'Unexpected unit type in workflow: {unit_type}'  # noqa: EM102
                    )

                # the scale factor is the product of input and output scaling
                f_scale = f_in * f_out

            scale_factors.update({input_name: f_scale})

    return scale_factors


def postProcess(evtName, input_units, f_scale_units):  # noqa: ANN001, ANN201, N802, N803, D103
    # if f_scale_units is None
    if None in [input_units, f_scale_units]:
        f_scale = 1.0
    else:
        for cur_var in list(f_scale_units.keys()):
            cur_unit = input_units.get(cur_var)
            unit_type = None
            for base_unit_type, unit_set in globals()['unit_types'].items():
                if cur_unit in unit_set:
                    unit_type = base_unit_type
            if unit_type == 'acceleration':
                f_scale = f_scale_units.get(cur_var)

    acc = np.loadtxt('acceleration.out')
    # os.remove("acceleration.out")  # remove acceleration file to save space  # noqa: ERA001
    # acc = np.loadtxt("out_tcl/acceleration.out")  # noqa: ERA001
    # shutil.rmtree("out_tcl")  # remove output files to save space  # noqa: ERA001
    # KZ, 01/17/2022: I corrected the acc_surf from [:,-2] to [:,-3] (horizontal x direction)  # noqa: E501
    time = acc[:, 0]
    # acc_surf = acc[:,-2] / 9.81  # noqa: ERA001
    # KZ, 03/07/2022: removed the unit conversion here (did in createGM4BIM)
    acc_surf = acc[:, -3]
    dT = time[1] - time[0]  # noqa: N806

    timeSeries = dict(  # noqa: C408, N806
        name='accel_X',
        type='Value',
        dT=dT,
        data=[x * f_scale for x in acc_surf.tolist()],
    )

    patterns = dict(type='UniformAcceleration', timeSeries='accel_X', dof=1)  # noqa: C408

    # KZ, 01/17/2022: I added global y direction
    # KZ, 03/07/2022: removed the unit conversion here (did in createGM4BIM)
    acc_surf_y = acc[:, -1]
    timeSeries_y = dict(  # noqa: C408, N806
        name='accel_Y',
        type='Value',
        dT=dT,
        data=[y * f_scale for y in acc_surf_y.tolist()],
    )

    patterns_y = dict(type='UniformAcceleration', timeSeries='accel_Y', dof=2)  # noqa: C408

    # KZ, 01/17/2022: I updated this section accordingly
    """
    evts = dict(
        RandomVariables = [],
        name = "SiteResponseTool",
        type = "Seismic",
        description = "Surface acceleration",
        dT = dT,
        numSteps = len(acc_surf),
        timeSeries = [timeSeries],
        pattern = [patterns]
    )
    """
    evts = dict(  # noqa: C408
        RandomVariables=[],
        name='SiteResponseTool',
        type='Seismic',
        description='Surface acceleration',
        dT=dT,
        numSteps=len(acc_surf),
        timeSeries=[timeSeries, timeSeries_y],
        pattern=[patterns, patterns_y],
    )

    dataToWrite = dict(Events=[evts])  # noqa: C408, N806

    with open(evtName, 'w') as outfile:  # noqa: PTH123
        json.dump(dataToWrite, outfile, indent=4)

    print('DONE postProcess')  # noqa: T201

    return 0


def run_opensees(  # noqa: ANN201, D103, PLR0913
    BIM_file, EVENT_file, event_path, model_script, model_script_path, ndm, getRV  # noqa: ANN001, N803
):
    sys.path.insert(0, os.getcwd())  # noqa: PTH109

    print('**************** run_opensees ****************')  # noqa: T201
    # load the model builder script
    with open(BIM_file) as f:  # noqa: PTH123
        BIM_in = json.load(f)  # noqa: N806

    model_params = BIM_in['GeneralInformation']
    model_units = BIM_in['GeneralInformation']['units']  # noqa: F841
    location = BIM_in['GeneralInformation']['location']  # noqa: F841

    # convert units if necessary
    # KZ, 01/17/2022: Vs30 and DepthToRock are not subjected to the model_units for now...  # noqa: E501
    """
    if model_units['length'] in ['inch', 'in']:
        model_params['Vs30'] = model_params['Vs30'] * 0.0254
        model_params['DepthToRock'] = model_params['DepthToRock'] * 0.3048
    elif model_units['length'] in ['foot', 'ft', 'feet']:
        model_params['Vs30'] = model_params['Vs30'] * 0.0254
        model_params['DepthToRock'] = model_params['DepthToRock'] * 0.3048
    """

    sys.path.insert(0, model_script_path)

    # Create input motion from SimCenterEvent
    if getRV:
        write_RV(BIM_file, EVENT_file, event_path)
    else:
        get_records(BIM_file, EVENT_file, event_path)
        # load the event file
        with open(EVENT_file) as f:  # noqa: PTH123
            EVENT_in_All = json.load(f)  # noqa: N806
            EVENT_in = EVENT_in_All['Events'][0]  # noqa: N806

        event_list = EVENT_in['timeSeries']
        pattern_list = EVENT_in['pattern']  # noqa: F841

        fileNames = ['xInput', 'yInput']  # noqa: N806
        # define the time series
        for evt_i, event in enumerate(event_list):
            acc = event['data']
            vel = integrate.cumtrapz(acc, dx=event['dT']) * gravityG
            vel = np.insert(vel, 0, 0.0)
            disp = integrate.cumtrapz(vel, dx=event['dT'])
            disp = np.insert(disp, 0, 0.0)
            time = np.arange(0, event['dT'] * len(acc), event['dT'])
            np.savetxt(fileNames[evt_i] + '.acc', acc)
            np.savetxt(fileNames[evt_i] + '.vel', vel)
            np.savetxt(fileNames[evt_i] + '.disp', disp)
            np.savetxt(fileNames[evt_i] + '.time', time)

        # run the analysis
        shutil.copyfile(
            os.path.join(model_script_path, model_script),  # noqa: PTH118
            os.path.join(os.getcwd(), model_script),  # noqa: PTH109, PTH118
        )

        build_model(model_params, int(ndm) - 1)

        subprocess.Popen('OpenSees ' + model_script, shell=True).wait()  # noqa: S602

        # FMK
        # update Event file with acceleration recorded at surface
        # acc = np.loadtxt('accelerationElasAct.out')  # noqa: ERA001
        # acc_surf_x = acc[:, -3] / gravityG  # noqa: ERA001
        # EVENT_in_All['Events'][0]['timeSeries'][0]['data'] = acc_surf_x.tolist()  # noqa: ERA001
        # if int(ndm) == 3:
        #    acc_surf_z = acc[:, -1] / gravityG  # noqa: ERA001
        #    EVENT_in_All['Events'][0]['timeSeries'][1]['data'] = acc_surf_z.tolist()  # noqa: ERA001

        # EVENT_in_All['Events'][0]['location'] = location  # noqa: ERA001

        # EVENT_file2 = 'EVENT2.json' for debug
        # with open(EVENT_file, 'w') as f:
        #    json.dump(EVENT_in_All, f, indent=2)  # noqa: ERA001

        # KZ, 01/17/2022: get unit scaling factor (this is a temporary patch as we assume to have "g"  # noqa: E501
        # as the output acceleration from the site response analysis- need to discuss with Frank/Pedro/Steve  # noqa: E501
        # about this...)
        # scale the input data to the event unit used internally
        input_units = {'AccelerationEvent': 'g'}
        output_units = BIM_in.get('units', None)
        f_scale_units = get_scale_factors(input_units, output_units)

        postProcess('fmkEVENT', input_units, f_scale_units)


def get_records(BIM_file, EVENT_file, data_dir):  # noqa: ANN001, ANN201, N803, D103
    with open(BIM_file) as f:  # noqa: PTH123
        bim_file = json.load(f)

    with open(EVENT_file) as f:  # noqa: PTH123
        event_file = json.load(f)

    event_id = event_file['Events'][0]['event_id']

    # get the scale factor if a user specified it (KZ, 01/17/2022)
    try:
        event_data = np.array(bim_file['Events']['Events']).T
        event_loc = np.where(event_data == event_id)[0][1]
        f_scale_user = float(event_data.T[event_loc][1])
    except:  # noqa: E722
        f_scale_user = 1.0

    # FMK scale_factor = dict([(evt['fileName'], evt.get('factor',1.0)) for evt in bim_file["Events"]["Events"]])[event_id]  # noqa: E501
    # KZ: multiply the scale_factor by f_scale_user
    scale_factor = 1.0 * f_scale_user

    event_file['Events'][0].update(load_record(event_id, data_dir, scale_factor))

    with open(EVENT_file, 'w') as f:  # noqa: PTH123
        json.dump(event_file, f, indent=2)


def write_RV(BIM_file, EVENT_file, data_dir):  # noqa: ANN001, ANN201, N802, N803, D103
    # Copied from SimCenterEvent, write name of motions

    with open(BIM_file) as f:  # noqa: PTH123
        bim_data = json.load(f)

    event_file = {'randomVariables': [], 'Events': []}

    # events = bim_data['Events']['Events']  # noqa: ERA001
    events = bim_data['Events'][0]['Events']

    if len(events) > 1:
        event_file['randomVariables'].append(
            {
                'distribution': 'discrete_design_set_string',
                'name': 'eventID',
                'value': 'RV.eventID',
                'elements': [],
            }
        )
        event_file['Events'].append(
            {
                #'type': 'Seismic',  # noqa: ERA001
                #'type': bim_data['Events']['type'],  # noqa: ERA001
                'type': bim_data['Events'][0]['type'],
                'event_id': 'RV.eventID',
                'unitScaleFactor': 1.0,
                'data_dir': data_dir,
            }
        )

        RV_elements = np.array(events).T[0].tolist()  # noqa: N806
        # RV_elements = []  # noqa: ERA001
        # for event in events:
        #    if event['EventClassification'] == 'Earthquake':
        #        RV_elements.append(event['fileName'])  # noqa: ERA001
        #    elif event['EventClassification'] == 'Hurricane':  # noqa: ERA001
        #        RV_elements.append(event['fileName'])  # noqa: ERA001
        #    elif event['EventClassification'] == 'Flood':  # noqa: ERA001
        #        RV_elements.append(event['fileName'])  # noqa: ERA001

        event_file['randomVariables'][0]['elements'] = RV_elements
    else:
        event_file['Events'].append(
            {
                #'type': bim_data['Events']['type'],  # noqa: ERA001
                'type': bim_data['Events'][0]['type'],
                'event_id': events[0]['fileName'],
                'unitScaleFactor': 1.0,
                'data_dir': str(data_dir),
            }
        )

    # if time histories are used, then load the first event
    # if bim_data['Events']['type'] == 'timeHistory':
    if bim_data['Events'][0]['type'] == 'timeHistory':
        event_file['Events'][0].update(
            load_record(events[0][0], data_dir, empty=len(events) > 1)
        )

    with open(EVENT_file, 'w') as f:  # noqa: PTH123
        json.dump(event_file, f, indent=2)


def load_record(fileName, data_dir, scale_factor=1.0, empty=False):  # noqa: ANN001, ANN201, FBT002, N803, D103
    # Copied from SimCenterEvent, write data of motions into Event

    fileName = fileName.split('x')[0]  # noqa: N806

    with open(posixpath.join(data_dir, f'{fileName}.json')) as f:  # noqa: PTH123
        event_data = json.load(f)

    event_dic = {
        'name': fileName,
        'dT': event_data['dT'],
        'numSteps': len(event_data['data_x']),
        'timeSeries': [],
        'pattern': [],
    }

    if not empty:
        for i, (src_label, tar_label) in enumerate(
            zip(['data_x', 'data_y'], ['accel_X', 'accel_Y'])
        ):
            if src_label in event_data.keys():  # noqa: SIM118
                event_dic['timeSeries'].append(
                    {
                        'name': tar_label,
                        'type': 'Value',
                        'dT': event_data['dT'],
                        'data': list(np.array(event_data[src_label]) * scale_factor),
                    }
                )
                event_dic['pattern'].append(
                    {
                        'type': 'UniformAcceleration',
                        'timeSeries': tar_label,
                        'dof': i + 1,
                    }
                )

    return event_dic


def build_model(model_params, numEvt):  # noqa: ANN001, ANN201, N803, D103, PLR0915
    try:
        depthToRock = model_params['DepthToRock']  # noqa: N806
    except:  # noqa: E722
        depthToRock = 0  # noqa: N806
    Vs30 = model_params['Vs30']  # noqa: N806

    # Vs30 model
    thickness, Vs = SVM(Vs30, depthToRock, VsRock, elementSize)  # noqa: N806

    numElems = len(Vs)  # noqa: N806
    # Config model
    f = open('freefield_config.tcl', 'w')  # noqa: SIM115, PTH123
    f.write('# site response configuration file\n')
    f.write(f'set soilThick {thickness:.1f}\n')
    f.write(f'set numLayers {numElems:d}\n')
    f.write('# layer thickness - bottom to top\n')
    eleVsize = thickness / numElems  # noqa: N806
    travelTime = 0  # noqa: N806
    for ii in range(numElems):
        f.write(f'set layerThick({ii + 1:d}) {eleVsize:.2f}\n')
        f.write(f'set nElemY({ii + 1:d}) 1\n')
        f.write(f'set sElemY({ii + 1:d}) {eleVsize:.3f}\n')
        travelTime += eleVsize / Vs[ii]  # noqa: N806

    averageVs = thickness / travelTime  # time averaged shear wave velocity  # noqa: N806
    naturalFrequency = averageVs / 4 / thickness  # Vs/4H  # noqa: N806

    f.write(f'set nElemT {numElems:d}\n')
    f.write('# motion file (used if the input arguments do not include motion)\n')
    f.write('set accFile  xInput.acc\n')
    f.write('set dispFile xInput.disp\n')
    f.write('set velFile  xInput.vel\n')
    f.write('set timeFile xInput.time\n')

    if numEvt > 1:
        f.write('set numEvt 2\n')
        f.write('set accFile2  yInput.acc\n')
        f.write('set dispFile2 yInput.disp\n')
        f.write('set velFile2  yInput.vel\n')
    else:
        f.write('set numEvt 1\n')

    f.write(f'set rockVs {VsRock:.1f}\n')
    f.write(f'set omega1 {2.0 * pi * naturalFrequency:.2f}\n')
    f.write(f'set omega2 {2.0 * pi * naturalFrequency * 5.0:.2f}\n')
    f.close()

    # Create Material
    f = open('freefield_material.tcl', 'w')  # noqa: SIM115, PTH123

    if model_params['Model'] in 'BA':
        # Borja and Amies 1994 J2 model
        rhoSoil = model_params['Den']  # noqa: N806
        poisson = 0.3
        sig_v = rhoSoil * gravityG * eleVsize * 0.5
        for ii in range(numElems):
            f.write(f'set rho({ii + 1:d}) {rhoSoil:.1f}\n')
            shearG = rhoSoil * Vs[ii] * Vs[ii]  # noqa: N806
            bulkK = shearG * 2.0 * (1 + poisson) / 3.0 / (1.0 - 2.0 * poisson)  # noqa: N806
            f.write(f'set shearG({ii + 1:d}) {shearG:.2f}\n')
            f.write(f'set bulkK({ii + 1:d}) {bulkK:.2f}\n')
            f.write(
                'set su({:d}) {:.2f}\n'.format(
                    ii + 1, model_params['Su_rat'] * sig_v
                )
            )
            sig_v = sig_v + rhoSoil * gravityG * eleVsize
            f.write(
                'set h({:d}) {:.2f}\n'.format(ii + 1, shearG * model_params['h/G'])
            )
            f.write('set m({:d}) {:.2f}\n'.format(ii + 1, model_params['m']))
            f.write('set h0({:d}) {:.2f}\n'.format(ii + 1, model_params['h0']))
            f.write('set chi({:d}) {:.2f}\n'.format(ii + 1, model_params['chi']))
            f.write(
                f'set mat({ii + 1:d}) "J2CyclicBoundingSurface {ii + 1:d} $shearG({ii + 1:d}) $bulkK({ii + 1:d}) $su({ii + 1:d}) $rho({ii + 1:d}) $h({ii + 1:d}) $m({ii + 1:d}) $h0({ii + 1:d}) $chi({ii + 1:d}) 0.5"\n\n\n'  # noqa: E501
            )
    elif model_params['Model'] in 'PIMY':
        # PIMY model
        rhoSoil = model_params['Den']  # noqa: N806
        poisson = 0.3
        sig_v = rhoSoil * gravityG * eleVsize * 0.5
        for ii in range(numElems):
            f.write(f'set rho({numElems - ii:d}) {rhoSoil:.1f}\n')
            shearG = rhoSoil * Vs[ii] * Vs[ii]  # noqa: N806
            bulkK = shearG * 2.0 * (1 + poisson) / 3.0 / (1.0 - 2.0 * poisson)  # noqa: N806
            f.write(f'set Vs({numElems - ii:d}) {Vs[ii]:.2f}\n')
            f.write(f'set shearG({numElems - ii:d}) {shearG:.2f}\n')
            f.write(f'set bulkK({numElems - ii:d}) {bulkK:.2f}\n')
            f.write(
                'set su({:d}) {:.2f}\n'.format(
                    numElems - ii, model_params['Su_rat'] * sig_v
                )
            )
            sig_v = sig_v + rhoSoil * gravityG * eleVsize
            f.write(
                'set h({:d}) {:.2f}\n'.format(
                    numElems - ii, shearG * model_params['h/G']
                )
            )
            f.write('set m({:d}) {:.2f}\n'.format(numElems - ii, model_params['m']))
            f.write(
                'set h0({:d}) {:.2f}\n'.format(numElems - ii, model_params['h0'])
            )
            f.write(
                'set chi({:d}) {:.2f}\n'.format(numElems - ii, model_params['chi'])
            )
            f.write(
                f'set mat({numElems - ii:d}) "PressureIndependMultiYield {numElems - ii:d} 3 $rho({numElems - ii:d}) $shearG({numElems - ii:d}) $bulkK({numElems - ii:d}) $su({numElems - ii:d}) 0.1 0.0 2116.0 0.0 31"\n\n\n'  # noqa: E501
            )
    else:
        rhoSoil = model_params['Den']  # noqa: N806
        poisson = 0.3
        for ii in range(numElems):
            f.write(f'set rho({ii + 1:d}) {rhoSoil:.1f}\n')
            f.write(f'set shearG({ii + 1:d}) {rhoSoil * Vs[ii] * Vs[ii]:.2f}\n')
            f.write(f'set nu({ii + 1:d}) {poisson:.2f}\n')
            f.write(
                f'set E({ii + 1:d}) {2 * rhoSoil * Vs[ii] * Vs[ii] * (1 + poisson):.2f}\n\n'  # noqa: E501
            )
            f.write(
                f'set mat({ii + 1:d}) "ElasticIsotropic {ii + 1:d} $E({ii + 1:d}) $nu({ii + 1:d}) $rho({ii + 1:d})"\n\n\n'  # noqa: E501
            )

    f.close()


def SVM(Vs30, depthToRock, VsRock, elementSize):  # noqa: ANN001, ANN201, N802, N803, D103, PLR0915
    # Sediment Velocity Model (SVM)
    # Developed by Jian Shi and Domniki Asimaki (2018)
    # Generates a shear velocity profile from Vs30 for shallow crust profiles
    # Valid for 173.1 m/s < Vs30 < 1000 m/s

    # Check Vs30
    if Vs30 < 173.1 or Vs30 > 1000:  # noqa: PLR2004
        print(f'Caution: Vs30 {Vs30} is not within the valid range of the SVM! \n')  # noqa: T201

    # Parameters specific to: California
    z_star = 2.5  # [m] depth considered to have constant Vs
    p1 = -2.1688e-4
    p2 = 0.5182
    p3 = 69.452
    r1 = -59.67
    r2 = -0.2722
    r3 = 11.132
    s1 = 4.110
    s2 = -1.0521e-4
    s3 = -10.827
    s4 = -7.6187e-3

    #  SVM Parameters f(Vs30)
    Vs0 = p1 * (Vs30**2) + p2 * Vs30 + p3  # noqa: N806
    k = np.exp(r1 * (Vs30**r2) + r3)
    n = s1 * np.exp(s2 * Vs30) + s3 * np.exp(s4 * Vs30)

    # Check element size for max. frequency
    maxFrequency = 50  # Hz  # noqa: N806
    waveLength = Vs0 / maxFrequency  # noqa: N806
    # Need four elements per wavelength
    if 4.0 * elementSize <= waveLength:  # noqa: SIM108
        step_size = elementSize
    else:
        step_size = waveLength / 4.0

    depth = max(30.0, depthToRock)
    z = np.linspace(
        0.0 + 0.5 * step_size, depth - 0.5 * step_size, int(depth / step_size)
    )  # discretize depth to bedrock

    # Vs Profile
    Vs = np.zeros(len(z))  # noqa: N806
    Vs[0] = Vs0
    for ii in range(1, len(z)):
        if z[ii] <= z_star:
            Vs[ii] = Vs0
        else:
            Vs[ii] = Vs0 * (1 + k * (z[ii] - z_star)) ** (1.0 / n)

    if depthToRock > 0:
        thickness = depthToRock
        Vs_cropped = Vs[np.where(z <= depthToRock)]  # noqa: N806
    else:
        Vs_cropped = Vs[np.where(Vs <= VsRock)]  # noqa: N806
        thickness = z[len(Vs_cropped) - 1] + 0.5 * step_size

    if plotFlag:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.plot(Vs, z, label='Vs profile')
        plt.plot(Vs_cropped, z[0 : len(Vs_cropped)], label='Vs profile to bedrock')
        plt.grid(True)  # noqa: FBT003
        ax = plt.gca()
        ax.invert_yaxis()
        plt.legend()
        plt.text(100, 12.5, f'Vs30 = {Vs30:.1f}m/s')
        plt.text(100, 17.5, f'Depth to bedrock = {depthToRock:.1f}m')
        ax.set_xlabel('Vs (m/s)')
        ax.set_ylabel('Depth (m)')
        ax.set_xlim(left=0)
        ax.set_title('Sediment Velocity Model (SVM)')
        fig.savefig('Vs.png')

    return thickness, Vs_cropped


if __name__ == '__main__':
    # SVM(380, 0, 360, 0.5)  # noqa: ERA001
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameAIM', default=None)
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--pathEventData')
    parser.add_argument('--mainScript')
    parser.add_argument('--modelPath')
    parser.add_argument('--ndm')
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    sys.exit(
        run_opensees(
            args.filenameAIM,
            args.filenameEVENT,
            args.pathEventData,
            args.mainScript,
            args.modelPath,
            args.ndm,
            args.getRV,
        )
    )
