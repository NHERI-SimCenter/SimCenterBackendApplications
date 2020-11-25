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
# Adam Zsarnóczay
# Joanna J. Zou
# Long Chen

import os, sys
import argparse, json
import importlib
import subprocess
import shutil
from scipy import integrate
import numpy as np
from math import pi
import matplotlib.pyplot as plt


convert_EDP = {
    'max_abs_acceleration' : 'PGA'
}

def write_RV():

    pass

    # TODO: check simulation data exists and contains all important fields
    # TODO: get simulation data & write to SIM file

def run_opensees(EVENT_input_path, SAM_input_path, BIM_input_path,
                   EDP_input_path):

    sys.path.insert(0, os.getcwd())

    # load the model builder script
    with open(BIM_input_path, 'r') as f:
        BIM_in = json.load(f)

    model_params = BIM_in['GI']
    
    with open(SAM_input_path, 'r') as f:
        SAM_in = json.load(f)

    sys.path.insert(0, SAM_in['modelPath'])

    model_script_path = SAM_in['mainScript']

    # load the event file
    with open(EVENT_input_path, 'r') as f:
        EVENT_in = json.load(f)['Events'][0]

    event_list = EVENT_in['timeSeries']
    pattern_list = EVENT_in['pattern']

    fileNames = ['xInput', 'yInput']
    # define the time series
    for evt_i, event in enumerate(event_list):

        acc = event['data']
        vel = integrate.cumtrapz(acc, dx=event['dT']) * 9.81
        vel = np.insert(vel, 0, 0.0)
        disp = integrate.cumtrapz(vel, dx=event['dT'])
        disp = np.insert(disp, 0, 0.0)
        time = np.arange(0, event['dT'] * len(acc), event['dT'])
        np.savetxt(fileNames[evt_i] + '.acc', acc)
        np.savetxt(fileNames[evt_i] + '.vel', vel)
        np.savetxt(fileNames[evt_i] + '.disp', disp)
        np.savetxt(fileNames[evt_i] + '.time', time)


    # create the EDP specification
    # load the EDP file
    with open(EDP_input_path, 'r') as f:
        EDP_in = json.load(f)

    EDP_list = EDP_in['EngineeringDemandParameters'][0]['responses']

    edp_specs = {}
    for response in EDP_list:
        if response['type'] not in edp_specs.keys():
            edp_specs.update({response['type']: {}})
        edp_specs[response['type']].update(
            {response['id']: dict([(dof, list(np.atleast_1d(response['node'])))
                              for dof in response['dofs']])})

    #for edp_name, edp_data in edp_specs.items():
    #    print(edp_name, edp_data)

    # run the analysis
    shutil.copyfile(os.path.join(SAM_in['modelPath'], model_script_path), os.path.join(os.getcwd(), model_script_path))
    build_model(model_params, evt_i + 1)
    #print('OpenSees ' + model_script_path + ' {:.2f} {:.1f}'.format(model_params['Vs30'], model_params['Depth_to_Rock']))
    #subprocess.Popen('OpenSees ' + model_script_path + ' {:.2f} {:.1f}'.format(model_params['Vs30'], model_params['Depth_to_Rock']), shell=True).wait()
    subprocess.Popen('OpenSees ' + model_script_path, shell=True).wait()

    acc = np.loadtxt('accelerationElasAct.out')
    acc_surf_x = acc[:,-3] / 9.81
    acc_surf_z = acc[:,-1] / 9.81

    
    #for edp_name, edp_data in edp_specs.items():
    #    print(edp_name, edp_data)

    # TODO: default analysis script

    # save the EDP results

    #print(EDP_res)

    for response in EDP_list:
        edp = [max(abs(acc_surf_x)), max(abs(acc_surf_z))]

        response['scalar_data'] = edp # [val for dof, val in edp.items()]


    with open(EDP_input_path, 'w') as f:
        json.dump(EDP_in, f, indent=2)


def build_model(model_params, numEvt):
    elementSize = 0.5
    g = 9.81
    VsRock = 760 #site class B
    try:
        depthToRock = model_params['DepthToRock']
    except:
        depthToRock = 0
    Vs30 = model_params['Vs30']

    # Vs30 model
    thickness, Vs = SVM(Vs30, depthToRock, VsRock, elementSize)

    numElems = len(Vs)
    # Config model
    f = open('freefield_config.tcl', 'w')
    f.write('# site response configuration file\n')
    f.write('set soilThick {:.1f}\n'.format(thickness))
    f.write('set numLayers {:d}\n'.format(numElems))
    f.write('# layer thickness - bottom to top\n')
    eleVsize = thickness/numElems
    travelTime = 0
    for ii in range(numElems):
        f.write('set layerThick({:d}) {:.2f}\n'.format(ii+1, eleVsize))
        f.write('set nElemY({:d}) 1\n'.format(ii+1))
        f.write('set sElemY({:d}) {:.3f}\n'.format(ii+1, eleVsize))
        travelTime += eleVsize / Vs[ii]
    
    averageVs = thickness / travelTime # time averaged shear wave velocity
    naturalFrequency = averageVs / 4 / thickness  # Vs/4H

    f.write('set nElemT {:d}\n'.format(numElems))
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

    f.write('set rockVs {:.1f}\n'.format(VsRock))
    f.write('set omega1 {:.2f}\n'.format(2.0 * pi * naturalFrequency))
    f.write('set omega2 {:.2f}\n'.format(2.0 * pi * naturalFrequency * 5.0))
    f.close()

    # Create Material
    f = open('freefield_material.tcl', 'w')

    if model_params['Model'] in 'BA':
        # Borja and Amies 1994 J2 model
        rhoSoil = model_params['Den']
        poisson = 0.3
        sig_v = rhoSoil * g * eleVsize * 0.5
        for ii in range(numElems):
            f.write('set rho({:d}) {:.1f}\n'.format(ii+1, rhoSoil))
            shearG = rhoSoil * Vs[ii] * Vs[ii]
            bulkK = shearG * 2.0 * (1 + poisson) / 3.0 / (1.0 - 2.0 * poisson)
            f.write('set shearG({:d}) {:.2f}\n'.format(ii+1, shearG))
            f.write('set bulkK({:d}) {:.2f}\n'.format(ii+1, bulkK))
            f.write('set su({:d}) {:.2f}\n'.format(ii+1, model_params['Su_rat'] * sig_v))
            sig_v = sig_v + rhoSoil * g * eleVsize
            f.write('set h({:d}) {:.2f}\n'.format(ii+1, shearG * model_params['h/G']))
            f.write('set m({:d}) {:.2f}\n'.format(ii+1, model_params['m']))
            f.write('set h0({:d}) {:.2f}\n'.format(ii+1, model_params['h0']))
            f.write('set chi({:d}) {:.2f}\n'.format(ii+1, model_params['chi']))          
            f.write('set mat({:d}) "J2CyclicBoundingSurface {:d} $shearG({:d}) $bulkK({:d}) $su({:d}) $rho({:d}) $h({:d}) $m({:d}) $h0({:d}) $chi({:d}) 0.5"\n\n\n'.format(ii+1, ii+1, ii+1, ii+1, ii+1, ii+1, ii+1, ii+1, ii+1, ii+1))
    else:
        rhoSoil = model_params['Den']
        poisson = 0.3
        for ii in range(numElems):
            f.write('set rho({:d}) {:.1f}\n'.format(ii+1, rhoSoil))
            f.write('set shearG({:d}) {:.2f}\n'.format(ii+1, rhoSoil * Vs[ii] * Vs[ii]))
            f.write('set nu({:d}) {:.2f}\n'.format(ii+1, poisson))
            f.write('set E({:d}) {:.2f}\n\n'.format(ii+1, 2 * rhoSoil * Vs[ii] * Vs[ii] * (1 + poisson)))
            f.write('set mat({:d}) "ElasticIsotropic {:d} $E({:d}) $nu({:d}) $rho({:d})"\n\n\n'.format(ii+1, ii+1, ii+1, ii+1, ii+1))


    f.close()

def SVM(Vs30, depthToRock, VsRock, elementSize):
    # Sediment Velocity Model (SVM) 
    # Developed by Jian Shi and Domniki Asimaki (2018)
    # Generates a shear velocity profile from Vs30 for shallow crust profiles
    # Valid for 173.1 m/s < Vs30 < 1000 m/s

    # Check Vs30
    if Vs30 < 173.1 or Vs30 > 1000:
        print('Caution: Vs30 is not within the valid range of the SVM! \n') 

    # Parameters specific to: California
    z_star = 2.5;    # [m] depth considered to have constant Vs
    p1 = -2.1688e-4
    p2 =  0.5182
    p3 =  69.452
    r1 = -59.67
    r2 = -0.2722
    r3 =  11.132
    s1 =  4.110
    s2 = -1.0521e-4
    s3 = -10.827
    s4 = -7.6187e-3

    #  SVM Parameters f(Vs30)
    Vs0 = p1 * (Vs30 ** 2) + p2 * Vs30 + p3
    k = np.exp(r1 * (Vs30 ** r2) + r3)
    n = s1 * np.exp(s2 * Vs30) + s3 * np.exp(s4 * Vs30)

    # Check element size for max. frequency
    maxFrequency = 50   #Hz
    waveLength = Vs0 / maxFrequency
    # Need four elements per wavelength
    if 4.0 * elementSize <= waveLength:
        step_size = elementSize
    else:
        step_size = waveLength / 4.0

    depth = max(30.0, depthToRock)
    z = np.linspace(0.0 + 0.5 * step_size, depth - 0.5 * step_size, int(depth / step_size)) # discretize depth to bedrock

    # Vs Profile
    Vs = np.zeros(len(z))
    Vs[0] = Vs0
    for ii in range(1, len(z)):
        if z[ii] <= z_star:
            Vs[ii] = Vs0
        else:
            Vs[ii] = Vs0 * (1 + k * (z[ii] - z_star)) ** (1.0 / n)

    if depthToRock > 0:
        thickness = depthToRock
        Vs_cropped = Vs[np.where(z <= depthToRock)]
    else:
        Vs_cropped = Vs[np.where(Vs <= VsRock)]
        thickness = z[len(Vs_cropped) - 1] + 0.5 * step_size
    
    fig = plt.figure()
    plt.plot(Vs, z, label='Vs profile')
    plt.plot(Vs_cropped, z[0 : len(Vs_cropped)], label='Vs profile to bedrock')
    plt.grid(True)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.legend()
    plt.text(100, 12.5, 'Vs30 = {:.1f}m/s'.format(Vs30))
    plt.text(100, 17.5, 'Depth to bedrock = {:.1f}m'.format(depthToRock))
    ax.set_xlabel('Vs (m/s)')
    ax.set_ylabel('Depth (m)')
    ax.set_xlim(left = 0)
    ax.set_title('Sediment Velocity Model (SVM)')
    fig.savefig('Vs.png')

    return thickness, Vs_cropped

if __name__ == '__main__':

    SVM(380, 0, 360, 0.5)
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameBIM', default=None)
    parser.add_argument('--filenameSAM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameEDP', default=None)
    parser.add_argument('--filenameSIM', default=None)
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.getRV:
        sys.exit(write_RV())
    else:
        sys.exit(run_opensees(
            args.filenameEVENT, args.filenameSAM, args.filenameBIM,
            args.filenameEDP))
