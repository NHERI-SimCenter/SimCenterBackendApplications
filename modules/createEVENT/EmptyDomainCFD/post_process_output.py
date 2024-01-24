# -*- coding: utf-8 -*-
# Copyright (c) 2016-2017, The Regents of the University of California (Regents).
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the FreeBSD Project.
#
# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
# THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS
# PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
# UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

#
# Contributors:
# Abiy Melaku


#
# This script reads OpenFOAM output and plot the characteristics of the 
# approaching wind. For now, it read and plots only velocity field data.  
#

import sys
import os
import subprocess
import json
import stat
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt   
import matplotlib.gridspec as gridspec   
from scipy import signal
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy import stats
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read_velocity_data(path):
    """
    This functions takes names of different OpenFOAM velocity measurements and connect
    them into one file removing overlaps if any. All the probes must be in the same 
    location, otherwise an error might showup. 

    Parameters
    ----------
    *args 
        List of file paths of velocity data to be connected together. 

    Returns
    -------
    time, pressure
        Returns the velocity time and velocity data of the connected file.
    """

    num_files  = len(path)
    connected_time = [] # Connected array of time 
    connected_U = []  # connected array of pressure.

    time1 = [] 
    U1    = []
    time2 = []
    U2    = []
    probes = []
    
    for i in range(num_files):         
        probes, time2, U2 = read_velocity_probes(path[i])
        if i != 0:        
            try:
                index = np.where(time2 > time1[-1])[0][0]
            except:
                # sys.exit('Fatal Error!: the pressure files have time gap')
                index = 0 # Join them even if they have a time gap
            connected_time = np.concatenate((connected_time, time2[index:]))
            connected_U = np.concatenate((connected_U, U2[index:]))
        else:
            connected_time = time2
            connected_U = U2 

        time1 = time2
        U1 = U2
    shape = np.shape(connected_U)
    U = np.zeros((shape[1], shape[2], shape[0]))
    
    for i in range(shape[1]):
        for j in range(shape[2]):
            U[i,j,:] = connected_U[:,i,j]
    return probes, connected_time, U

def read_velocity_probes(fileName):
    """
    Created on Wed May 16 14:31:42 2018
    
    Reads velocity probe data from OpenFOAM and return the probe location, time, 
    and the velocity vector for each time step.
    """
    probes = []
    U = []
    time  = []
    
    with open(fileName, "r") as f:
        for line in f:
            if line.startswith('#'):
                if line.startswith('# Probe'):
                    line = line.replace('(','')
                    line = line.replace(')','')
                    line = line.split()
                    probes.append([float(line[3]), float(line[4]), float(line[5])])
                else:
                    continue
            else: 
                line = line.replace('(','')
                line = line.replace(')','')
                line = line.split()
                try:
                    time.append(float(line[0]))
                except:
                    continue
                u_probe_i = np.zeros([len(probes),3])
                for i in  range(len(probes)):
                    u_probe_i[i,:] = [float(line[3*i + 1]), float(line[3*i + 2]), float(line[3*i + 3])]
                U.append(u_probe_i)
    
    probes = np.asarray(probes, dtype=np.float32)
    time = np.asarray(time, dtype=np.float32)
    U = np.asarray(U, dtype=np.float32)

    return probes, time, U

def calculate_length_scale(u, uav, dt, min_corr=0.0):
    
     """
     Calculates the length scale of a velocity time history given.
    
     """   
     
     u = u - np.mean(u)

     corr = signal.correlate(u, u, mode='full')
    
     u_std = np.std(u)
    
     corr = corr[int(len(corr)/2):]/(u_std**2*len(u))
        
     loc = np.argmax(corr < min_corr)  

     corr = corr[:loc]
    
     L  = uav*np.trapz(corr, dx=dt)
   
     return L

def psd(x, dt, nseg):
    """
    Calculates the power spectral density of a given signal using the welch
    method. 

    Parameters
    ----------
    x 
        The time history of the signal.         
    dt
        The time step . 
    nseg
        The the number of segments to average the time series. 

    Returns
    -------
    freq, spectra
        Returns the frequency and spectra of the signal
    
    """
    x_no_mean = x - np.mean(x)
    freq, spectra = signal.welch(x_no_mean, fs=1.0/dt, nperseg=len(x_no_mean)/nseg)
       
    return freq[1:], spectra[1:]

class VelocityData:
    """
    A class that holds a velocity data and performs the following operations:
            - mean velocity profile 
            - turbulence intensity profiles  
            - integral scale of turbulence profiles      
    """
    def __init__(self, path,sampling_rate=400, filter_data=False, filter_freq=400, 
                 start_time=None, end_time=None, uDirn='x', resample_dt = None):
        self.path = path
        self.sampling_rate = sampling_rate
        self.filter_data = filter_data
        self.filter_freq = filter_freq
        self.start_time = start_time
        self.end_time = end_time
        self.component_count = 3
        self.uDirn = uDirn
        self.resample_dt = resample_dt
        self.__read_cfd_data()        
        self.__set_time()
        self.Nt = len(self.time)
        self.T = self.time[-1]        
        self.dt = np.mean(np.diff(self.time))    
        self.f_max = 1.0/(2.0*self.dt)
        self.probe_count = np.shape(self.probes)[0]
        self.Np = self.probe_count
        self.z = self.probes[:,2]
        self.y = self.probes[:,1]
        self.x = self.probes[:,0]
        self.__filter_signal()
        self.__calculate_all()

    def __read_cfd_data (self):
        if os.path.isdir(self.path):
            print("Reading from path : %s" % (self.path))
            time_names = os.listdir(self.path)
            sorted_index = np.argsort(np.float_(time_names)).tolist()
            file_names  = []
            
            for i in range(len(sorted_index)):
                file_name = os.path.join(self.path, time_names[sorted_index[i]], "U")
                file_names.append( file_name)
                
                
            self.probes, self.time, self.U = read_velocity_data(file_names)     
            
            
            # Coefficient of variation
            cv = np.std(np.diff(self.time))/np.mean(np.diff(self.time))
            
            if cv > 1.0e-4:
                self.__adjust_time_step()

        else:
            print("Cannot find the file path: %s" % (self.path))  

    
    
    def __adjust_time_step (self):
        
        if self.resample_dt == None:
           dt = np.mean(np.diff(self.time))
        else: 
           dt = self.resample_dt
        
        time = np.arange(start=self.time[0], stop=self.time[-1], step=dt)
        
        shape = np.shape(self.U)
        
        U = np.zeros((shape[0],shape[1],len(time)))

        for i in range(shape[0]):
            for j in range(shape[1]):
                U[i, j, :] = np.interp(time, self.time, self.U[i, j, :])
        
        self.time = time 
        self.U = U            

    
    def __filter_signal(self):
        if self.filter_data:
            low_pass = signal.butter(10, self.filter_freq,'lowpass', fs=self.sampling_rate, output='sos')
            for i in range(self.probe_count):
                for j in range(self.component_count):
                    self.U[i,j,:] = signal.sosfilt(low_pass, self.U[i,j,:])

    def __set_time (self):
        if(self.start_time != None):
            start_index = int(np.argmax(self.time > self.start_time))
            self.time = self.time[start_index:]
            self.U = self.U[:,:,start_index:]
            
        if(self.end_time != None):
            end_index = int(np.argmax(self.time > self.end_time))
            self.time = self.time[:end_index]
            self.U = self.U[:,:,:end_index]

    def __calculate_all(self):
        
        self.u = np.zeros((self.probe_count, self.component_count, self.Nt))

        #Calculate the mean velocity profile.
        if self.uDirn == 'x':
            self.Uav = np.mean(self.U[:,0,:], axis=1)
        if self.uDirn == 'y':
           self.Uav = np.mean(self.U[:,1,:], axis=1)
        if self.uDirn == 'z':
           self.Uav = np.mean(self.U[:,2,:], axis=1)
           
        #Calculate the turbulence intensity.
        self.I = np.std(self.U, axis=2) # gets the standard deviation
        self.Ru = np.var(self.U[:, 0, :], axis=1) # gets reynolds stress
        self.Rv = np.var(self.U[:, 1, :], axis=1) # gets reynolds stress
        self.Rw = np.var(self.U[:, 2, :], axis=1) # gets reynolds stress
        
        for i in range(self.component_count):
            self.I[:,i] = self.I[:,i]/self.Uav
            
        
        #Calculate the length scale profiles. 
        self.L = np.zeros((self.probe_count, self.component_count))
        for i in range(self.probe_count):
            for j in range(self.component_count):
                self.u[i,j,:] = self.U[i,j,:] - np.mean(self.U[i,j,:])
                self.L[i,j] = calculate_length_scale(self.u[i,j,:], self.Uav[i], self.dt, 0.05)


        #Calculate the shear stress profiles. 
        self.uv_bar = np.zeros(self.Np)
        self.uw_bar = np.zeros(self.Np)
        
        for i in range(self.Np):
            self.uv_bar[i] = np.cov(self.U[i,0,:], self.U[i,1,:])[0,1]
            self.uw_bar[i] = np.cov(self.U[i,0,:], self.U[i,2,:])[0,1]

    def get_Uav(self, z, dixn='z'):
        from scipy import interpolate
        
        if dixn == 'x':
            f = interpolate.interp1d(self.x, self.Uav)
        elif dixn == 'y':
            f = interpolate.interp1d(self.y, self.Uav)
        else:
            f = interpolate.interp1d(self.z, self.Uav)
        
        return f(z)


def plot_wind_profiles(case_path, prof_name):
    
    #Read JSON data    
    json_path =  os.path.join(case_path, "constant", "simCenter", "input", "EmptyDomainCFD.json")
    with open(json_path) as json_file:
        json_data =  json.load(json_file)
      
    # Returns JSON object as a dictionary
    wc_data = json_data["windCharacteristics"]
    rm_data = json_data['resultMonitoring']

    prof_path  = os.path.join(case_path,  "postProcessing", prof_name)
    
    prof = VelocityData(prof_path, start_time=None, end_time=None)
    
    output_path = os.path.join(case_path, "constant", "simCenter", "output", "windProfiles")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    Path(output_path).mkdir(parents=True, exist_ok=True)

    #Create wind profile data profile z, Uav, Iu ..., Lu ...,
    prof_np = np.zeros((len(prof.y), 9))
    prof_np[:,0] = prof.y
    prof_np[:,1] = prof.Uav
    prof_np[:,2] = prof.I[:,0]
    prof_np[:,3] = prof.I[:,1]
    prof_np[:,4] = prof.I[:,2]
    prof_np[:,5] = prof.uv_bar/np.square(prof.Uav)
    prof_np[:,6] = prof.L[:,0]
    prof_np[:,7] = prof.L[:,1]
    prof_np[:,8] = prof.L[:,2]
   
    subplot_titles = ("Mean Velocity", "Turbulence Intensity, Iu", "Turbulence Intensity, Iv", "Turbulence Intensity, Iw",
                       "Shear Stress", "Length Scale, Lu", "Length Scale, Lv", "Length Scale, Lw")
    
    fig = make_subplots(rows=2, cols=4, start_cell="top-left", subplot_titles=subplot_titles, vertical_spacing=0.15)



    # Mean Velocity
    fig.add_trace(go.Scatter(x=prof_np[:,1], y=prof_np[:,0], 
                             mode='lines+markers', name='Inflow', ), row=1, col=1)
    fig.update_xaxes(title_text="$U_{av} [m/s]$", range=[0, 1.15*np.max(prof_np[:,1])], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=1, col=1)
    fig.update_yaxes(title_text="$z [m]$", range=[0, 1.01*np.max(prof_np[:,0])], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=1, col=1)
   

    # Turbulence Intensity Iu
    fig.add_trace(go.Scatter(x=prof_np[:,2], y=prof_np[:,0], 
                             mode='lines+markers', name='Inflow', ), row=1, col=2)
    fig.update_xaxes(title_text="$I_{u}$", range=[0, 1.3*np.max(prof_np[:,2])], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=1, col=2)
    fig.update_yaxes(title_text="", range=[0, 1.01*np.max(prof_np[:,0])], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=1, col=2)

    # Turbulence Intensity Iv
    fig.add_trace(go.Scatter(x=prof_np[:,3], y=prof_np[:,0], 
                             mode='lines+markers', name='Inflow', ), row=1, col=3)
    fig.update_xaxes(title_text="$I_{v}$", range=[0, 1.3*np.max(prof_np[:,3])], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=1, col=3)
    fig.update_yaxes(title_text="", range=[0, 1.01*np.max(prof_np[:,0])], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=1, col=3)

    # Turbulence Intensity Iw
    fig.add_trace(go.Scatter(x=prof_np[:,4], y=prof_np[:,0], 
                             mode='lines+markers', name='Inflow', ), row=1, col=4)
    fig.update_xaxes(title_text="$I_{w}$", range=[0, 1.3*np.max(prof_np[:,4])], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=1, col=4)
    fig.update_yaxes(title_text="", range=[0, 1.01*np.max(prof_np[:,0])], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=1, col=4)



    # Shear Stress Profile 
    fig.add_trace(go.Scatter(x=prof_np[:,5], y=prof_np[:,0], 
                             mode='lines+markers', name='Inflow', ), row=2, col=1)
    fig.update_xaxes(title_text=r'$\overline{uv}/U^2_{av}$', range=[1.3*np.min(prof_np[:,5]), 1.5*np.max(prof_np[:,5])], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=2, col=1)
    fig.update_yaxes(title_text="$z [m]$", range=[0, 1.01*np.max(prof_np[:,0])], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=2, col=1)


    # Length scale Lu
    fig.add_trace(go.Scatter(x=prof_np[:,6], y=prof_np[:,0], 
                             mode='lines+markers', name='Inflow', ), row=2, col=2)
    fig.update_xaxes(title_text="$L_{u} [m]$", range=[0, 1.5*np.max(prof_np[:,6])], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=2, col=2)
    fig.update_yaxes(title_text="", range=[0, 1.01*np.max(prof_np[:,0])], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=2, col=2)


    # Length scale Lv
    fig.add_trace(go.Scatter(x=prof_np[:,7], y=prof_np[:,0], 
                             mode='lines+markers', name='Inflow', ), row=2, col=3)
    fig.update_xaxes(title_text="$L_{v} [m]$", range=[0, 1.5*np.max(prof_np[:,7])], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=2, col=3)
    fig.update_yaxes(title_text="", range=[0, 1.01*np.max(prof_np[:,0])], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=2, col=3)


    # Length scale Lw
    fig.add_trace(go.Scatter(x=prof_np[:,8], y=prof_np[:,0], 
                             mode='lines+markers', name='Inflow', ), row=2, col=4)
    fig.update_xaxes(title_text="$L_{w} [m]$", range=[0, 1.5*np.max(prof_np[:,8])], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=2, col=4)
    fig.update_yaxes(title_text="", range=[0, 1.01*np.max(prof_np[:,0])], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=2, col=4)


    fig.update_layout(height=850, width=1200, title_text="",showlegend=False)
    fig.show()
    fig.write_html(os.path.join(output_path, prof_name + ".html"), include_mathjax="cdn")



    subplot_titles = ("u-component", "v-component", "w-component")
    
    fig = make_subplots(rows=1, cols=3, start_cell="top-left", subplot_titles=subplot_titles, vertical_spacing=0.15)



    #u-component
    fig.add_trace(go.Scatter(x=prof_np[:,1], y=prof_np[:,0], 
                             mode='lines+markers', name='Inflow', ), row=1, col=1)
    
    fig.update_xaxes(type="log", title_text="$U_{av} [m/s]$", range=[0, 1.15*np.max(prof_np[:,1])], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=1, col=1)
    
    fig.update_yaxes(type="log", title_text="$z [m]$", range=[0, 1.01*np.max(prof_np[:,0])], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=1, col=1)

    #v-component
    fig.add_trace(go.Scatter(x=prof_np[:,1], y=prof_np[:,0], 
                             mode='lines+markers', name='Inflow', ), row=1, col=1)
    
    fig.update_xaxes(type="log", title_text="$U_{av} [m/s]$", range=[0, 1.15*np.max(prof_np[:,1])], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=1, col=1)
    fig.update_yaxes(type="log", title_text="$z [m]$", range=[0, 1.01*np.max(prof_np[:,0])], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=1, col=1)
    

    #v-component
    fig.add_trace(go.Scatter(x=prof_np[:,1], y=prof_np[:,0], 
                             mode='lines+markers', name='Inflow', ), row=1, col=1)
    
    fig.update_xaxes(type="log", title_text="$U_{av} [m/s]$", range=[0, 1.15*np.max(prof_np[:,1])], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=1, col=1)
    fig.update_yaxes(type="log", title_text="$z [m]$", range=[0, 1.01*np.max(prof_np[:,0])], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=1, col=1)

if __name__ == '__main__':    
    
    input_args = sys.argv

    # # Set filenames
    # case_path = sys.argv[1]
    # prof_name = sys.argv[2]

    case_path = "C:\\Users\\fanta\\OneDrive\\Documents\\WE-UQ\\LocalWorkDir\\EmptyDomainCFD"
    prof_name = "probe7H"
    
    plot_wind_profiles(case_path, prof_name)
    
