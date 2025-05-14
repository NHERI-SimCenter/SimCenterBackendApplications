# -*- coding: utf-8 -*-  # noqa: INP001, D100, UP009
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
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGESG
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

import json
import numpy as np
import foam_dict_reader as foam
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_wind_profiles_and_spectra(case_path, prof):
    
    save_path =  os.path.join(case_path, "constant", "simCenter", "input", "targetWindProfiles.html")
    
    z = prof[:,0]
    U = prof[:,1]
    Iu = np.sqrt(prof[:, 2])/U
    Iv = np.sqrt(prof[:, 5])/U
    Iw = np.sqrt(prof[:, 7])/U
    uw = prof[:, 4]/(U**2.0)
    xLu = prof[:, 8]
    xLv = prof[:, 11]
    xLw = prof[:, 14]
    
    
    
    subplot_titles = ("Mean Velocity", "Turbulence Intensity, Iu", "Turbulence Intensity, Iv", "Turbulence Intensity, Iw",
                       "Shear Stress", "Length Scale, Lu", "Length Scale, Lv", "Length Scale, Lw")
    
    fig = make_subplots(rows=2, cols=4, start_cell="top-left", subplot_titles=subplot_titles, vertical_spacing=0.15)

        
    fig.add_trace(go.Scatter(x=U, y=z, line=dict(color='black', width=3.0, dash='dot'),
                             mode='lines', name='Target', ), row=1, col=1)
    
    fig.update_xaxes(title_text="$U_{av} [m/s]$", range=[0, 1.25*np.max(U)], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=1, col=1)
    fig.update_yaxes(title_text="$z [m]$", range=[0, 1.01*np.max(z)], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=1, col=1)
   

    # Turbulence Intensity Iu
    fig.add_trace(go.Scatter(x=Iu, y=z, line=dict(color='black', width=3.0, dash='dot'),
                             mode='lines', name='Target', ), row=1, col=2)

    fig.update_xaxes(title_text="$I_{u}$", range=[0, 1.3*np.max(Iu)], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=1, col=2)
    fig.update_yaxes(title_text="", range=[0, 1.01*np.max(z)], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=1, col=2)

    # Turbulence Intensity Iv
    fig.add_trace(go.Scatter(x=Iv, y=z, line=dict(color='black', width=3.0, dash='dot'),
                             mode='lines', name='Target', ), row=1, col=3)

    fig.update_xaxes(title_text="$I_{v}$", range=[0, 1.3*np.max(Iv)], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=1, col=3)
    fig.update_yaxes(title_text="", range=[0, 1.01*np.max(z)], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=1, col=3)

    # Turbulence Intensity Iw
    fig.add_trace(go.Scatter(x=Iw, y=z, line=dict(color='black', width=3.0, dash='dot'),
                             mode='lines', name='Target', ), row=1, col=4)

    fig.update_xaxes(title_text="$I_{w}$", range=[0, 1.3*np.max(Iw)], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=1, col=4)
    fig.update_yaxes(title_text="", range=[0, 1.01*np.max(z)], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=1, col=4)


    # Shear Stress Profile Ruw
    
    fig.add_trace(go.Scatter(x=-uw, y=z, line=dict(color='black', width=3.0, dash='dot'),
                             mode='lines', name='Target', ), row=2, col=1)
    fig.update_xaxes(title_text=r'$-\overline{uw}/U^2_{av}$', range=[0.0, 1.3*np.max(-uw)], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=2, col=1)
    fig.update_yaxes(title_text="$z [m]$", range=[0, 1.01*np.max(z)], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=2, col=1)


    # Length scale Lu
    fig.add_trace(go.Scatter(x=xLu, y=z, line=dict(color='black', width=3.0, dash='dot'),
                             mode='lines', name='Target', ), row=2, col=2)
    fig.update_xaxes(title_text="${}^{x}L_{u} [m]$", range=[0, 1.5*np.max(xLu)], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=2, col=2)
    fig.update_yaxes(title_text="", range=[0, 1.01*np.max(z)], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=2, col=2)


    # Length scale Lv
    fig.add_trace(go.Scatter(x=xLv, y=z, line=dict(color='black', width=3.0, dash='dot'),
                             mode='lines', name='Target', ), row=2, col=3)

    fig.update_xaxes(title_text="${}^{x}L_{v} [m]$", range=[0, 1.5*np.max(xLv)], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=2, col=3)
    fig.update_yaxes(title_text="", range=[0, 1.01*np.max(z)], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=2, col=3)


    # Length scale Lw
    fig.add_trace(go.Scatter(x=xLw, y=z, line=dict(color='black', width=3.0, dash='dot'),
                             mode='lines', name='Target', ), row=2, col=4)
    fig.update_xaxes(title_text="${}^{x}L_{w} [m]$", range=[0, 1.5*np.max(xLw)], 
                     showline=True, linewidth=1.5, linecolor='black',ticks='outside', row=2, col=4)
    fig.update_yaxes(title_text="", range=[0, 1.01*np.max(z)], showline=True, 
                     linewidth=1.5, linecolor='black',ticks='outside', row=2, col=4)


    fig.update_layout(height=850, width=1200, title_text="",showlegend=False)
    fig.write_html(save_path, include_mathjax="cdn")



def get_u_star(z_0, u_ref, z_ref):
    """
    Calculates the friction velocity in ASL determined from C2-1 in ASCE 49-21
    The main assumption is this value is constant over the ASL height. 
    """
    k = 0.4
    return u_ref*k/np.log(z_ref/z_0)


def get_uw(z, z_g, u_star):
    """
    Calculate the Reynolds shear stress component uw based on height. Taken from 
    ESDU 85020

    Parameters:
    z : float or np.array
        Height above the ground (m)
    zg : float
        Boundary layer height (m)
    u_star : float
        Friction velocity (m/s)

    Returns:
    uw : float or np.array
        Reynolds stress component uw (m^2/s^2)
    """

    z = np.asarray(z)  # Ensure z can be used as an array

    return -u_star**2.0*(1.0 - z/z_g)**2.0

def get_fc(lat):
    """ 
    Calculates the Coriolis parameter (fc) based on latitude.
    Formula taken from ASCE 49-21.
    fc = 2 * omega * sin(latitude)
    where omega is the Earth's angular velocity (7.2921e-5 rad/s).
    (360 degrees) takes approximately 23 hours, 56 minutes,
    
    """
    omega = 7.2921e-5 # Angular velocity of Earth

    return 2.0*omega*np.sin(np.deg2rad(lat)) 



def get_z_g(lat, u_star):
    """
    Calculates the gradient height for the ABL, given latitude and friction velocity.
    Taken from ASCE 7 49-21
    """

    f_c = get_fc(lat)
    a = 1.0/6.0

    return a*u_star/f_c

def get_z_s(lat, u_star):
    """
    Calculates the gradient height for the ASL given latitude and friction velocity.
    """
    f_c = get_fc(lat)

    return 0.02*u_star/f_c
                   

def get_mean_u(z_0, z_g, u_star, z):
    """
    Calculate the mean velocity profile U(z)/u_star based on Harris and Deaves (1980).

    Parameters:
    z     : float or array-like - height(s) at which to evaluate the velocity
    z_0    : float - aerodynamic surface roughness length
    z_g    : float - gradient height
    u_star: float - friction velocity

    Returns:
    U     : float or array-like - mean velocity at height z
    """
    
    kappa = 0.4  # von Kármán constant
    a1 = 5.75
    a2 = -1.87
    a3 = -1.33
    a4 = 0.25

    z = np.array(z, dtype=float)
    zeta = z / z_g

    z = np.maximum(z, z_0 * 1.000001)  # Prevent log(0)

    U_by_u_star = (1.0 / kappa) * (
        np.log(z / z_0) +
        a1 * zeta +
        a2 * zeta**2 +
        a3 * zeta**3 +
        a4 * zeta**4
    )

    return u_star * U_by_u_star


# def get_TI(z_0, z_g, u_star, z):
#     """
#     Computes longitudinal, crosswind, and vertical turbulence intensities (Iu, Iv, Iw)
#     throughout the full ABL based on height z.

#     Parameters:
#     - z     : float or np.array - height(s) above ground
#     - z0    : float - aerodynamic roughness length
#     - zg    : float - boundary layer height
#     - u_star: float - friction velocity

#     Returns:
#     - Iu, Iv, Iw : arrays of turbulence intensities at height(s) z
#     """
#     z = np.array(z, dtype=float)
#     U = get_mean_u(z_0, z_g, u_star, z)

#     Iu = 2.5 * u_star * np.exp(-1.5 * z / z_g) / U
#     linear_factor = np.maximum(0, 1 - 0.5 * (z / z_g))  # Ensure non-negativity
#     Iv = 1.60 * u_star * linear_factor / U
#     Iw = 1.25 * u_star * linear_factor / U

#     return Iu, Iv, Iw


def get_TI(z_0, z_g, u_star, z):
    """
    Computes longitudinal, crosswind, and vertical turbulence intensities (Iu, Iv, Iw)
    throughout the full ABL based on height z.
    
    Based on the equation on C2-24 and scaling in C2-18a&b

    Parameters:
    - z     : float or np.array - height(s) above ground
    - z_0    : float - aerodynamic roughness length
    - z_g    : float - boundary layer height
    - u_star: float - friction velocity

    Returns:
    - Iu, Iv, Iw : arrays of turbulence intensities at height(s) z
    """

    z = np.array(z, dtype=float)
    U_z = get_mean_u(z_0, z_g, u_star, z)

    I = np.zeros((len(z), 3))

    eta = 1.0 - z/z_g
    eta = np.clip(eta, 0, 1.0)
    
    I[:, 0] = 2.63*u_star*eta*((0.538 + 0.090*np.log(z/z_0))**(eta**16.0))/U_z
    
    I[:, 1] = 0.8*I[:, 0]
    I[:, 2] = 0.5*I[:, 0]
    
    return I

def get_LS(z_0, z): 
    """
    Integral lenght scale calculation based on Simiu and Yeo 2019, Figure 2.6
    """
    z = np.array(z, dtype=float)
    
    # Compute constant C based on surface roughness z_0 using empirical power-law fit
    C = 21.04*z_0**(-0.42)

    # Compute exponent m as a function of z_0 using a semi-logarithmic fit
    m = 0.0624*np.log(z_0) + 0.417

    # Compute the streamwise integral length scale (L_u) as a function of height z
    xLu = C * z**m

    #Holds the nine components 
    L = np.zeros((len(z), 9))
        
    # Estimate lateral (L_v) and vertical (L_w) integral length scales
    L[:, 0] = xLu
    L[:, 1] = 0.3 * xLu
    L[:, 2] = 0.5 * xLu

    L[:, 3] = 0.46 * xLu
    L[:, 4] = 0.14 * xLu
    L[:, 5] = 0.32 * xLu
    
    L[:, 6] = 0.20 * xLu
    L[:, 7] = 0.06 * xLu
    L[:, 8] = 0.07 * xLu
    return L


def generate_wind_profiles(lat, z_0, u_ref, z_ref, z, l_scale=1.0, v_scale=1.0):
    
    n_points = np.shape(z)[0]
    
    u_star = get_u_star(z_0, u_ref, z_ref)
    z_g = get_z_g(lat, u_star)
    
    # print("U_star", u_star)
    # print("z_g", z_g)
    # print("u_ref", u_ref)
    # print("z_ref", z_ref)
    # print("lat", lat)
    
    r_scale = v_scale**2.0

    U = get_mean_u(z_0, z_g, u_star, z)*v_scale
    I = get_TI(z_0, z_g, u_star, z)
    L = get_LS(z_0, z)*l_scale
    uw = get_uw(z, z_g, u_star)*r_scale
    
    
    wind_profiles = np.zeros((n_points, 17))
    
    
    wind_profiles[:, 0] = l_scale*z
    wind_profiles[:, 1] = U
    wind_profiles[:, 2] = (I[:,0]*U)**2.0 #Ruu
    wind_profiles[:, 3] = 0.0 # Ruv = 0
    wind_profiles[:, 4] = uw # Ruw
    wind_profiles[:, 5] = (I[:,1]*U)**2.0 # Rvv
    wind_profiles[:, 6] = 0.0 # Rvw
    wind_profiles[:, 7] = (I[:,2]*U)**2.0 # Rww
    wind_profiles[:, 8:] = L
    
    return wind_profiles

def write_boundary_data_files(input_json_path, case_path):
    """
    This functions writes wind profile files in "constant/boundaryData/inlet"
    if TInf options are used for the simulation. Otherwise, simple boundary 
    conditions settings from OpenFOAM is used.  
    """  
    # Read JSON data
    with open(input_json_path + '/IsolatedBuildingCFD.json') as json_file:  # noqa: PTH123
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    boundary_data = json_data['boundaryConditions']
    geom_data = json_data['GeometricData']    
    wind_data = json_data['windCharacteristics']

    l_scale = 1.0/float(geom_data['geometricScale'])    
    v_scale = 1.0/float(wind_data['velocityScale'])

    norm_type = geom_data['normalizationType']
    building_height = l_scale*geom_data['buildingHeight']

    z0 = wind_data['aerodynamicRoughnessLength']
    fs_ref_height = wind_data['referenceHeight']/l_scale    
    fs_wind_speed = wind_data['referenceWindSpeed']/v_scale

    origin = np.array(geom_data['origin'])

    Ly = geom_data['domainWidth']  # noqa: N806
    Lz = geom_data['domainHeight']  # noqa: N806
    Lf = geom_data['fetchLength']  # noqa: N806

    if norm_type == 'Relative':
        Ly *= building_height  # noqa: N806
        Lz *= building_height  # noqa: N806
        Lf *= building_height  # noqa: N806

    x_min = -Lf - origin[0]
    y_min = -Ly / 2.0 - origin[1]
    y_max = y_min + Ly
    
    bd_path = case_path + '/constant/boundaryData/inlet/'

    if boundary_data['inletBoundaryCondition'] == 'TInf':
        
        latitude = 45.0

        if boundary_data['inflowProperties']['windProfileOption'] == 'Table':            
            wind_profiles = np.array(boundary_data['inflowProperties']['windProfiles'])

        if boundary_data['inflowProperties']['windProfileOption'] == 'ASCE-49':
            latitude = boundary_data['inflowProperties']['latitude']
            n_samples = 200 
            dz = Lz/n_samples
            z = np.linspace(dz/2.0, Lz, n_samples, endpoint=False)
            
            wind_profiles = generate_wind_profiles(latitude, z0, fs_wind_speed, fs_ref_height, z, l_scale, v_scale)
                        
            csv_path = os.path.join(case_path, 'constant', 'simCenter', 'input', 'windProfiles.csv')
            np.savetxt(csv_path, wind_profiles, delimiter=",", fmt="%.6f")

                    
        


        # Write points file
        n_pts = np.shape(wind_profiles)[0]
        points = np.zeros((n_pts, 3))
        
        points[:, 0] = x_min
        points[:, 1] = (y_min + y_max)/2.0
        points[:, 2] = wind_profiles[:, 0]

        # Shift the last element of the y coordinate
        # a bit to make planer interpolation easier
        points[-1:, 1] = y_max

        foam.write_foam_field(points, bd_path + 'points')

        # Write wind speed file as a scalar field
        foam.write_scalar_field(wind_profiles[:, 1], bd_path + 'U')

        # Write Reynolds stress profile (6 columns -> it's a symmetric tensor field)
        foam.write_foam_field(wind_profiles[:, 2:8], bd_path + 'R')

        # Write length scale file (8 columns -> it's a tensor field)
        foam.write_foam_field(wind_profiles[:, 8:17], bd_path + 'L')

        plot_wind_profiles_and_spectra(case_path, wind_profiles)

# if __name__ == '__main__':  
    
#     # input_args = sys.argv

#     # Set file names
#     # input_json_path = sys.argv[1]
#     # case_path = sys.argv[2]

#     case_path = "C:\\Users\\fanta\\Documents\\WE-UQ\\LocalWorkDir\\IsolatedBuildingCFD"
#     input_json_path = "C:\\Users\\fanta\\Documents\\WE-UQ\\LocalWorkDir\\IsolatedBuildingCFD\\constant\\simCenter\\input\\"

#     # Write the files
#     write_boundary_data_files(input_json_path, case_path)
    
    
    

