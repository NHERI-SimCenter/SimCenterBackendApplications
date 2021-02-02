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
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, interp2d

def baker_jayaram_correlation_2008(T1, T2, flag_orth = False):
    """
    Computing inter-event correlation coeffcieint between Sa of two periods
    Reference:
        Baker and Jayaram (2008) Correlation of Spectral Acceleration
        Values from NGA Ground Motion Models
    Input:
        T1: period 1 in second
        T2: period 2 in second
        flag_orth: if the correlation coefficient is computed for the two
                   orthogonal components
    Output:
        rho: correlation coefficient
    Note:
        The valid range of T1 and T2 is 0.01s ~ 10.0s
    """

    Tmin = min([T1, T2])
    Tmax = max([T1, T2])
    # Cofficient C1
    C1 = 1.0 - np.cos(np.pi / 2.0 - 0.366 * np.log(Tmax / max([Tmin, 0.109])))
    # Cofficient C2
    if Tmax < 0.2:
        C2 = 1.0 - 0.105 * (1.0 - 1.0 / (1.0 + np.exp(100.0 * Tmax - 5.0))) * \
            (Tmax - Tmin) / (Tmax - 0.0099)
    else:
        C2 = 0.0
    # Cofficient C3
    if Tmax < 0.109:
        C3 = C2
    else:
        C3 = C1
    # Cofficient C4
    C4 = C1 + 0.5 * (np.sqrt(C3) - C3) * (1.0 + np.cos(np.pi * Tmin / 0.109))
    # rho for a singe component
    if Tmax <= 0.109:
        rho = C2;
    elif Tmin > 0.109:
        rho = C1
    elif Tmax < 0.2:
        rho = min([C2, C4])
    else:
        rho = C4
    # rho for orthogonal components Cofficient C1
    if flag_orth:
        rho = rho * (0.79 - 0.023 * np.log(np.sqrt(Tmin * Tmax)))

    return rho


def bradley_correlation_2011(IM, T = None, flag_Ds = True):
    """
    Computing inter-event correlation coeffcieint between Sa(T) and Ds575/D595
    Reference:
        Bradley (2011) Correlation of Significant Duration with Amplitude and
        Cumulative Intensity Measures and Its Use in Ground Motion Selection
    Input:
        IM: string of intensity measure from options as follows
            'Sa', 'PGA', 'PGV', 'ASI', 'SI', 'DSI', 'CAV', 'Ds595'
        T: Sa period
        flag_Ds: true - Ds575, false = Ds595
    Output:
        rho: correlation coefficient
    Note:
        The valid range of T is 0.01s ~ 10.0s
    """
    # PGA
    if IM == 'PGA':
        if flag_Ds:
            return -0.442
        else:
            return -0.305
    elif IM == 'PGV':
        if flag_Ds:
            return -0.259
        else:
            return -0.211
    elif IM == 'ASI':
        if flag_Ds:
            return -0.411
        else:
            return -0.370
    elif IM == 'SI':
        if flag_Ds:
            return -0.131
        else:
            return -0.079
    elif IM == 'DSI':
        if flag_Ds:
            return 0.074
        else:
            return 0.163
    elif IM == 'CAV':
        if flag_Ds:
            return 0.077
        else:
            return 0.122
    elif IM == 'Ds595':
        if flag_Ds:
            return 0.843
        else:
            return None
    elif IM == 'Sa':
        if flag_Ds:
            if T < 0.09:
                a_p = -0.45; a_c = -0.39; b_p = 0.01; b_c = 0.09
            elif T < 0.30:
                a_p = -0.39; a_c = -0.39; b_p = 0.09; b_c = 0.30
            elif T < 1.40:
                a_p = -0.39; a_c = -0.06; b_p = 0.30; b_c = 1.40
            elif T < 6.50:
                a_p = -0.06; a_c = 0.16; b_p = 1.40; b_c = 6.50
            elif T <= 10.0:
                a_p = 0.16; a_c = 0.00; b_p = 6.50; b_c = 10.00
        else:
            if T < 0.04:
                a_p = -0.41; a_c = -0.41; b_p = 0.01; b_c = 0.04
            elif T < 0.08:
                a_p = -0.41; a_c = -0.38; b_p = 0.04; b_c = 0.08
            elif T < 0.26:
                a_p = -0.38; a_c = -0.35; b_p = 0.08; b_c = 0.26
            elif T < 1.40:
                a_p = -0.35; a_c = -0.02; b_p = 0.26; b_c = 1.40
            elif T <= 6.00:
                a_p = -0.02; a_c = 0.23; b_p = 1.40; b_c = 6.00
            elif T <= 10.00:
                a_p = 0.23; a_c = 0.02; b_p = 6.00; b_c = 10.0
        rho = a_p + np.log(T / b_p) / np.log(b_c / b_p) * (a_c - a_p)
        return rho


def jayaram_baker_correlation_2009(T, h, flag_clustering = False):
    """
    Computing intra-event correlation coeffcieint between Sa(T) at two sites
    Reference:
        Jayaram and Baker (2009) Correlation model for spatially distributed
        ground-motion intensities
    Input:
        T: Sa period
        h: distance between the two sites
        flag_clustering: the geologic condition of the soil varies widely over
                         the region (default: false)
    Output:
        rho: correlation between normalized intra-event residuals
    """

    if T >= 1.0:
        b = 22.0 + 3.7 * T
    else:
        if flag_clustering:
            b = 8.5 + 17.2 * T
        else:
            b = 40.7 - 15.0 * T
    rho = np.exp(-3.0 * h / b)
    return rho


def load_loth_baker_correlation_2013(datapath):
    """
    Loading the three matrices in the Loth-Baker correaltion model (2013)
    Reference:
        Loth and Baker (2013) A spatial cross-correlation model of spectral
        accelerations at multiple periods (with the Erratum)
    Input:
        datapath: the path to the files (optional)
    Output:
        B1: short-range coregionalization matrix
        B2: long-range coregionalization matrix
        B3: Nugget effect correlationalization matrix
    """
    B2 = pd.read_csv(datapath + 'loth_baker_correlation_2013_B2.csv', header = 0)
    B1 = pd.read_csv(datapath + 'loth_baker_correlation_2013_B1.csv', header = 0)
    B3 = pd.read_csv(datapath + 'loth_baker_correlation_2013_B3.csv', header = 0)
    return B1, B2, B3


def compute_rho_loth_baker_correlation_2013(T1, T2, h, B1, B2, B3):
    """
    Computing intra-event correlation coeffcieint between Sa(Ti) and Sa(Tj)
    at two sites
    Reference:
        Loth and Baker (2013) A spatial cross-correlation model of spectral
        accelerations at multiple periods (with the Erratum)
    Input:
        T1: Sa period 1
        T2: Sa period 2
        h: site distance
        B1: short-range coregionalization matrix
        B2: long-range coregionalization matrix
        B3: Nugget effect correlationalization matrix
    Output:
        rho: correlation between Sa(Ti) and Sa(Tj) at two sites
    Note:
        The valid range for T1 and T2 is 0.01s ~ 10.0s
    """
    # Interpolation functions
    f1 = interp2d(B1['Period (s)'], B1['Period (s)'], B1.iloc[:, 1:])
    f2 = interp2d(B2['Period (s)'], B2['Period (s)'], B2.iloc[:, 1:])
    f3 = interp2d(B3['Period (s)'], B3['Period (s)'], B3.iloc[:, 1:])
    # Three coefficients
    b1 = f1(T1, T2)
    b2 = f2(T1, T2)
    b3 = f3(T1, T2)
    # Covariance functions
    Ch = b1 * np.exp(-3.0 * h / 20.0) + b2 * np.exp(-3.0 * h / 70.0) + b3 * (h == 0)
    # Correlation coefficient
    rho = Ch
    return rho


def loth_baker_correlation_2013(stations, periods, num_simu):
    """
    Simulating intra-event residuals
    Reference:
        Loth and Baker (2013) A spatial cross-correlation model of spectral
        accelerations at multiple periods (with the Erratum)
    Input:
        stations: stations coordinates
        periods: simulated spectral periods
        num_simu: number of realizations
    Output:
        residuals: intra-event residuals
    Note:
        The valid range for T1 and T2 is 0.01s ~ 10.0s
    """
    # Loading modeling coefficients
    B1, B2, B3 = load_loth_baker_correlation_2013(os.path.dirname(__file__) + '/data/')
    # Computing distance matrix
    num_stations = len(stations)
    stn_dist = np.zeros((num_stations, num_stations))
    for i in range(num_stations):
        loc_i = np.array([stations[i]['Latitude'],
                          stations[i]['Longitude']])
        for j in range(num_stations):
            loc_j = np.array([stations[j]['Latitude'],
                              stations[j]['Longitude']])
            stn_dist[i, j] = np.linalg.norm(loc_i - loc_j) * 111.0
    # Creating a covariance matrices for each of the principal components
    num_periods = len(periods)
    covMatrix = np.zeros((num_stations * num_periods, num_stations * num_periods))
    for i in range(num_periods):
        for j in range(num_periods):
            covMatrix[num_stations * i:num_stations * (i + 1), num_stations * j:num_stations * (j + 1)] = \
                compute_rho_loth_baker_correlation_2013(periods[i], periods[j], stn_dist, B1, B2, B3)

    mu = np.zeros(num_stations * num_periods)
    residuals_raw = np.random.multivariate_normal(mu, covMatrix, num_simu).T
    residuals = residuals_raw.reshape(num_simu, num_stations, num_periods).swapaxes(0,1).swapaxes(1,2)
    # return
    return residuals


def load_markhvida_ceferino_baker_correlation_2017(datapath):
    """
    Loading the three matrices in the Markhivida et al. correaltion model (2017)
    Reference:
        Markhvida et al. (2017) Modeling spatially correlated spectral
        accelerations at multiple periods using principal component analysis
        and geostatistics
    Input:
        datapath: the path to the files (optional)
    Output:
        B1: short-range coregionalization matrix
        B2: long-range coregionalization matrix
        B3: Nugget effect correlationalization matrix
    """
    MCB_model = pd.read_csv(datapath + 'markhvida_ceferino_baker_correlation_2017_model_coeff.csv',
                            index_col = None, header = 0)
    MCB_pca = pd.read_csv(datapath + 'markhvida_ceferino_baker_correlation_2017_pca_coeff.csv',
                            index_col = None, header = 0)
    MCB_var = pd.read_csv(datapath + 'markhvida_ceferino_baker_correlation_2017_var_scale.csv',
                            index_col = None, header = 0)
    return MCB_model, MCB_pca, MCB_var


def markhvida_ceferino_baker_correlation_2017(stations, periods, num_simu, num_pc):
    """
    Simulating intra-event residuals
    Reference:
        Markhvida et al. (2017) Modeling spatially correlated spectral
        accelerations at multiple periods using principal component analysis
        and geostatistics
    Input:
        stations: stations coordinates
        periods: simulated spectral periods
        num_simu: number of realizations
        num_pc: number of principle components
    Output:
        residuals: intra-event residuals
    Note:
        The valid range for T1 and T2 is 0.01s ~ 5.0s
    """
    # Loading factors
    MCB_model, MCB_pca, MCB_var = \
        load_markhvida_ceferino_baker_correlation_2017(os.path.dirname(__file__) + '/data/')
    c0 = MCB_model.loc[MCB_model['Type'] == 'c0']
    c0 = c0[c0.keys()[1:]]
    c1 = MCB_model.loc[MCB_model['Type'] == 'c1']
    c1 = c1[c1.keys()[1:]]
    c2 = MCB_model.loc[MCB_model['Type'] == 'c2']
    c2 = c2[c2.keys()[1:]]
    a1 = MCB_model.loc[MCB_model['Type'] == 'a1']
    a1 = a1[a1.keys()[1:]]
    a2 = MCB_model.loc[MCB_model['Type'] == 'a2']
    a2 = a2[a2.keys()[1:]]
    model_periods = MCB_pca['Period (s)']
    model_coef = MCB_pca.iloc[:, 1:num_pc + 1]
    # Computing distance matrix
    num_stations = len(stations)
    stn_dist = np.zeros((num_stations, num_stations))
    for i in range(num_stations):
        loc_i = np.array([stations[i]['Latitude'],
                          stations[i]['Longitude']])
        for j in range(num_stations):
            loc_j = np.array([stations[j]['Latitude'],
                              stations[j]['Longitude']])
            stn_dist[i, j] = np.linalg.norm(loc_i - loc_j) * 111.0
    # Scaling variance if less than 19 principal components are used
    c0 = c0 / MCB_var.iloc[0, num_pc - 1]
    c1 = c1 / MCB_var.iloc[0, num_pc - 1]
    c2 = c2 / MCB_var.iloc[0, num_pc - 1]
    # Creating a covariance matrices for each of the principal components
    covMatrix = np.zeros((num_stations, num_stations, num_pc))
    for i in range(num_pc):
        if c1.iloc[0, i] == 0:
            # nug
            covMatrix[:, :, i] = np.eye(num_stations) * c0.iloc[0, i]
        else:
            # iso nest
            covMatrix[:, :, i] = c0.iloc[0, i] + \
                                 c1.iloc[0, i] * np.exp(-3.0 * stn_dist) / a1.iloc[0, i] + \
                                 c2.iloc[0, i] * np.exp(-3.0 * stn_dist) / a2.iloc[0, i]
    # Simulating residuals
    residuals_pca = np.zeros((num_stations, num_simu, num_pc))
    mu = np.zeros(num_stations)
    for i in range(num_pc):
        residuals_pca[:, :, i] = np.random.multivariate_normal(mu, covMatrix[:, :, i], num_simu).T
    # Interpolating model_coef by periods
    interp_fun = interp1d(model_periods, model_coef, axis = 0)
    model_Tmax = 5.0
    simu_periods = [i for i in periods if i <= model_Tmax]
    simu_coef = interp_fun(simu_periods)
    # Simulating residuals
    num_periods = len(simu_periods)
    residuals = np.empty([num_stations, num_periods, num_simu])
    for i in range(num_simu):
        residuals[:, :, i] = np.matmul(residuals_pca[:, i, :], simu_coef.T)
    # Appending residuals for periods greater than model_Tmax (fixing at 5.0)
    if max(periods) > model_Tmax:
        Tmax_coef = interp_fun(model_Tmax)
        Tmax_residuals = np.empty([num_stations, 1, num_simu])
        for i in range(num_simu):
            Tmax_residuals[:, :, i] = np.matmul(residuals_pca[:, i, :], np.matrix(Tmax_coef).T)
        for tmp_periods in periods:
            if tmp_periods > model_Tmax:
                residuals = np.concatenate((residuals, Tmax_residuals), axis = 1)
    # return
    return residuals
