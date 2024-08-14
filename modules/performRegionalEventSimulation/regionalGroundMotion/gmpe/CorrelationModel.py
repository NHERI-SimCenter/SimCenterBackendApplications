#  # noqa: N999, D100
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


def baker_jayaram_correlation_2008(im1, im2, flag_orth=False):  # noqa: FBT002, C901
    """Computing inter-event correlation coeffcieint between Sa of two periods
    Reference:
        Baker and Jayaram (2008) Correlation of Spectral Acceleration
        Values from NGA Ground Motion Models
    Input:
        im1: 1st intensity measure name
        im2: 2nd intensity measure name
        flag_orth: if the correlation coefficient is computed for the two
                   orthogonal components
    Output:
        rho: correlation coefficient
    Note:
        The valid range of T1 and T2 is 0.01s ~ 10.0s
    """  # noqa: D205, D400, D401
    # Parse periods from im1 and im2
    if im1.startswith('SA'):
        T1 = float(im1[3:-1])  # noqa: N806
    elif im1.startswith('PGA'):
        T1 = 0.0  # noqa: N806
    else:
        return 0.0  # noqa: DOC201
    if im2.startswith('SA'):
        T2 = float(im2[3:-1])  # noqa: N806
    elif im2.startswith('PGA'):
        T2 = 0.0  # noqa: N806
    else:
        return 0.0

    # Compute Tmin and Tmax (lower bounds 0.01 for T < 0.01)
    Tmin = max(min([T1, T2]), 0.01)  # noqa: N806
    Tmax = max(max([T1, T2]), 0.01)  # noqa: N806, PLW3301
    # Coefficient C1
    C1 = 1.0 - np.cos(np.pi / 2.0 - 0.366 * np.log(Tmax / max([Tmin, 0.109])))  # noqa: N806
    # Coefficient C2
    if Tmax < 0.2:  # noqa: PLR2004
        C2 = 1.0 - 0.105 * (1.0 - 1.0 / (1.0 + np.exp(100.0 * Tmax - 5.0))) * (  # noqa: N806
            Tmax - Tmin
        ) / (Tmax - 0.0099)
    else:
        C2 = 0.0  # noqa: N806
    # Coefficient C3
    if Tmax < 0.109:  # noqa: PLR2004
        C3 = C2  # noqa: N806
    else:
        C3 = C1  # noqa: N806
    # Coefficient C4
    C4 = C1 + 0.5 * (np.sqrt(C3) - C3) * (1.0 + np.cos(np.pi * Tmin / 0.109))  # noqa: N806
    # rho for a single component
    if Tmax <= 0.109:  # noqa: PLR2004
        rho = C2
    elif Tmin > 0.109:  # noqa: PLR2004
        rho = C1
    elif Tmax < 0.2:  # noqa: PLR2004
        rho = min([C2, C4])
    else:
        rho = C4
    # rho for orthogonal components Coefficient C1
    if flag_orth:
        rho = rho * (0.79 - 0.023 * np.log(np.sqrt(Tmin * Tmax)))  # noqa: PLR6104

    return rho


def bradley_correlation_2011(IM, T=None, flag_Ds=True):  # noqa: FBT002, C901, N803, PLR0911
    """Computing inter-event correlation coeffcieint between Sa(T) and Ds575/D595
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
    """  # noqa: D205, D400, D401
    # PGA
    if IM == 'PGA':  # noqa: RET503
        if flag_Ds:
            return -0.442  # noqa: DOC201
        else:  # noqa: RET505
            return -0.305
    elif IM == 'PGV':
        if flag_Ds:
            return -0.259
        else:  # noqa: RET505
            return -0.211
    elif IM == 'ASI':
        if flag_Ds:
            return -0.411
        else:  # noqa: RET505
            return -0.370
    elif IM == 'SI':
        if flag_Ds:
            return -0.131
        else:  # noqa: RET505
            return -0.079
    elif IM == 'DSI':
        if flag_Ds:
            return 0.074
        else:  # noqa: RET505
            return 0.163
    elif IM == 'CAV':
        if flag_Ds:
            return 0.077
        else:  # noqa: RET505
            return 0.122
    elif IM == 'Ds595':
        if flag_Ds:
            return 0.843
        else:  # noqa: RET505
            return None
    elif IM == 'Sa':
        if flag_Ds:
            if T < 0.09:  # noqa: PLR2004
                a_p = -0.45
                a_c = -0.39
                b_p = 0.01
                b_c = 0.09
            elif T < 0.30:  # noqa: PLR2004
                a_p = -0.39
                a_c = -0.39
                b_p = 0.09
                b_c = 0.30
            elif T < 1.40:  # noqa: PLR2004
                a_p = -0.39
                a_c = -0.06
                b_p = 0.30
                b_c = 1.40
            elif T < 6.50:  # noqa: PLR2004
                a_p = -0.06
                a_c = 0.16
                b_p = 1.40
                b_c = 6.50
            elif T <= 10.0:  # noqa: PLR2004
                a_p = 0.16
                a_c = 0.00
                b_p = 6.50
                b_c = 10.00
        elif T < 0.04:  # noqa: PLR2004
            a_p = -0.41
            a_c = -0.41
            b_p = 0.01
            b_c = 0.04
        elif T < 0.08:  # noqa: PLR2004
            a_p = -0.41
            a_c = -0.38
            b_p = 0.04
            b_c = 0.08
        elif T < 0.26:  # noqa: PLR2004
            a_p = -0.38
            a_c = -0.35
            b_p = 0.08
            b_c = 0.26
        elif T < 1.40:  # noqa: PLR2004
            a_p = -0.35
            a_c = -0.02
            b_p = 0.26
            b_c = 1.40
        elif T <= 6.00:  # noqa: PLR2004
            a_p = -0.02
            a_c = 0.23
            b_p = 1.40
            b_c = 6.00
        elif T <= 10.00:  # noqa: PLR2004
            a_p = 0.23
            a_c = 0.02
            b_p = 6.00
            b_c = 10.0
        rho = a_p + np.log(T / b_p) / np.log(b_c / b_p) * (a_c - a_p)
        return rho  # noqa: RET504


def jayaram_baker_correlation_2009(im, h, flag_clustering=False):  # noqa: FBT002
    """Computing intra-event correlation coeffcieint between Sa(T) at two sites
    Reference:
        Jayaram and Baker (2009) Correlation model for spatially distributed
        ground-motion intensities
    Input:
        im: intensity measure name
        h: distance between the two sites
        flag_clustering: the geologic condition of the soil varies widely over
                         the region (default: false)
    Output:
        rho: correlation between normalized intra-event residuals
    """  # noqa: D205, D400, D401
    # parse period form im
    try:
        # for Sa
        if im.startswith('SA'):
            T = float(im[3:-1])  # noqa: N806
        elif im.startswith('PGA'):
            T = 0.0  # noqa: N806
    except ValueError:
        print(  # noqa: T201
            f'CorrelationModel.jayaram_baker_correlation_2009: error - cannot handle {im}'
        )

    if T >= 1.0:
        b = 22.0 + 3.7 * T
    elif flag_clustering:
        b = 8.5 + 17.2 * T
    else:
        b = 40.7 - 15.0 * T
    rho = np.exp(-3.0 * h / b)
    return rho  # noqa: DOC201, RET504


def load_loth_baker_correlation_2013(datapath):
    """Loading the three matrices in the Loth-Baker correaltion model (2013)
    Reference:
        Loth and Baker (2013) A spatial cross-correlation model of spectral
        accelerations at multiple periods (with the Erratum)
    Input:
        datapath: the path to the files (optional)
    Output:
        B1: short-range coregionalization matrix
        B2: long-range coregionalization matrix
        B3: Nugget effect correlationalization matrix
    """  # noqa: D205, D400, D401
    B2 = pd.read_csv(datapath + 'loth_baker_correlation_2013_B2.csv', header=0)  # noqa: N806
    B1 = pd.read_csv(datapath + 'loth_baker_correlation_2013_B1.csv', header=0)  # noqa: N806
    B3 = pd.read_csv(datapath + 'loth_baker_correlation_2013_B3.csv', header=0)  # noqa: N806
    return B1, B2, B3  # noqa: DOC201


def compute_rho_loth_baker_correlation_2013(T1, T2, h, B1, B2, B3):  # noqa: N803
    """Computing intra-event correlation coeffcieint between Sa(Ti) and Sa(Tj)
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
    """  # noqa: D205, D400, D401
    # Interpolation functions
    f1 = interp2d(B1['Period (s)'], B1['Period (s)'], B1.iloc[:, 1:])
    f2 = interp2d(B2['Period (s)'], B2['Period (s)'], B2.iloc[:, 1:])
    f3 = interp2d(B3['Period (s)'], B3['Period (s)'], B3.iloc[:, 1:])
    # Three coefficients (T1, T2 < 0.01 would be given the boundary value)
    b1 = f1(T1, T2)
    b2 = f2(T1, T2)
    b3 = f3(T1, T2)
    # Covariance functions
    Ch = b1 * np.exp(-3.0 * h / 20.0) + b2 * np.exp(-3.0 * h / 70.0) + b3 * (h == 0)  # noqa: N806
    # Correlation coefficient
    rho = Ch
    return rho  # noqa: DOC201, RET504


def loth_baker_correlation_2013(stations, im_name_list, num_simu):  # noqa: C901
    """Simulating intra-event residuals
    Reference:
        Loth and Baker (2013) A spatial cross-correlation model of spectral
        accelerations at multiple periods (with the Erratum)
    Input:
        stations: stations coordinates
        im_name_list: simulated intensity measure names
        num_simu: number of realizations
    Output:
        residuals: intra-event residuals
    Note:
        The valid range for T1 and T2 is 0.01s ~ 10.0s
    """  # noqa: D205, D400, D401
    # Parse periods from intensity measure list
    periods = []
    for cur_im in im_name_list:
        try:
            if cur_im.startswith('SA'):
                periods.append(float(cur_im[3:-1]))
            elif cur_im.startswith('PGA'):
                periods.append(0.0)
        except ValueError:  # noqa: PERF203
            print(  # noqa: T201
                f'CorrelationModel.loth_baker_correlation_2013: error - cannot handle {cur_im}'
            )
    # Loading modeling coefficients
    B1, B2, B3 = load_loth_baker_correlation_2013(  # noqa: N806
        os.path.dirname(__file__) + '/data/'  # noqa: PTH120
    )
    # Computing distance matrix
    num_stations = len(stations)
    stn_dist = np.zeros((num_stations, num_stations))
    for i in range(num_stations):
        loc_i = np.array([stations[i]['Latitude'], stations[i]['Longitude']])
        for j in range(num_stations):
            loc_j = np.array([stations[j]['Latitude'], stations[j]['Longitude']])
            stn_dist[i, j] = get_distance_from_lat_lon(loc_i, loc_j)
    # Creating a covariance matrices for each of the principal components
    num_periods = len(periods)
    covMatrix = np.zeros((num_stations * num_periods, num_stations * num_periods))  # noqa: N806
    for i in range(num_periods):
        for j in range(num_periods):
            covMatrix[
                num_stations * i : num_stations * (i + 1),
                num_stations * j : num_stations * (j + 1),
            ] = compute_rho_loth_baker_correlation_2013(
                periods[i], periods[j], stn_dist, B1, B2, B3
            )

    mu = np.zeros(num_stations * num_periods)
    residuals_raw = np.random.multivariate_normal(mu, covMatrix, num_simu)
    # reorder residual_raw [[period1],[period2],...,[]]-->[[site1],[site2],...,[]]
    residuals_reorder = []
    for i in range(num_simu):
        tmp = []
        for j in range(num_stations):
            for k in range(num_periods):
                tmp.append(residuals_raw[i, j + k * num_stations])  # noqa: PERF401
        residuals_reorder.append(tmp)
    residuals_reorder = np.array(residuals_reorder)
    residuals = (
        residuals_reorder.reshape(num_simu, num_stations, num_periods)
        .swapaxes(0, 1)
        .swapaxes(1, 2)
    )
    # return
    return residuals  # noqa: DOC201, RET504


def load_markhvida_ceferino_baker_correlation_2017(datapath):
    """Loading the three matrices in the Markhivida et al. correaltion model (2017)
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
    """  # noqa: D205, D400, D401
    MCB_model = pd.read_csv(  # noqa: N806
        datapath + 'markhvida_ceferino_baker_correlation_2017_model_coeff.csv',
        index_col=None,
        header=0,
    )
    MCB_pca = pd.read_csv(  # noqa: N806
        datapath + 'markhvida_ceferino_baker_correlation_2017_pca_coeff.csv',
        index_col=None,
        header=0,
    )
    MCB_var = pd.read_csv(  # noqa: N806
        datapath + 'markhvida_ceferino_baker_correlation_2017_var_scale.csv',
        index_col=None,
        header=0,
    )
    return MCB_model, MCB_pca, MCB_var  # noqa: DOC201


def markhvida_ceferino_baker_correlation_2017(  # noqa: C901
    stations,
    im_name_list,
    num_simu,
    num_pc=19,
):
    """Simulating intra-event residuals
    Reference:
        Markhvida et al. (2017) Modeling spatially correlated spectral
        accelerations at multiple periods using principal component analysis
        and geostatistics
    Input:
        stations: stations coordinates
        im_name_list: simulated intensity measure names
        num_simu: number of realizations
        num_pc: number of principle components
    Output:
        residuals: intra-event residuals
    Note:
        The valid range for T1 and T2 is 0.01s ~ 5.0s
    """  # noqa: D205, D400, D401
    # Parse periods from intensity measure list
    periods = []
    for cur_im in im_name_list:
        try:
            if cur_im.startswith('SA'):
                periods.append(float(cur_im[3:-1]))
            elif cur_im.startswith('PGA'):
                periods.append(0.0)
            else:
                raise ValueError(  # noqa: DOC501, TRY003, TRY301
                    f'CorrelationModel Markhvida et al. (2017): error - cannot handle {cur_im}'  # noqa: EM102
                )
        except ValueError:  # noqa: PERF203
            print(  # noqa: T201
                f'CorrelationModel.loth_baker_correlation_2013: error - cannot handle {cur_im}'
            )
    # Loading factors
    MCB_model, MCB_pca, MCB_var = load_markhvida_ceferino_baker_correlation_2017(  # noqa: N806
        os.path.dirname(__file__) + '/data/'  # noqa: PTH120
    )
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
    model_coef = MCB_pca.iloc[:, 1 : num_pc + 1]
    # Computing distance matrix
    num_stations = len(stations)
    stn_dist = np.zeros((num_stations, num_stations))
    for i in range(num_stations):
        loc_i = np.array([stations[i]['lat'], stations[i]['lon']])
        for j in range(num_stations):
            loc_j = np.array([stations[j]['lat'], stations[j]['lon']])
            stn_dist[i, j] = get_distance_from_lat_lon(loc_i, loc_j)
    # Scaling variance if less than 19 principal components are used
    c0 = c0 / MCB_var.iloc[0, num_pc - 1]  # noqa: PLR6104
    c1 = c1 / MCB_var.iloc[0, num_pc - 1]  # noqa: PLR6104
    c2 = c2 / MCB_var.iloc[0, num_pc - 1]  # noqa: PLR6104
    # Creating a covariance matrices for each of the principal components
    covMatrix = np.zeros((num_stations, num_stations, num_pc))  # noqa: N806
    for i in range(num_pc):
        if c1.iloc[0, i] == 0:
            # nug
            covMatrix[:, :, i] = np.eye(num_stations) * c0.iloc[0, i]
        else:
            # iso nest
            covMatrix[:, :, i] = (
                c0.iloc[0, i] * (stn_dist == 0)
                + c1.iloc[0, i] * np.exp(-3.0 * stn_dist / a1.iloc[0, i])
                + c2.iloc[0, i] * np.exp(-3.0 * stn_dist / a2.iloc[0, i])
            )
    # Simulating residuals
    residuals_pca = np.zeros((num_stations, num_simu, num_pc))
    mu = np.zeros(num_stations)
    for i in range(num_pc):
        residuals_pca[:, :, i] = np.random.multivariate_normal(
            mu, covMatrix[:, :, i], num_simu
        ).T
    # Interpolating model_coef by periods
    interp_fun = interp1d(model_periods, model_coef, axis=0)
    model_Tmax = 5.0  # noqa: N806
    simu_periods = [i for i in periods if i <= model_Tmax]
    if (len(simu_periods) == 1) and (simu_periods[0] == 0):
        # for PGA only (using 0.01 sec as the approxiamate)
        simu_coef = model_coef.iloc[0, :]
    else:
        simu_periods = [0.01 if x == 0 else x for x in simu_periods]
        simu_coef = interp_fun(simu_periods)
    # Simulating residuals
    num_periods = len(simu_periods)
    residuals = np.empty([num_stations, num_periods, num_simu])
    for i in range(num_simu):
        residuals[:, :, i] = np.reshape(
            np.matmul(residuals_pca[:, i, :], simu_coef.T), residuals[:, :, i].shape
        )
    # Appending residuals for periods greater than model_Tmax (fixing at 5.0)
    if max(periods) > model_Tmax:
        Tmax_coef = interp_fun(model_Tmax)  # noqa: N806
        Tmax_residuals = np.empty([num_stations, 1, num_simu])  # noqa: N806
        for i in range(num_simu):
            Tmax_residuals[:, :, i] = np.matmul(
                residuals_pca[:, i, :], np.matrix(Tmax_coef).T
            )
        for tmp_periods in periods:
            if tmp_periods > model_Tmax:
                residuals = np.concatenate((residuals, Tmax_residuals), axis=1)
    # return
    return residuals  # noqa: DOC201


def load_du_ning_correlation_2021(datapath):
    """Loading the three matrices in the Du and Ning correlation model (2021)
    Reference:
        Du and Ning (2021) Modeling spatial cross-correlation of multiple
        ground motion intensity measures (SAs, PGA, PGV, Ia, CAV, and significant
        durations) based on principal component and geostatistical analyses
    Input:
        datapath: the path to the files (optional)
    Output:
        DN_model: model coeff.
        DN_pca: pca coeff.
        DN_var: var of pca
    """  # noqa: D205, D400, D401
    DN_model = pd.read_csv(  # noqa: N806
        datapath + 'du_ning_correlation_2021_model_coeff.csv',
        index_col=None,
        header=0,
    )
    DN_pca = pd.read_csv(  # noqa: N806
        datapath + 'du_ning_correlation_2021_pca_coeff.csv', index_col=None, header=0
    )
    DN_var = pd.read_csv(  # noqa: N806
        datapath + 'du_ning_correlation_2021_var_scale.csv', index_col=None, header=0
    )
    return DN_model, DN_pca, DN_var  # noqa: DOC201


def du_ning_correlation_2021(stations, im_name_list, num_simu, num_pc=23):
    """Simulating intra-event residuals
    Reference:
        Du and Ning (2021) Modeling spatial cross-correlation of multiple
        ground motion intensity measures (SAs, PGA, PGV, Ia, CAV, and significant
        durations) based on principal component and geostatistical analyses
    Input:
        stations: stations coordinates
        im_name_list: simulated intensity measure names
        num_simu: number of realizations
        num_pc: number of principle components
    Output:
        residuals: intra-event residuals
    Note:
        The valid range for T1 and T2 is 0.01s ~ 5.0s
    """  # noqa: D205, D400, D401
    # Parse periods_ims from intensity measure list
    periods_ims = []
    for cur_im in im_name_list:
        if cur_im.startswith('SA'):
            periods_ims.append(float(cur_im[3:-1]))
        else:
            periods_ims.append(cur_im)
    # Loading factors
    DN_model, DN_pca, DN_var = load_du_ning_correlation_2021(  # noqa: N806
        os.path.dirname(__file__) + '/data/'  # noqa: PTH120
    )
    c1 = DN_model.loc[DN_model['Type'] == 'c1']
    c1 = c1[c1.keys()[1:]]
    a1 = DN_model.loc[DN_model['Type'] == 'a1']
    a1 = a1[a1.keys()[1:]]
    b1 = DN_model.loc[DN_model['Type'] == 'b1']
    b1 = b1[b1.keys()[1:]]
    a2 = DN_model.loc[DN_model['Type'] == 'a2']
    a2 = a2[a2.keys()[1:]]
    b2 = DN_model.loc[DN_model['Type'] == 'b2']
    b2 = b2[b2.keys()[1:]]
    # model_periods is pseudo periods and PGA, PGV, Ia, CAV, DS575H, DS595H
    model_periods = DN_pca['Period&IM']
    model_ims_list = ['PGA', 'PGV', 'Ia', 'CAV', 'DS575H', 'DS595H']
    ims_map = {'PGA': 11, 'PGV': 12, 'Ia': 13, 'CAV': 14, 'DS575H': 15, 'DS595H': 16}
    # convert periods to float
    model_periods = [float(x) for x in model_periods if x not in model_ims_list] + [
        x for x in model_periods if x in model_ims_list
    ]
    model_coef = DN_pca.iloc[:, 1 : num_pc + 1]
    # Computing distance matrix
    num_stations = len(stations)
    stn_dist = np.zeros((num_stations, num_stations))
    for i in range(num_stations):
        loc_i = np.array([stations[i]['lat'], stations[i]['lon']])
        for j in range(num_stations):
            loc_j = np.array([stations[j]['lat'], stations[j]['lon']])
            stn_dist[i, j] = get_distance_from_lat_lon(loc_i, loc_j)
    # Scaling variance if less than 23 principal components are used
    c1 = c1 / DN_var.iloc[0, num_pc - 1]  # noqa: PLR6104
    a1 = a1 / DN_var.iloc[0, num_pc - 1]  # noqa: PLR6104
    a2 = a2 / DN_var.iloc[0, num_pc - 1]  # noqa: PLR6104
    # Creating a covariance matrices for each of the principal components
    covMatrix = np.zeros((num_stations, num_stations, num_pc))  # noqa: N806
    for i in range(num_pc):
        if a1.iloc[0, i] == 0:
            # nug
            covMatrix[:, :, i] = np.eye(num_stations) * c1.iloc[0, i]
        else:
            # iso nest
            covMatrix[:, :, i] = (
                c1.iloc[0, i] * (stn_dist == 0)
                + a1.iloc[0, i] * np.exp(-3.0 * stn_dist / b1.iloc[0, i])
                + a2.iloc[0, i] * np.exp(-3.0 * stn_dist / b2.iloc[0, i])
            )
    # Simulating residuals
    residuals_pca = np.zeros((num_stations, num_simu, num_pc))
    mu = np.zeros(num_stations)
    for i in range(num_pc):
        residuals_pca[:, :, i] = np.random.multivariate_normal(
            mu, covMatrix[:, :, i], num_simu
        ).T
    # Interpolating model_coef by periods
    pseudo_periods = [x for x in model_periods if type(x) == float] + [  # noqa: E721
        ims_map[x]
        for x in model_periods
        if type(x) == str  # noqa: E721
    ]
    interp_fun = interp1d(pseudo_periods, model_coef, axis=0)
    model_Tmax = 10.0  # noqa: N806
    simu_periods = [min(i, model_Tmax) for i in periods_ims if type(i) == float] + [  # noqa: E721
        ims_map[i]
        for i in periods_ims
        if type(i) == str  # noqa: E721
    ]
    if (len(simu_periods) == 1) and (simu_periods[0] == 0):
        # for PGA only (using 0.01 sec as the approximate)
        simu_coef = model_coef.iloc[0, :]
    else:
        simu_periods = [0.01 if x == 0 else x for x in simu_periods]
        simu_coef = interp_fun(simu_periods)
    # Simulating residuals
    num_periods = len(simu_periods)
    residuals = np.empty([num_stations, num_periods, num_simu])
    for i in range(num_simu):
        residuals[:, :, i] = np.reshape(
            np.matmul(residuals_pca[:, i, :], simu_coef.T), residuals[:, :, i].shape
        )

    # return
    return residuals  # noqa: DOC201


def baker_bradley_correlation_2017(im1=None, im2=None):  # noqa: C901
    """Correlation between Sa and other IMs
    Baker, J. W., and Bradley, B. A. (2017). “Intensity measure correlations observed in
    the NGA-West2 database, and dependence of correlations on rupture and site parameters.”
    Based on the script: https://github.com/bakerjw/NGAW2_correlations/blob/master/corrPredictions.m
    Input:
        T: period of Sa
        im1: 1st intensity measure name
        im2: 2nd intensity measure name
    Output:
        rho: correlation coefficient
    """  # noqa: D205, D400
    # im map:
    im_map = {'DS575H': 0, 'DS595H': 1, 'PGA': 2, 'PGV': 3}

    period_list = []
    im_list = []
    if im1.startswith('SA'):
        im_list.append('SA')
        period_list.append(float(im1[3:-1]))
    else:
        tmp_tag = im_map.get(im1.upper(), None)
        if tmp_tag is None:
            print(  # noqa: T201
                f'CorrelationModel.baker_bradley_correlation_2017: warning - return 0.0 for unknown {im1}'
            )
            return 0.0  # noqa: DOC201
        im_list.append(tmp_tag)
        period_list.append(None)
    if im2.startswith('SA'):
        im_list.append('SA')
        period_list.append(float(im2[3:-1]))
    else:
        tmp_tag = im_map.get(im2.upper(), None)
        if tmp_tag is None:
            print(  # noqa: T201
                f'CorrelationModel.baker_bradley_correlation_2017: warning - return 0.0 for unknown {im2}'
            )
            return 0.0
        im_list.append(tmp_tag)

    if im1.startswith('SA') and im2.startswith('SA'):
        # two Sa intensities
        return baker_jayaram_correlation_2008(im1, im2)

    if 'SA' not in im_list:
        # two non-Sa intensities
        # rho matrix
        rho_mat = [
            [1.000, 0.843, -0.442, -0.259],
            [0.843, 1.000, -0.405, -0.211],
            [-0.442, -0.405, 1.000, 0.733],
            [-0.259, -0.211, 0.733, 1.000],
        ]
        # return
        return rho_mat[im_list[0]][im_list[1]]

    # one Sa + one non-Sa
    im_list.remove('SA')
    im_tag = im_list[0]
    T = [x for x in period_list if x is not None][0]  # noqa: N806, RUF015
    # modeling coefficients
    a = [
        [0.00, -0.45, -0.39, -0.39, -0.06, 0.16],
        [0.00, -0.41, -0.41, -0.38, -0.35, 0.02, 0.23],
        [1.00, 0.97],
        [0.73, 0.54, 0.80, 0.76],
    ]
    b = [
        [0.00, -0.39, -0.39, -0.06, 0.16, 0.00],
        [0.00, -0.41, -0.38, -0.35, -0.02, 0.23, 0.02],
        [0.895, 0.25],
        [0.54, 0.81, 0.76, 0.70],
    ]
    c = [[], [], [0.06, 0.80], [0.045, 0.28, 1.10, 5.00]]
    d = [[], [], [1.6, 0.8], [1.8, 1.5, 3.0, 3.2]]
    e = [
        [0.01, 0.09, 0.30, 1.40, 6.50, 10.00],
        [0.01, 0.04, 0.08, 0.26, 1.40, 6.00, 10.00],
        [0.20, 10.00],
        [0.10, 0.75, 2.50, 10.00],
    ]

    # rho
    if im_tag < 2:  # noqa: PLR2004
        for j in range(1, len(e[im_tag])):
            if e[im_tag][j] >= T:
                rho = a[im_tag][j] + (b[im_tag][j] - a[im_tag][j]) / np.log(
                    e[im_tag][j] / e[im_tag][j - 1]
                ) * np.log(T / e[im_tag][j - 1])
                break
    else:
        for j in range(len(e[im_tag])):
            if e[im_tag][j] >= T:
                rho = (a[im_tag][j] + b[im_tag][j]) / 2 - (
                    a[im_tag][j] - b[im_tag][j]
                ) / 2 * np.tanh(d[im_tag][j] * np.log(T / c[im_tag][j]))
                break

    # return
    return rho


def get_distance_from_lat_lon(site_loc1, site_loc2):  # noqa: D103
    # earth radius (km)
    earth_radius_avg = 6371.0
    # site lat and lon
    lat1, lon1 = site_loc1
    lat2, lon2 = site_loc2
    # convert to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    # calculate haversine
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    dist = (
        2.0
        * earth_radius_avg
        * np.arcsin(
            np.sqrt(
                np.sin(0.5 * dlat) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * dlon) ** 2
            )
        )
    )
    # return
    return dist  # noqa: RET504
