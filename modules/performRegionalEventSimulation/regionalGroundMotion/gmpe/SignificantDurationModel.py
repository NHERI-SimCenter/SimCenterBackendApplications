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
# Based on the script by Baker Research Group: https://www.jackwbaker.com/GMMs_archive.html

import numpy as np


def abrahamson_silva_ds_1999(
    magnitude=7.0,
    distance=10.0,
    soil=True,  # noqa: FBT002
    duration_type='DS575H',
):
    """Significant duration model by Abrahamson and Silva (1996) Empirical ground motion
    models, report prepared for Brookhaven National Laboratory.
    Input
    magnitude: earthquake magnitude
    distance: site-rupture distance
    soil: True for foil prediction, False for rock prediction
    duration_type: 'DS575H': Ds5-75 (horizontal), 'DS575V': Ds5-75 (vertical)
                   'DS595H': Ds5-95 (horizontal), 'DS595V': Ds5-95 (vertical)
    Output:
    log(ds_median): log(median) significant duration prediction
    ds_sigma: logarithmic standard deviation of the prediction
    """  # noqa: D205, D400
    # map the duration_type to integer key
    dur_map = {'DS575H': 0, 'DS575V': 1, 'DS595H': 2, 'DS595V': 3}
    dur_tag = dur_map.get(duration_type.upper(), None)
    if dur_tag is None:
        print(  # noqa: T201
            "SignificantDurationModel.abrahamson_silva_ds_1999: duration_type='DS575H','DS575V','DS595H','DS595V'?"
        )
        return None, None  # noqa: DOC201, RUF100
    # modeling coefficients
    beta = [3.2, 3.2, 3.2, 3.2]
    b1 = [5.204, 4.610, 5.204, 4.610]
    b2 = [0.851, 1.536, 0.851, 1.536]
    m_star = [6, 6, 6, 6]
    c1 = [0.805, 1.076, 0.805, 1.076]
    c2 = [0.063, 0.107, 0.063, 0.107]
    rc = [10, 10, 10, 10]
    Drat = [0.000, 0.000, 0.845, 0.646]  # noqa: N806
    sigma = [0.55, 0.46, 0.49, 0.45]
    # median
    if distance > rc[dur_tag]:
        ds_median = np.exp(
            np.log(
                (
                    np.exp(b1[dur_tag] + b2[dur_tag] * (magnitude - m_star[dur_tag]))
                    / (10 ** (1.5 * magnitude + 16.05))
                )
                ** (-1 / 3)
                / (4.9e6 * beta[dur_tag])
                + soil * c1[dur_tag]
                + c2[dur_tag] * (distance - rc[dur_tag])
            )
            + Drat[dur_tag]
        )
    else:
        ds_median = np.exp(
            np.log(
                (
                    np.exp(b1[dur_tag] + b2[dur_tag] * (magnitude - m_star[dur_tag]))
                    / (10 ** (1.5 * magnitude + 16.05))
                )
                ** (-1 / 3)
                / (4.9e6 * beta[dur_tag])
                + soil * c1[dur_tag]
            )
            + Drat[dur_tag]
        )
    # sigma
    ds_sigma = sigma[dur_tag]

    # return
    return np.log(ds_median), ds_sigma


def bommer_stafford_alarcon_ds_2009(
    magnitude=7.0,
    distance=10.0,
    vs30=760.0,
    ztor=0.0,
    duration_type='DS575H',
):
    """Significant duration model by Bommer, Stafford, Alarcon (2009) Empirical
    Equations for the Prediction of the Significant, Bracketed, and Uniform
    Duration of Earthquake Ground Motion
    Input
    magnitude: earthquake magnitude
    distance: site-rupture distance
    vs30: average soil shear-wave velocity over the top 30 meters
    ztor: depth to top of rupture (km)
    duration_type: 'DS575H': Ds5-75 (horizontal), 'DS595H': Ds5-95 (horizontal)
    Output:
    log(ds_median): log(median) significant duration prediction
    ds_sigma: logarithmic standard deviation of the prediction
    ds_tau: within-event logarithmic standard deviation
    ds_phi: between-event logarithmic standard deviation
    """  # noqa: D205, D400
    # duration type map
    dur_map = {'DS575H': 0, 'DS595H': 1}
    dur_tag = dur_map.get(duration_type.upper(), None)
    if dur_tag is None:
        print(  # noqa: T201
            "SignificantDurationModel.bommer_stafford_alarcon_ds_2009: duration_type='DS575H','DS595H'?"
        )
        return None, None, None, None  # noqa: DOC201, RUF100

    # modeling coefficients
    c0 = [-5.6298, -2.2393]
    m1 = [1.2619, 0.9368]
    r1 = [2.0063, 1.5686]
    r2 = [-0.2520, -0.1953]
    h1 = [-2.3316, 2.5000]
    v1 = [-0.2900, -0.3478]
    z1 = [-0.0522, -0.0365]
    tauCoeff = [0.3527, 0.3252]  # noqa: N806
    phiCoeff = [0.4304, 0.3460]  # noqa: N806
    sigma_c = [0.1729, 0.1114]  # noqa: F841
    sigma_Tgm = [0.5289, 0.4616]  # noqa: N806

    # median
    ds_median = np.exp(
        c0[dur_tag]
        + m1[dur_tag] * magnitude
        + (r1[dur_tag] + r2[dur_tag] * magnitude)
        * np.log(np.sqrt(distance**2 + h1[dur_tag] ** 2))
        + v1[dur_tag] * np.log(vs30)
        + z1[dur_tag] * ztor
    )
    # standard deviations
    ds_sigma = sigma_Tgm[dur_tag]
    ds_tau = tauCoeff[dur_tag]
    ds_phi = phiCoeff[dur_tag]

    # return
    return np.log(ds_median), ds_sigma, ds_tau, ds_phi


def afshari_stewart_ds_2016(  # noqa: C901
    magnitude=7.0,
    distance=10.0,
    vs30=760.0,
    mechanism='unknown',
    z1=None,
    region='california',
    duration_type='DS575H',
):
    """Significant duration model by Afshari and Stewart (2016) hysically Parameterized
    Prediction Equations for Significant Duration in Active Crustal Regions
    Input
    magnitude: earthquake magnitude
    distance: site-rupture distance
    vs30: average soil shear-wave velocity over the top 30 meters
    mechanism: 'unknown', 'normal', 'reverse', 'strike-slip'
    z1: depth to shear velocity of 1 km/s isosurface (m)
    region: 'california', 'japan', 'other'
    duration_type: 'DS575H': Ds5-75 (horizontal), 'DS595H': Ds5-95 (horizontal), 'DS2080H': Ds20-80 (horizontal)
    Output:
    log(ds_median): log(median) significant duration prediction
    ds_sigma: logarithmic standard deviation of the prediction
    ds_tau: within-event logarithmic standard deviation
    ds_phi: between-event logarithmic standard deviation
    """  # noqa: D205, D400
    # mechanism map
    mech_map = {'unknown': 0, 'normal': 1, 'reverse': 2, 'strike-slip': 3}
    mech_tag = mech_map.get(mechanism.lower(), None)
    if mech_tag is None:
        print(  # noqa: T201
            "SignificantDurationModel.afshari_stewart_ds_2016: mechanism='unknown','normal','reverse','strike-slip'?"
        )
        return None, None, None, None  # noqa: DOC201, RUF100
    # region map
    reg_map = {'california': 0, 'japan': 1, 'other': 2}
    reg_tag = reg_map.get(region.lower(), None)
    if reg_tag is None:
        print(  # noqa: T201
            "SignificantDurationModel.afshari_stewart_ds_2016: region='california', 'japan', 'other'?"
        )
        return None, None, None, None
    # duration type map
    dur_map = {'DS575H': 0, 'DS595H': 1, 'DS2080H': 2}
    dur_tag = dur_map.get(duration_type.upper(), None)
    if dur_tag is None:
        print(  # noqa: T201
            "SignificantDurationModel.afshari_stewart_ds_2016: duration_type='DS575H','DS595H','DS2080H'?"
        )
        return None, None, None, None

    # source coefficients
    M1 = [5.35, 5.20, 5.20]  # noqa: N806
    M2 = [7.15, 7.40, 7.40]  # noqa: N806
    b0 = [
        [1.2800, 2.1820, 0.8822],
        [1.5550, 2.5410, 1.4090],
        [0.7806, 1.6120, 0.7729],
        [1.2790, 2.3020, 0.8804],
    ]
    b1 = [
        [5.576, 3.628, 6.182],
        [4.992, 3.170, 4.778],
        [7.061, 4.536, 6.579],
        [5.578, 3.467, 6.188],
    ]
    b2 = [0.9011, 0.9443, 0.7414]
    b3 = [-1.684, -3.911, -3.164]
    Mstar = [6, 6, 6]  # noqa: N806
    # path coefficients
    c1 = [0.1159, 0.3165, 0.0646]
    RR1 = [10, 10, 10]  # noqa: N806
    RR2 = [50, 50, 50]  # noqa: N806
    c2 = [0.1065, 0.2539, 0.0865]
    c3 = [0.0682, 0.0932, 0.0373]
    # site coefficients
    c4 = [-0.2246, -0.3183, -0.4237]
    Vref = [368.2, 369.9, 369.6]  # noqa: N806
    V1 = [600, 600, 600]  # noqa: N806
    c5 = [0.0006, 0.0006, 0.0005]
    dz1ref = [200, 200, 200]
    # standard deviation coefficients
    phi1 = [0.54, 0.43, 0.56]
    phi2 = [0.41, 0.35, 0.45]
    tau1 = [0.28, 0.25, 0.30]
    tau2 = [0.25, 0.19, 0.19]

    # basin depth
    if reg_tag == 0:
        mu_z1 = np.exp(
            -7.15 / 4 * np.log((vs30**4 + 570.94**4) / (1360**4 + 570.94**4))
        )
    else:
        mu_z1 = np.exp(
            -5.23 / 4 * np.log((vs30**4 + 412.39**4) / (1360**4 + 412.39**4))
        )
    # differential basin depth
    if z1 is None or z1 < 0 or reg_tag == 2:  # noqa: PLR2004
        dz1 = 0
    else:
        dz1 = z1 - mu_z1

    # source term
    if magnitude < M1[dur_tag]:
        F_E = b0[mech_tag][dur_tag]  # noqa: N806
    else:
        if magnitude < M2[dur_tag]:
            deltaSigma = np.exp(  # noqa: N806
                b1[mech_tag][dur_tag] + b2[dur_tag] * (magnitude - Mstar[dur_tag])
            )
        else:
            deltaSigma = np.exp(  # noqa: N806
                b1[mech_tag][dur_tag]
                + b2[dur_tag] * (M2[dur_tag] - Mstar[dur_tag])
                + b3[dur_tag] * (magnitude - M2[dur_tag])
            )

        M_0 = 10 ** (1.5 * magnitude + 16.05)  # noqa: N806
        f_0 = 4.9e6 * 3.2 * (deltaSigma / M_0) ** (1 / 3)
        F_E = 1 / f_0  # noqa: N806
    # path term
    if distance < RR1[dur_tag]:
        F_P = c1[dur_tag] * distance  # noqa: N806
    elif distance < RR2[dur_tag]:
        F_P = c1[dur_tag] * RR1[dur_tag] + c2[dur_tag] * (distance - RR1[dur_tag])  # noqa: N806
    else:
        F_P = (  # noqa: N806
            c1[dur_tag] * RR1[dur_tag]
            + c2[dur_tag] * (RR2[dur_tag] - RR1[dur_tag])
            + c3[dur_tag] * (distance - RR2[dur_tag])
        )
    # F_deltaz term
    if dz1 <= dz1ref[dur_tag]:
        F_deltaz = c5[dur_tag] * dz1  # noqa: N806
    else:
        F_deltaz = c5[dur_tag] * dz1ref[dur_tag]  # noqa: N806
    # site term
    if vs30 < V1[dur_tag]:
        F_S = c4[dur_tag] * np.log(vs30 / Vref[dur_tag]) + F_deltaz  # noqa: N806
    else:
        F_S = c4[dur_tag] * np.log(V1[dur_tag] / Vref[dur_tag]) + F_deltaz  # noqa: N806

    # median
    ds_median = np.exp(np.log(F_E + F_P) + F_S)
    # standard deviations
    # between event
    if magnitude < 5.5:  # noqa: PLR2004
        ds_phi = phi1[dur_tag]
    elif magnitude < 5.75:  # noqa: PLR2004
        ds_phi = phi1[dur_tag] + (phi2[dur_tag] - phi1[dur_tag]) * (
            magnitude - 5.5
        ) / (5.75 - 5.5)
    else:
        ds_phi = phi2[dur_tag]
    # within event
    if magnitude < 6.5:  # noqa: PLR2004
        ds_tau = tau1[dur_tag]
    elif magnitude < 7:  # noqa: PLR2004
        ds_tau = tau1[dur_tag] + (tau2[dur_tag] - tau1[dur_tag]) * (
            magnitude - 6.5
        ) / (7 - 6.5)
    else:
        ds_tau = tau2[dur_tag]
    # total
    ds_sigma = np.sqrt(ds_phi**2 + ds_tau**2)

    # return
    return np.log(ds_median), ds_sigma, ds_tau, ds_phi
