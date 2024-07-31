#  # noqa: INP001, D100
# Copyright (c) 2021 Leland Stanford Junior University
# Copyright (c) 2021 The Regents of the University of California
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
# This module is used to select a set of ground motions such with uniformly distributed user-defined
# intensity measures (IM)
#
# Contributors:
# Kuanshi Zhong
# Sang-ri Yi
# Frank Mckenna
#


# TODO recommended ranges???  # noqa: FIX002, TD002, TD003, TD004


# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# from matplotlib.colors import LinearSegmentedColormap
import json
import os
import sys

import numpy as np
import plotly.express as px
from scipy.spatial import distance_matrix
from scipy.stats import gmean, qmc


def main(inputArgs, err):  # noqa: ANN001, ANN201, N803, D103
    gms = gmCluster(inputArgs, err)  # noqa: F841


class gmCluster:  # noqa: N801, D101
    def __init__(self, inputArgs, err):  # noqa: ANN001, ANN204, ARG002, C901, N803, D107, PLR0912, PLR0915
        np.random.seed(seed=42)  # noqa: NPY002
        curDir = os.path.dirname(__file__)  # noqa: PTH120, N806
        gmDataBaseDir = os.path.join(curDir, 'gmdata.json')  # noqa: PTH118, N806
        inputJsonPath = inputArgs[1]  # noqa: N806

        with open(inputJsonPath) as fj:  # noqa: PTH123
            inputJson = json.load(fj)  # noqa: N806

        nim = len(inputJson['IM'])

        im_ub = np.zeros((nim,))
        im_lb = np.zeros((nim,))
        im_nbins = np.zeros((nim,))
        im_names = []
        im_periods = []
        i = 0
        for imName, value in inputJson['IM'].items():  # noqa: N806
            im_names += [imName]
            im_ub[i] = float(value['upperBound'])
            im_lb[i] = float(value['lowerBound'])
            im_nbins[i] = int(value['numBins'])
            im_periods += [value['Periods']]

            if not (im_ub[i] > im_lb[i]):
                msg = (
                    'error parsing IMs: lowerbound of '
                    + imName
                    + ' should be smaller than upperbound'
                )
                print(msg)  # noqa: T201
                print(im_ub[i])  # noqa: T201
                print(im_lb[i])  # noqa: T201
                errf.write(msg)
                errf.close()
                exit(-1)  # noqa: PLR1722

            i += 1  # noqa: SIM113

        npergrid = int(inputJson['numSampPerBin'])

        # TODO: Convert the units... Or fix the units......  # noqa: FIX002, TD002, TD003

        # nim = len(im_names)
        ngrid = np.prod(im_nbins)
        #
        # Clustring parameters
        #
        numEQmax = int(  # noqa: N806
            max(1, round(ngrid / 10))
        )  # Maximum number of records from the single earthquake
        # numEQmax = 1

        #
        # Get grid of IMs - change everything to log-space
        #
        log_im_ub = np.log(im_ub)
        log_im_lb = np.log(im_lb)
        log_im_range = log_im_ub - log_im_lb
        # X is log-IM

        id_im_scaling_ancher = -1
        found_scaling_anchor = False
        nim_eff = nim

        ## For the scaling anchor, we prioritize PSA and PGA
        for ni in range(len(im_names)):
            if im_names[ni].startswith('PSA') or im_names[ni].startswith('PGA'):  # noqa: SIM102
                # scaling anchor
                if not found_scaling_anchor:
                    id_im_scaling_ancher = (
                        ni  # TODO  # noqa: FIX002, TD002, TD003, TD004
                    )
                    found_scaling_anchor = True
                    nim_eff = nim - 1

        ## Only if we didn't find PSA or PGA, we consider PGV, PGD, Ia as scaling anchor
        if not found_scaling_anchor:
            for ni in range(len(im_names)):
                if im_names[ni].startswith('PG') or im_names[ni].startswith('Ia'):  # noqa: SIM102
                    if not found_scaling_anchor:
                        id_im_scaling_ancher = (
                            ni  # TODO  # noqa: FIX002, TD002, TD003, TD004
                        )
                        found_scaling_anchor = True
                        nim_eff = nim - 1

        if nim <= 0:
            # ERROR
            msg = 'number of IMs should be greater than 1'
            print(msg)  # noqa: T201
            errf.write(msg)
            errf.close()
            exit(-1)  # noqa: PLR1722

        elif nim_eff == 0:
            # One variable we have is the scaling anchor
            myID = [1]  # noqa: N806
            Scaling_ref = np.linspace(log_im_lb[0], log_im_ub[0], int(im_nbins[0]))  # noqa: N806
            IM_log_ref = np.zeros(0)  # dummy  # noqa: N806
            isGrid = True  # noqa: N806

        elif nim_eff == 1:
            if found_scaling_anchor:
                if found_scaling_anchor:
                    myID = np.delete([0, 1], id_im_scaling_ancher)  # noqa: N806
                    Scaling_ref = np.linspace(  # noqa: N806
                        log_im_lb[id_im_scaling_ancher],
                        log_im_ub[id_im_scaling_ancher],
                        int(im_nbins[id_im_scaling_ancher]),
                    )
                else:
                    myID = [0]  # noqa: N806
                X = np.linspace(  # noqa: N806
                    log_im_lb[myID[0]], log_im_ub[myID[0]], int(im_nbins[myID[0]])
                )
                IM_log_ref = X[np.newaxis].T  # noqa: N806
                isGrid = True  # noqa: N806

        elif nim_eff == 2:  # noqa: PLR2004
            if found_scaling_anchor:
                myID = np.delete([0, 1, 2], id_im_scaling_ancher)  # noqa: N806
                Scaling_ref = np.linspace(  # noqa: N806
                    log_im_lb[id_im_scaling_ancher],
                    log_im_ub[id_im_scaling_ancher],
                    int(im_nbins[id_im_scaling_ancher]),
                )
            else:
                myID = [0, 1]  # noqa: N806

            X, Y = np.meshgrid(  # noqa: N806
                np.linspace(
                    log_im_lb[myID[0]], log_im_ub[myID[0]], int(im_nbins[myID[0]])
                ),
                np.linspace(
                    log_im_lb[myID[1]], log_im_ub[myID[1]], int(im_nbins[myID[1]])
                ),
            )
            IM_log_ref = np.vstack([X.reshape(-1), Y.reshape(-1)]).T  # noqa: N806
            isGrid = True  # noqa: N806
        elif nim_eff == 3:  # noqa: PLR2004
            if found_scaling_anchor:
                myID = np.delete([0, 1, 2, 3], id_im_scaling_ancher)  # noqa: N806
                Scaling_ref = np.linspace(  # noqa: N806
                    log_im_lb[id_im_scaling_ancher],
                    log_im_ub[id_im_scaling_ancher],
                    int(im_nbins[id_im_scaling_ancher]),
                )
            else:
                myID = [0, 1, 2]  # noqa: N806

            X, Y, Z = np.meshgrid(  # noqa: N806
                np.linspace(
                    log_im_lb[myID[0]], log_im_ub[myID[0]], int(im_nbins[myID[0]])
                ),
                np.linspace(
                    log_im_lb[myID[1]], log_im_ub[myID[1]], int(im_nbins[myID[1]])
                ),
                np.linspace(
                    log_im_lb[myID[2]], log_im_ub[myID[2]], int(im_nbins[myID[2]])
                ),
            )
            IM_log_ref = np.vstack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).T  # noqa: N806
            isGrid = True  # noqa: N806
        else:
            if found_scaling_anchor:
                myID = np.delete(range(nim_eff + 1), id_im_scaling_ancher)  # noqa: N806
                Scaling_ref = np.linspace(  # noqa: N806
                    log_im_lb[id_im_scaling_ancher],
                    log_im_ub[id_im_scaling_ancher],
                    int(im_nbins[id_im_scaling_ancher]),
                )
            else:
                myID = range(nim_eff)  # noqa: N806

            # Let us do LHS sampling
            sampler = qmc.LatinHypercube(d=nim)
            U = sampler.random(n=ngrid)  # noqa: N806
            X = np.zeros((ngrid, nim_eff))  # noqa: N806
            for i in range(nim_eff):
                X[:, i] = (
                    U[:, i] * (log_im_ub[myID[i]] - log_im_lb[myID[i]])
                    + log_im_lb[myID[i]]
                )
            IM_log_ref = X  # noqa: N806
            isGrid = False  # noqa: N806

        #
        # Read Database
        #
        with open(gmDataBaseDir) as fd:  # noqa: PTH123
            gmData = json.load(fd)  # noqa: N806

        RSN = gmData['RSN']  # noqa: N806
        geomPSA = gmData['geomPSA']  # noqa: N806
        geomPGA = gmData['geomPGA']  # noqa: N806
        geomPGV = gmData['geomPGV']  # noqa: N806
        geomPGD = gmData['geomPGD']  # noqa: N806
        geomDS575 = gmData['geomDS575']  # noqa: N806
        geomDS595 = gmData['geomDS595']  # noqa: N806
        geomIa = gmData['geomIa']  # noqa: N806

        periods = gmData['period']
        numgm = gmData['numgm']
        eqnameID = gmData['eqnameID']  # noqa: N806
        units = gmData['unit']

        #
        # Define Sa(T_cond)
        #

        # compute scaling factors

        #
        # Compute SaRatio(T_lowbound,T_cond,T_highbound) and Ds575
        #

        IM_log_data_pool = np.zeros((numgm, 0))  # noqa: N806
        scaling_exponent = np.zeros((nim,))
        myunits = []
        for ni in range(nim):
            if im_names[ni].startswith('PSA'):
                Sa_T1 = np.zeros((numgm,))  # noqa: N806
                T_cond = float(im_periods[ni][0])  # central (<= 5.0)  # noqa: N806
                for ng in range(numgm):
                    Sa_T1[ng] = np.interp(T_cond, periods, geomPSA[ng])

                Sa1_pool = Sa_T1[np.newaxis].T  # noqa: N806
                IM_log_data_pool = np.hstack([IM_log_data_pool, np.log(Sa1_pool)])  # noqa: N806
                scaling_exponent[ni] = 1

                myunits += ['(' + units['PSA'] + ')']
            elif im_names[ni] == 'SaRatio':
                Sa_T1 = np.zeros((numgm,))  # noqa: N806
                Sa_T_geomean = np.zeros((numgm,))  # noqa: N806

                T_lowbound = float(im_periods[ni][0])  # low-bound  # noqa: N806
                T_cond = float(im_periods[ni][1])  # central (<= 5.0)  # noqa: N806
                T_highbound = float(im_periods[ni][2])  # high-bound  # noqa: N806

                idx_T_range = np.where(  # noqa: N806
                    (np.array(periods) > T_lowbound)
                    * (np.array(periods) < T_highbound)
                )[0]

                for ng in range(numgm):
                    Sa_T1[ng] = np.interp(T_cond, periods, geomPSA[ng])
                    Sa_T_geomean[ng] = gmean(
                        np.array(geomPSA[ng])[idx_T_range.astype(int)]
                    )

                SaRatio_pool = (Sa_T1 / Sa_T_geomean)[np.newaxis].T  # noqa: N806
                IM_log_data_pool = np.hstack(  # noqa: N806
                    [IM_log_data_pool, np.log(SaRatio_pool)]
                )
                scaling_exponent[ni] = 0

                myunits += ['']
            elif im_names[ni] == 'DS575':
                ds_pool = (np.array(geomDS575))[np.newaxis].T
                IM_log_data_pool = np.hstack([IM_log_data_pool, np.log(ds_pool)])  # noqa: N806
                scaling_exponent[ni] = 0
                myunits += ['(' + units['DS575'] + ')']

            elif im_names[ni] == 'DS595':
                ds_pool = (np.array(geomDS595))[np.newaxis].T
                IM_log_data_pool = np.hstack([IM_log_data_pool, np.log(ds_pool)])  # noqa: N806
                scaling_exponent[ni] = 0
                myunits += ['(' + units['DS595'] + ')']

            elif im_names[ni] == 'PGA':
                pg_pool = (np.array(geomPGA))[np.newaxis].T
                IM_log_data_pool = np.hstack([IM_log_data_pool, np.log(pg_pool)])  # noqa: N806
                scaling_exponent[ni] = 1
                myunits += ['(' + units['PGA'] + ')']

            elif im_names[ni] == 'PGV':
                pg_pool = (np.array(geomPGV))[np.newaxis].T
                IM_log_data_pool = np.hstack([IM_log_data_pool, np.log(pg_pool)])  # noqa: N806
                scaling_exponent[ni] = 1
                myunits += ['(' + units['PGV'] + ')']

            elif im_names[ni] == 'PGD':
                pg_pool = (np.array(geomPGD))[np.newaxis].T
                IM_log_data_pool = np.hstack([IM_log_data_pool, np.log(pg_pool)])  # noqa: N806
                scaling_exponent[ni] = 1
                myunits += ['(' + units['PGD'] + ')']

            elif im_names[ni] == 'Ia':
                ai_pool = (np.array(geomIa))[np.newaxis].T
                IM_log_data_pool = np.hstack([IM_log_data_pool, np.log(ai_pool)])  # noqa: N806
                scaling_exponent[ni] = 2
                myunits += ['(' + units['Ia'] + ')']
            else:
                msg = 'unrecognized IM name ' + im_names[ni]
                print(msg)  # noqa: T201
                errf.write(msg)
                errf.close()
                exit(-1)  # noqa: PLR1722

        if found_scaling_anchor:
            IM_log_data_scaling_anchor = IM_log_data_pool[:, id_im_scaling_ancher]  # noqa: N806
            # IM_log_ref_scaling_anchor = IM_log_ref[:,id_im_scaling_ancher]
            IM_log_ref_scaling_anchor = Scaling_ref  # noqa: N806

            IM_log_data_pool2 = np.delete(  # noqa: N806
                IM_log_data_pool.copy(), id_im_scaling_ancher, 1
            )
            IM_log_ref2 = IM_log_ref.copy()  # noqa: N806

            scaling_exponent = (
                scaling_exponent / scaling_exponent[id_im_scaling_ancher]
            )
            scaling_exponent2 = np.delete(
                scaling_exponent.copy(), id_im_scaling_ancher
            )
            log_im_range2 = np.delete(log_im_range.copy(), id_im_scaling_ancher)
            lenRef2 = np.mean(1 / np.delete(im_nbins.copy(), id_im_scaling_ancher))  # noqa: N806
        else:
            IM_log_data_pool2 = IM_log_data_pool  # noqa: N806
            IM_log_ref2 = IM_log_ref  # noqa: N806
            scaling_exponent2 = scaling_exponent
            log_im_range2 = log_im_range
            lenRef2 = np.linalg.norm(1 / im_nbins)  # noqa: N806

        if id_im_scaling_ancher >= 0:
            if isGrid:
                nScalingGrid = im_nbins[id_im_scaling_ancher]  # noqa: N806
                nGridPerIM = ngrid / im_nbins[id_im_scaling_ancher]  # noqa: N806
            else:
                nScalingGrid = ngrid  # noqa: N806
                nGridPerIM = ngrid / im_nbins[id_im_scaling_ancher]  # noqa: N806
        else:
            nScalingGrid = 1  # noqa: N806
            nGridPerIM = ngrid  # noqa: N806

        sf_min = 0.5  # minimum of no-panalty scaling
        sf_max = 10.0  # maximum of no-pad nalty scaling
        sf_penalty = 10.0  # per unit outside the tolerance range(sf_min~sf_max)

        # selected_gm_ID_list =[]
        # selected_gm_err_list =[]
        # selected_gm_eqID_list =[]
        # selected_gm_scale_list =[]
        selected_gm_ID = []  # noqa: N806
        selected_gm_err = []
        selected_gm_eqID = []  # noqa: N806
        selected_gm_scale = []

        err_sum = np.zeros((int(nScalingGrid), int(nGridPerIM)))

        nsa_tmp, ngr_tmp = np.meshgrid(
            range(int(nScalingGrid)), range(int(nGridPerIM))
        )
        nsas = list(nsa_tmp.reshape(-1)) * npergrid
        ngrs = list(ngr_tmp.reshape(-1)) * npergrid

        randid = np.random.permutation(range(len(nsas)))  # noqa: NPY002

        for nc in range(len(nsas)):
            nsa = nsas[randid[nc]]
            ngr = ngrs[randid[nc]]

            if not found_scaling_anchor:
                # If there is a scaling anchor
                # T_cond = 2
                # Sa_T1 = np.zeros((numgm,))
                # for ng in range(numgm):
                #     Sa_T1[ng] = np.interp(T_cond, periods, geomPSA[ng])
                # SaT_ref = min(1.5, 0.9 / T_cond)
                sf_pool = np.ones((numgm,))
                penalty_pool = np.zeros((numgm,))

            else:
                SaT_ref = np.exp(IM_log_ref_scaling_anchor[nsa])  # noqa: N806
                Sa_T1 = np.exp(IM_log_data_scaling_anchor)  # noqa: N806

                # penalty for scaling factor

                sf_pool = SaT_ref / Sa_T1  # scaling factors
                penalty_pool = np.zeros((numgm,))
                temptag1 = np.where(sf_pool < sf_min)
                penalty_pool[temptag1] = (sf_min - sf_pool[temptag1]) ** 2
                temptag2 = np.where(sf_pool > sf_max)
                penalty_pool[temptag2] = (sf_max - sf_pool[temptag2]) ** 2

            if IM_log_data_pool2.shape[1] > 0:
                IM_log_data_pool3 = (  # noqa: N806
                    IM_log_data_pool2
                    + np.log(sf_pool[np.newaxis]).T * scaling_exponent2[np.newaxis]
                )
                normData = IM_log_data_pool3 / log_im_range2  # noqa: N806
                normRefGrid = IM_log_ref2 / log_im_range2  # noqa: N806
                err_mat = (
                    distance_matrix(normData, normRefGrid, p=2) ** 2 / lenRef2**2
                    + np.tile(penalty_pool, (int(nGridPerIM), 1)).T * sf_penalty
                )
                err_pure = (
                    distance_matrix(normData, normRefGrid, p=2) ** 2 / lenRef2**2
                )
            else:
                err_mat = np.tile(penalty_pool, (int(nGridPerIM), 1)).T * sf_penalty
                err_pure = np.tile(penalty_pool, (int(nGridPerIM), 1)).T

            minerr = np.sort(err_mat, axis=0)
            minerr_tag = np.argsort(err_mat, axis=0)

            count = 0
            for ng in minerr_tag[:, ngr]:
                cureqID = eqnameID[ng]  # noqa: N806
                cureqID_existnum = np.sum(cureqID == np.array(selected_gm_eqID))  # noqa: N806

                if (selected_gm_ID.count(ng) == 0) and (cureqID_existnum < numEQmax):
                    break  # we only consider this

                count += 1
                if ng == minerr_tag[-1, ngr]:
                    msg = 'not enough ground motion to match your criteria'
                    print(msg)  # noqa: T201
                    errf.write(msg)
                    errf.close()
                    exit(-1)  # noqa: PLR1722

            selected_gm_ID += [ng]  # noqa: N806
            selected_gm_err += [minerr[count, ngr]]
            selected_gm_eqID += [cureqID]  # noqa: N806
            selected_gm_scale += [sf_pool[ng]]

            err_sum[nsa, ngr] += err_pure[ng, ngr]

        flat_gm_ID = selected_gm_ID  # noqa: N806
        flat_gm_scale = selected_gm_scale
        flat_RSN = [RSN[myid] for myid in flat_gm_ID]  # noqa: N806, F841

        #
        # Write the results
        #
        idx = np.argsort([RSN[myid] for myid in flat_gm_ID])
        my_results = {}
        my_results['gm_RSN'] = [int(RSN[int(flat_gm_ID[myid])]) for myid in idx]
        my_results['gm_scale'] = [flat_gm_scale[myid] for myid in idx]

        with open('gridIM_output.json', 'w') as f:  # noqa: PTH123
            f.write(json.dumps(my_results))

        #
        # Drawing starts
        #

        # import matplotlib.pyplot as plt
        # import matplotlib.ticker as mticker
        from scipy import interpolate

        # plt.style.use('default')
        #
        # # Option 1
        # plt.rcParams['font.size'] = 14
        # try:
        #     plt.rcParams["font.family"] = "Times New Roman"
        # except:
        #     pass

        theLogIM = []  # noqa: N806
        LogIMref = []  # noqa: N806
        for idx in range(nim):
            theLogSF = np.log(np.array(selected_gm_scale) ** scaling_exponent[idx])  # noqa: N806
            theLogIM += [np.array(IM_log_data_pool[selected_gm_ID, idx]) + theLogSF]  # noqa: N806
            LogIMref += [  # noqa: N806
                np.linspace(log_im_lb[idx], log_im_ub[idx], int(im_nbins[idx]))
            ]

        # my color map
        # colors = [(0, 0, 1), (1, 0, 0)]  # first color is black, last is red
        # mycm = LinearSegmentedColormap.from_list(
        #     "Custom", colors, N=20)

        colorscale = [[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']]

        if nim == 3:  # noqa: PLR2004
            flat_grid_error = err_sum.T.flatten() / npergrid

            if found_scaling_anchor:
                aa = np.delete(np.array([0, 1, 2]), id_im_scaling_ancher)
                idx1 = aa[0]
                idx2 = aa[1]
                idx3 = id_im_scaling_ancher
            else:
                idx1 = 0
                idx2 = 1
                idx3 = 2

            #
            # reference points
            #

            X, Y, Z = np.meshgrid(LogIMref[idx1], LogIMref[idx2], LogIMref[idx3])  # noqa: N806

            fig = px.scatter_3d(
                x=np.exp(X.reshape(-1)),
                y=np.exp(Y.reshape(-1)),
                z=np.exp(Z.reshape(-1)),
                color=flat_grid_error,
                log_x=True,
                log_y=True,
                log_z=True,
                color_continuous_scale=colorscale,
            )

            fig.update_traces(
                marker=dict(  # noqa: C408
                    size=7,
                    line=dict(width=2),  # noqa: C408
                )
            )

            fig['data'][0]['showlegend'] = True
            fig['data'][0]['name'] = 'anchor point'
            fig.layout.coloraxis.colorbar.title = (
                'Ground <br>motion <br>coverage <br>(error level)'
            )
            fig.update_coloraxes(
                cmin=0,
                cmax=1,
            )

            fig.add_scatter3d(
                x=np.exp(theLogIM[idx1]),
                y=np.exp(theLogIM[idx2]),
                z=np.exp(theLogIM[idx3]),
                mode='markers',
                marker=dict(  # noqa: C408
                    size=4,
                    line=dict(width=1, color='black'),  # noqa: C408
                    color='orange',
                ),
                name='selected ground motion',
            )

            fig.update_layout(
                scene=dict(  # noqa: C408
                    xaxis=dict(  # noqa: C408
                        tickmode='array',
                        # tickvals=[im_lb[idx1],im_ub[idx1],0.001,0.01,0.1,1,10,100],),
                        tickvals=[
                            im_lb[idx1],
                            im_ub[idx1],
                            0.001,
                            0.005,
                            0.01,
                            0.05,
                            0.1,
                            0.5,
                            1,
                            5,
                            10,
                            50,
                            100,
                        ],
                        title=im_names[idx1] + myunits[idx1],
                    ),
                    yaxis=dict(  # noqa: C408
                        tickmode='array',
                        # tickvals=[im_lb[idx2],im_ub[idx2],0.001,0.01,0.1,1,10,100],),
                        tickvals=[
                            im_lb[idx2],
                            im_ub[idx2],
                            0.001,
                            0.005,
                            0.01,
                            0.05,
                            0.1,
                            0.5,
                            1,
                            5,
                            10,
                            50,
                            100,
                        ],
                        title=im_names[idx2] + myunits[idx2],
                    ),
                    zaxis=dict(  # noqa: C408
                        tickmode='array',
                        # tickvals=[im_lb[idx3],im_ub[idx3],0.001,0.01,0.1,1,10,100],),
                        tickvals=[
                            im_lb[idx3],
                            im_ub[idx3],
                            0.001,
                            0.005,
                            0.01,
                            0.05,
                            0.1,
                            0.5,
                            1,
                            5,
                            10,
                            50,
                            100,
                        ],
                        title=im_names[idx3] + myunits[idx3],
                    ),
                    aspectmode='cube',
                ),
                legend=dict(  # noqa: C408
                    x=0,
                    y=0,
                    xanchor='left',
                    yanchor='top',
                ),
                # paper_bgcolor='rgba(0,0,0,0)',
                autosize=False,
                height=500,
                width=550,
                legend_orientation='h',
                scene_camera=dict(eye=dict(x=2, y=2, z=0.6)),  # noqa: C408
                margin=dict(l=20, r=20, t=20, b=20),  # noqa: C408
            )

            """
            fig = plt.figure();
            ax = fig.add_subplot(projection='3d')

            sc = ax.scatter(X.reshape(-1), Y.reshape(-1), Z.reshape(-1), c=flat_grid_error, cmap=mycm, vmin=0,
                            vmax=1, s=95,alpha=0.7, edgecolors='k')

            ax.scatter(theLogIM[idx1], theLogIM[idx2], theLogIM[idx3], s=20, color='y', edgecolors='k', alpha=1)

            plt.xlabel(im_names[idx1] + myunits[idx1]);
            plt.ylabel(im_names[idx2] +  myunits[idx2])
            ax.set_zlabel(im_names[idx3] + myunits[idx3]);

            ticks_final=[]
            for idx in range(nim):
                myticks = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3, 1e4])
                tick_idx = np.argwhere((myticks>im_lb[idx]) * (myticks<im_ub[idx])).T[0]
                ticks_final += [np.hstack([myticks[tick_idx.astype(int)], np.array([im_lb[idx], im_ub[idx]])])]

            ax.set_xticks(np.log(ticks_final[idx1]))
            ax.set_xticklabels(ticks_final[idx1])

            ax.set_yticks(np.log(ticks_final[idx2]))
            ax.set_yticklabels(ticks_final[idx2])

            ax.set_zticks(np.log(ticks_final[idx3]))
            ax.set_zticklabels(ticks_final[idx3])
            plt.legend(["anchor point", "selected ground motion"], ncol=2, bbox_to_anchor=(0,0.02,1,0.05), loc="upper left")
            plt.title("Ground motion coverage", x=0.5, y=0.9)

            cax = fig.add_axes([ax.get_position().x1 + 0.05,
                                ax.get_position().y0 + 0.2,
                                0.03,
                                ax.get_position().height/2])
            fig.colorbar(sc,label= "coverage (error level)", cax=cax)

            ax.view_init(10, 30)
            """
        if nim == 2:  # noqa: PLR2004
            flat_grid_error = err_sum.flatten() / npergrid

            idx1 = 0
            idx2 = 1

            #
            # data points
            #

            X, Y = np.meshgrid(LogIMref[idx1], LogIMref[idx2])  # noqa: N806

            #
            # interpolated area
            #
            lowerboundX = np.min(np.log(im_lb[0]) - log_im_range[0] * 0.05)  # noqa: N806
            upperboundX = np.max(np.log(im_ub[0]) + log_im_range[0] * 0.05)  # noqa: N806
            lowerboundY = np.min(np.log(im_lb[1]) - log_im_range[1] * 0.05)  # noqa: N806
            upperboundY = np.max(np.log(im_ub[1]) + log_im_range[1] * 0.05)  # noqa: N806

            xx = np.linspace(lowerboundX, upperboundX, 20)
            yy = np.linspace(lowerboundY, upperboundY, 20)
            xxx, yyy = np.meshgrid(xx, yy)
            f = interpolate.interp2d(
                (X.reshape(-1)), (Y.reshape(-1)), flat_grid_error
            )
            zzz = f(xx, yy)

            #
            # Figure
            #

            fig = px.scatter(
                x=np.exp(X.reshape(-1)),
                y=np.exp(Y.reshape(-1)),
                color=flat_grid_error,
                log_x=True,
                log_y=True,
                color_continuous_scale=colorscale,
            )
            fig.update_traces(
                marker=dict(  # noqa: C408
                    size=15,
                    line=dict(width=2, color='black'),  # noqa: C408
                )
            )

            fig['data'][0]['showlegend'] = True
            fig['data'][0]['name'] = 'anchor point'

            fig.add_scatter(
                x=np.exp(theLogIM[idx1]),
                y=np.exp(theLogIM[idx2]),
                mode='markers',
                marker=dict(  # noqa: C408
                    size=5,
                    line=dict(width=1, color='black'),  # noqa: C408
                    color='orange',
                ),
                name='selected ground motion',
            )

            # fig = px.scatter(x=[None],y=[None],log_x=True,log_y=True,)
            # fig.update(layout_coloraxis_showscale=False)
            fig.layout.coloraxis.colorbar.title = (
                'Ground <br>motion <br>coverage <br>(error level)'
            )
            fig.add_heatmap(
                x=np.exp(xx),
                y=np.exp(yy),
                z=zzz,
                zmin=0,
                zmax=1,
                colorscale=colorscale,
                coloraxis='coloraxis',
                opacity=0.5,
                hoverinfo='skip',
            )

            fig.update_layout(
                xaxis=dict(  # noqa: C408
                    tickmode='array',
                    # tickvals=[im_lb[idx1],im_ub[idx1],0.001,0.01,0.1,1,10,100],),
                    tickvals=[
                        im_lb[idx1],
                        im_ub[idx1],
                        0.001,
                        0.005,
                        0.01,
                        0.05,
                        0.1,
                        0.5,
                        1,
                        5,
                        10,
                        50,
                        100,
                    ],
                    title=im_names[idx1] + myunits[idx1],
                ),
                yaxis=dict(  # noqa: C408
                    tickmode='array',
                    # tickvals=[im_lb[idx2],im_ub[idx2],0.001,0.01,0.1,1,10,100],),
                    tickvals=[
                        im_lb[idx2],
                        im_ub[idx2],
                        0.001,
                        0.005,
                        0.01,
                        0.05,
                        0.1,
                        0.5,
                        1,
                        5,
                        10,
                        50,
                        100,
                    ],
                    title=im_names[idx2] + myunits[idx2],
                ),
                legend=dict(  # noqa: C408
                    x=0,
                    y=-0.1,
                    xanchor='left',
                    yanchor='top',
                ),
                # paper_bgcolor='rgba(0,0,0,0)',
                autosize=False,
                height=500,
                width=550,
                legend_orientation='h',
                margin=dict(l=20, r=20, t=20, b=20),  # noqa: C408
            )
            fig.update_coloraxes(
                cmin=0,
                cmax=1,
            )

            """

            fig = plt.figure();
            ax = fig.add_subplot()
            ax.set_xscale('log')
            ax.set_yscale('log')

            #
            # data points
            #

            X, Y = np.meshgrid(LogIMref[0], LogIMref[1])

            #
            # interpolated area
            #
            lowerboundX = np.min(( np.log(im_lb[0])-log_im_range[0]*0.05 ))
            upperboundX = np.max(( np.log(im_ub[0])+log_im_range[0]*0.05))
            lowerboundY = np.min(( np.log(im_lb[1])-log_im_range[1]*0.05 ))
            upperboundY = np.max(( np.log(im_ub[1])+log_im_range[1]*0.05))

            xx =  np.linspace(lowerboundX, upperboundX, 20)
            yy =  np.linspace(lowerboundY, upperboundY, 20)
            xxx, yyy = np.meshgrid(xx, yy)
            f = interpolate.interp2d((X.reshape(-1)), (Y.reshape(-1)) , flat_grid_error)
            zzz = f(xx,yy)

            #
            # Start plotting
            #

            C = ax.pcolormesh(np.exp(xxx), np.exp(yyy), zzz, shading='nearest',cmap=mycm, vmin=0, vmax=1)
            sc = ax.scatter(np.exp(X.reshape(-1)), np.exp(Y.reshape(-1)) , c=  flat_grid_error,edgecolors='k',  cmap = mycm, vmin=0, vmax=1, s = 250)
            ax.scatter(np.exp(theLogIM[0]),  np.exp(theLogIM[1]), c='y', edgecolors='k', cmap=mycm, vmin=0, vmax=1,  s=55)

            #ax.plot(np.exp(theLogIM[0]), np.exp(theLogIM[1]) ,'.',markersize=10,markerfacecolor='y',color='k')

            plt.xlabel(im_names[0] + myunits[0]);
            plt.ylabel(im_names[1] + myunits[1]);

            #
            # minor formatting
            #
            idx1 = 0; idx2 = 1;
            tick_idx1 = np.argwhere((ax.get_xticks()>im_lb[idx1]) * (ax.get_xticks()<im_ub[idx1])).T[0]
            tick_idx2 = np.argwhere((ax.get_yticks()>im_lb[idx2]) * (ax.get_yticks()<im_ub[idx2])).T[0]
            plt.xticks(np.hstack([ax.get_xticks(), np.array([im_lb[idx1], im_ub[idx1]])]))
            plt.yticks(np.hstack([ax.get_yticks(), np.array([im_lb[idx2], im_ub[idx2]])]))



            #
            lowerboundX = np.min(( np.log(im_lb[0])-log_im_range[0]*0.05 , np.min(theLogIM[0])))
            upperboundX = np.max(( np.log(im_ub[0])+log_im_range[0]*0.05 , np.max(theLogIM[0])))
            lowerboundY = np.min(( np.log(im_lb[1])-log_im_range[1]*0.05 , np.min(theLogIM[1])))
            upperboundY = np.max(( np.log(im_ub[1])+log_im_range[1]*0.05 , np.max(theLogIM[1])))
            plt.xlim([np.exp(lowerboundX - log_im_range[idx1]*0.1), np.exp(upperboundX + log_im_range[idx1]*0.1)])
            plt.ylim([np.exp(lowerboundY - log_im_range[idx2]*0.1), np.exp(upperboundY + log_im_range[idx2]*0.1)])

            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            plt.grid()
            plt.legend(["anchor point", "selected ground motion"], ncol=2, bbox_to_anchor=(0,0.02,1,-0.15), loc="upper left")
            plt.title("Ground motion coverage", x=0.5, y=1.05)
            fig.colorbar(sc,label= "coverage (error level)")
            """
        if nim == 1:
            pass
        #     flat_grid_error = err_sum.flatten() / npergrid

        #     import matplotlib.pyplot as plt

        #     ngrid_1axis = int(im_nbins[0])
        #     mypoints = np.zeros((0,nim))

        #     for nsa in range(ngrid_1axis):

        #         idx1 = 0
        #         theLogSF1 = np.log(np.array(selected_gm_scale_list[nsa]) ** scaling_exponent[idx1])

        #         theLogIM1 = np.array(IM_log_data_pool[selected_gm_ID_list[nsa], idx1])

        #         mypoints_tmp = np.vstack([theLogIM1 + theLogSF1]).T
        #         mypoints = np.vstack([mypoints,mypoints_tmp])

        #     X = np.linspace(log_im_lb[idx1], log_im_ub[idx1], int(im_nbins[idx1]))
        #     IM_log_ref = np.vstack([X.reshape(-1)]).T

        #     fig = plt.figure()
        #     ax = fig.add_subplot()
        #     ax.scatter(mypoints[:, 0], 0*mypoints[:, 0],s=18)
        #     ax.scatter(IM_log_ref[:, 0], 0*IM_log_ref[:, 0],s=5)
        #     plt.xlabel(im_names[idx1]);

        # plt.savefig('gridIM_coverage.png',bbox_inches='tight')
        if nim == 2 or nim == 3:  # noqa: PLR1714, PLR2004
            with open(r'gridIM_coverage.html', 'w') as f:  # noqa: PTH123
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            f.close()


if __name__ == '__main__':
    errf = open('gridIM_log.err', 'w')  # noqa: SIM115, PTH123
    main(sys.argv, errf)
    # try:
    #     main(sys.argv,errf)
    #     errf.close()
    #
    # except Exception as e:
    #     print("Exception occurred while code Execution: " + str(repr(e)))
    #     errf.write("Exception occurred while code Execution: " + str(repr(e)))
    #     errf.close()
    #     exit(-1)
