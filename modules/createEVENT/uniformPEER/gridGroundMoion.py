# -*- coding: utf-8 -*-
#
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


#TODO recommended ranges???


import json
import sys
import numpy as np
from scipy.stats import qmc
from scipy.stats import gmean
from scipy.spatial import distance_matrix
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def main(inputArgs,err):
    gms = gmCluster(inputArgs,err)


class gmCluster():
    def __init__(self, inputArgs,err):
        curDir = os.path.dirname(__file__)
        gmDataBaseDir = os.path.join(curDir,"gmdata.json")
        inputJsonPath = inputArgs[1]

        with open(inputJsonPath,'r') as fj:
            inputJson = json.load(fj)

        nim = len(inputJson["IM"])

        im_ub = np.zeros((nim,))
        im_lb = np.zeros((nim,))
        im_nbins = np.zeros((nim,))
        im_names = []
        im_periods = []
        i = 0
        for imName, value in inputJson["IM"].items():
            im_names += [imName]
            im_ub[i] = float(value["upperBound"])
            im_lb[i] = float(value["lowerBound"])
            im_nbins[i] = int(value["numBins"])
            im_periods += [value["Periods"]]
            i +=1

        npergrid = int(inputJson["numSampPerBin"])

        # TODO: Convert the units... Or fix the units......

        #nim = len(im_names)
        ngrid = np.prod(im_nbins)
        #
        # Clustring parameters
        #
        numEQmax = int(max(1,round(ngrid/10))) # Maximum number of records from the single earthquake
        #numEQmax = 1

        #
        # Get grid of IMs - change everything to log-space
        #
        log_im_ub = np.log(im_ub)
        log_im_lb = np.log(im_lb)
        log_im_range = log_im_ub-log_im_lb
        # X is log-IM

        id_im_scaling_ancher = -1
        found_scaling_anchor = False
        nim_eff = nim
        for ni in range(len(im_names)):
            if im_names[ni].startswith("PSA") and not im_names[ni] == "SaRatio" :
                # scaling anchor
                if not found_scaling_anchor:
                    id_im_scaling_ancher = ni  # TODO
                    found_scaling_anchor = True
                    nim_eff = nim-1


        if nim<=0:
            # ERROR
            msg = "number of IMs should be greater than 1"
            print(msg)
            errf.write(msg)
            errf.close()
            exit(-1)

        elif nim_eff ==0:
            # One variable we have is the scaling anchor
                myID = [1]
                Scaling_ref = np.linspace(log_im_lb[0], log_im_ub[0], int(im_nbins[0]))
                IM_log_ref = np.zeros(0); # dummy
                isGrid = True

        elif nim_eff ==1:
            if found_scaling_anchor:
                if found_scaling_anchor:
                    myID = np.delete([0, 1], id_im_scaling_ancher)
                    Scaling_ref = np.linspace(log_im_lb[id_im_scaling_ancher], log_im_ub[id_im_scaling_ancher],
                                              int(im_nbins[id_im_scaling_ancher]))
                else:
                    myID = [0]
                X = np.linspace(log_im_lb[myID[0]], log_im_ub[myID[0]], int(im_nbins[myID[0]]))
                IM_log_ref = X[np.newaxis].T
                isGrid = True

        elif nim_eff ==2:
            if found_scaling_anchor:
                myID = np.delete([0,1,2],id_im_scaling_ancher)
                Scaling_ref = np.linspace(log_im_lb[id_im_scaling_ancher], log_im_ub[id_im_scaling_ancher], int(im_nbins[id_im_scaling_ancher]))
            else:
                myID = [0,1]

            X,Y = np.meshgrid(np.linspace(log_im_lb[myID[0]], log_im_ub[myID[0]], int(im_nbins[myID[0]])), np.linspace(log_im_lb[myID[1]], log_im_ub[myID[1]], int(im_nbins[myID[1]])))
            IM_log_ref = np.vstack([X.reshape(-1), Y.reshape(-1)]).T
            isGrid = True
        elif nim_eff ==3:
            if found_scaling_anchor:
                myID = np.delete([0,1,2,3],id_im_scaling_ancher)
                Scaling_ref = np.linspace(log_im_lb[id_im_scaling_ancher], log_im_ub[id_im_scaling_ancher], int(im_nbins[id_im_scaling_ancher]))
            else:
                myID = [0,1,2]

            X,Y,Z = np.meshgrid(np.linspace(log_im_lb[myID[0]], log_im_ub[myID[0]], int(im_nbins[myID[0]])), np.linspace(log_im_lb[myID[1]], log_im_ub[myID[1]], int(im_nbins[myID[1]])), np.linspace(log_im_lb[myID[2]], log_im_ub[myID[2]], int(im_nbins[myID[2]])))
            IM_log_ref = np.vstack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).T
            isGrid = True
        else:
            if found_scaling_anchor:
                myID = np.delete(range(nim_eff+1),id_im_scaling_ancher)
                Scaling_ref = np.linspace(log_im_lb[id_im_scaling_ancher], log_im_ub[id_im_scaling_ancher], int(im_nbins[id_im_scaling_ancher]))
            else:
                myID = range(nim_eff)

            # Let us do LHS sampling
            sampler= qmc.LatinHypercube(d=nim)
            U = sampler.random(n=ngrid)
            X= np.zeros((ngrid,nim_eff))
            for i in range(nim_eff):
                X[:,i] = U[:,i]*(log_im_ub[myID[i]]-log_im_lb[myID[i]]) + log_im_lb[myID[i]]
            IM_log_ref = X
            isGrid = False


        #
        # Read Database
        #
        with open(gmDataBaseDir,'r') as fd:
            gmData = json.load(fd)

        RSN = gmData["RSN"]
        geomPSA = gmData["geomPSA"]
        geomPGA = gmData["geomPGA"]
        geomPGV = gmData["geomPGV"]
        geomPGD = gmData["geomPGD"]
        geomDS575 = gmData["geomDS575"]
        geomDS595 = gmData["geomDS595"]
        geomIa = gmData["geomIa"]

        periods = gmData["period"]
        numgm = gmData["numgm"]
        eqnameID = gmData["eqnameID"]
        units = gmData["unit"]

        #
        # Define Sa(T_cond)
        #

        # compute scaling factors

        #
        # Compute SaRatio(T_lowbound,T_cond,T_highbound) and Ds575
        #

        IM_log_data_pool = np.zeros((numgm,0))
        scaling_exponent = np.zeros((nim,))
        myunits = []
        for ni in range(nim):
            if im_names[ni].startswith("PSA"):
                Sa_T1 = np.zeros((numgm,))
                T_cond = float(im_periods[ni][0])  # central (<= 5.0)
                for ng in range(numgm):
                    Sa_T1[ng] = np.interp(T_cond, periods, geomPSA[ng])

                Sa1_pool = Sa_T1[np.newaxis].T
                IM_log_data_pool = np.hstack([IM_log_data_pool, np.log(Sa1_pool)])
                scaling_exponent[ni] = 1

                myunits += ['('+units["PSA"]+')']
            elif im_names[ni]=="SaRatio":
                Sa_T1 = np.zeros((numgm,))
                Sa_T_geomean = np.zeros((numgm,))

                T_lowbound = float(im_periods[ni][0])  # low-bound
                T_cond = float(im_periods[ni][1])  # central (<= 5.0)
                T_highbound = float(im_periods[ni][2])  # high-bound

                idx_T_range = np.where((np.array(periods) > T_lowbound) * (np.array(periods) < T_highbound))[0]

                for ng in range(numgm):
                    Sa_T1[ng] = np.interp(T_cond, periods, geomPSA[ng])
                    Sa_T_geomean[ng] = gmean(np.array(geomPSA[ng])[idx_T_range.astype(int)])

                SaRatio_pool = (Sa_T1 / Sa_T_geomean)[np.newaxis].T
                IM_log_data_pool = np.hstack([IM_log_data_pool,np.log(SaRatio_pool)])
                scaling_exponent[ni] = 0

                myunits += [""]
            elif im_names[ni]=="DS575":
                ds_pool = (np.array(geomDS575))[np.newaxis].T
                IM_log_data_pool = np.hstack([IM_log_data_pool, np.log(ds_pool)])
                scaling_exponent[ni] = 0
                myunits += ['('+units["DS575"]+')']

            elif im_names[ni]=="DS595":
                ds_pool = (np.array(geomDS595))[np.newaxis].T
                IM_log_data_pool = np.hstack([IM_log_data_pool, np.log(ds_pool)])
                scaling_exponent[ni] = 0
                myunits += ['('+units["DS595"]+')']

            elif im_names[ni]=="PGA":
                pg_pool = (np.array(geomPGA))[np.newaxis].T
                IM_log_data_pool = np.hstack([IM_log_data_pool, np.log(pg_pool)])
                scaling_exponent[ni] = 1
                myunits += ['('+units["PGA"]+')']

            elif im_names[ni]=="PGV":
                pg_pool = (np.array(geomPGV))[np.newaxis].T
                IM_log_data_pool = np.hstack([IM_log_data_pool, np.log(pg_pool)])
                scaling_exponent[ni] = 1
                myunits += ['('+units["PGV"]+')']

            elif im_names[ni]=="PGD":
                pg_pool = (np.array(geomPGD))[np.newaxis].T
                IM_log_data_pool = np.hstack([IM_log_data_pool, np.log(pg_pool)])
                scaling_exponent[ni] = 1
                myunits += ['('+ units["PGD"]+')']

            elif im_names[ni]=="Arias":
                ai_pool = (np.array(geomIa))[np.newaxis].T
                IM_log_data_pool = np.hstack([IM_log_data_pool, np.log(ai_pool)])
                scaling_exponent[ni] = 2
                myunits += ['('+units["Ia"]+')']
            else:
                msg = "unrecognized IM name "+im_names[ni]
                print(msg)
                errf.write(msg)
                errf.close()
                exit(-1)

        if id_im_scaling_ancher>=0:
            IM_log_data_scaling_anchor = IM_log_data_pool[:,id_im_scaling_ancher]
            #IM_log_ref_scaling_anchor = IM_log_ref[:,id_im_scaling_ancher]
            IM_log_ref_scaling_anchor = Scaling_ref

            IM_log_data_pool2 = np.delete(IM_log_data_pool, id_im_scaling_ancher, 1)
            IM_log_ref2 = IM_log_ref.copy()
            scaling_exponent2 = np.delete(scaling_exponent, id_im_scaling_ancher)

            log_im_range2 = np.delete(log_im_range.copy(), id_im_scaling_ancher)

            lenRef2 = np.linalg.norm(1 / np.delete(im_nbins.copy(), id_im_scaling_ancher))


        if id_im_scaling_ancher>=0:
            if isGrid:
                nScalingGrid = im_nbins[id_im_scaling_ancher]
                nGridPerIM = ngrid/im_nbins[id_im_scaling_ancher]
            else:
                nScalingGrid = ngrid
                nGridPerIM = ngrid/im_nbins[id_im_scaling_ancher]
        else:
            nScalingGrid = 1
            nGridPerIM = ngrid



        sf_min = 0.5  # minimum of no-panalty scaling
        sf_max = 10.0  # maximum of no-pad nalty scaling
        sf_penalty = 10.0  # per unit outside the tolerance range(sf_min~sf_max)

        # selected_gm_ID_list =[]
        # selected_gm_err_list =[]
        # selected_gm_eqID_list =[]
        # selected_gm_scale_list =[]
        selected_gm_ID = []
        selected_gm_err = []
        selected_gm_eqID = []
        selected_gm_scale = []

        err_sum = np.zeros((int(nScalingGrid),int(nGridPerIM)))
        for nsa in range(int(nScalingGrid)):

            if not found_scaling_anchor:
                # If there is a scaling anchor
                T_cond = 2
                Sa_T1 = np.zeros((numgm,))
                for ng in range(numgm):
                    Sa_T1[ng] = np.interp(T_cond, periods, geomPSA[ng])
                SaT_ref = min(1.5, 0.9 / T_cond)

            else:
                SaT_ref = np.exp(IM_log_ref_scaling_anchor[nsa])
                Sa_T1 = np.exp(IM_log_data_scaling_anchor)

            # penalty for scaling factor

            sf_pool = SaT_ref / Sa_T1  # scaling factors
            penalty_pool = np.zeros((numgm,))
            temptag1 = np.where(sf_pool < sf_min)
            penalty_pool[temptag1] = (sf_min - sf_pool[temptag1]) ** 2
            temptag2 = np.where(sf_pool > sf_max)
            penalty_pool[temptag2] = (sf_max - sf_pool[temptag2]) ** 2;

            IM_log_data_pool3 = IM_log_data_pool2 + np.log(sf_pool[np.newaxis]).T * scaling_exponent2[np.newaxis]


            if IM_log_data_pool3.shape[1]>0:
                normData = IM_log_data_pool3/log_im_range2
                normRefGrid  =IM_log_ref2/log_im_range2
                err_mat = distance_matrix(normData, normRefGrid, p=2) ** 2 / lenRef2**2 + np.tile(penalty_pool,(int(nGridPerIM), 1)).T * sf_penalty
                err_pure = distance_matrix(normData, normRefGrid, p=2) ** 2 / lenRef2**2
            else:
                err_mat = np.tile(penalty_pool,(int(nGridPerIM), 1)).T * sf_penalty
                err_pure = np.tile(penalty_pool,(int(nGridPerIM), 1)).T

            minerr = np.sort(err_mat,axis=0)
            minerr_tag = np.argsort(err_mat,axis=0)

            for nrep in range(npergrid):
                for ngr in np.random.permutation(int(nGridPerIM)):
                    count = 0
                    error_sum = 0
                    for ng in minerr_tag[:,ngr]:

                        cureqID = eqnameID[ng]
                        cureqID_existnum = np.sum(cureqID == np.array(selected_gm_eqID))

                        if (selected_gm_ID.count(ng)==0) and(cureqID_existnum<numEQmax):
                            break  # we only consider this

                        count += 1
                        if ng==minerr_tag[-1,ngr]:
                            msg = "not enough ground motion to match your criteria"
                            print(msg)
                            errf.write(msg)
                            errf.close()
                            exit(-1)

                    selected_gm_ID += [ng]
                    selected_gm_err += [minerr[count,ngr]]
                    selected_gm_eqID += [cureqID]
                    selected_gm_scale += [sf_pool[ng]]

                    err_sum[nsa,ngr] += err_pure[ng,ngr]

        flat_gm_ID = selected_gm_ID
        flat_gm_scale = selected_gm_scale
        flat_grid_error = err_sum.flatten()/npergrid

        #
        # Write the results
        #
        idx = np.argsort([RSN[myid] for myid in flat_gm_ID])
        my_results={}
        my_results["gm_RSN"] = [int(RSN[int(flat_gm_ID[myid])]) for myid in idx]
        my_results["gm_scale"] = [flat_gm_scale[myid] for myid in idx]

        with open('gridIM_output.json', 'w') as fo:
            fo.write(json.dumps(my_results))



        #
        # Drawing starts
        #

        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from scipy import interpolate

        plt.style.use('default')

        # Option 1
        plt.rcParams['font.size'] = 14
        try:
            plt.rcParams["font.family"] = "Times New Roman"
        except:
            pass


        theLogIM =[]
        LogIMref = []
        for idx in range (nim):
            theLogSF = np.log(np.array(selected_gm_scale) ** scaling_exponent[idx])
            theLogIM += [np.array(IM_log_data_pool[selected_gm_ID, idx]) + theLogSF]
            LogIMref += [np.linspace(log_im_lb[idx], log_im_ub[idx], int(im_nbins[idx]))]

        if nim==3:

            aa = np.delete(np.array([0,1,2]), id_im_scaling_ancher)
            idx1 = aa[0]
            idx2 = aa[1]
            idx3 = id_im_scaling_ancher

            #
            # reference points
            #

            X, Y, Z = np.meshgrid(LogIMref[idx1], LogIMref[idx2], LogIMref[idx3])

            fig = plt.figure();
            ax = fig.add_subplot(projection='3d')

            sc = ax.scatter(X.reshape(-1), Y.reshape(-1), Z.reshape(-1), c=flat_grid_error, cmap='seismic', vmin=0,
                            vmax=1, s=95,alpha=0.7)

            ax.scatter(theLogIM[idx1], theLogIM[idx2], theLogIM[idx3], s=20, color='y', edgecolors='k', alpha=1)

            fig.colorbar(sc,label= "coverage (error level)")

            plt.xlabel(im_names[idx1] + myunits[idx1]);
            plt.ylabel(im_names[idx2] +  myunits[idx2])
            ax.set_zlabel(im_names[idx3] + myunits[idx3]);

            ticks_final=[]
            for idx in range(nim):
                myticks = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3, 1e4])
                tick_idx = np.argwhere((myticks>im_lb[idx]) * (myticks<im_ub[idx])).T[0]
                ticks_final += [np.hstack([myticks[int(tick_idx)], np.array([im_lb[idx], im_ub[idx]])])]

            ax.set_xticks(np.log(ticks_final[idx1]))
            ax.set_xticklabels(ticks_final[idx1])

            ax.set_yticks(np.log(ticks_final[idx2]))
            ax.set_yticklabels(ticks_final[idx2])

            ax.set_zticks(np.log(ticks_final[idx3]))
            ax.set_zticklabels(ticks_final[idx3])

            ax.view_init(10, 50)
            plt.legend(["anchor points", "collected samples"])

        if nim==2:

            fig = plt.figure();
            ax = fig.add_subplot()

            ax.set_xscale('log')
            ax.set_yscale('log')
            #
            # data points
            #


            X, Y = np.meshgrid(LogIMre[0], LogIMref[1])

            #
            # interpolated area
            #

            xx =  np.linspace(np.log(im_lb[0])-log_im_range[0]*0.05, np.log(im_ub[0])+log_im_range[0]*0.05, 20)
            yy =  np.linspace(np.log(im_lb[1])-log_im_range[1]*0.05, np.log(im_ub[1])+log_im_range[1]*0.05, 20)
            xxx, yyy = np.meshgrid(xx, yy)
            f = interpolate.interp2d((X.reshape(-1)), (Y.reshape(-1)) , flat_grid_error)
            zzz = f(xx,yy)

            C = ax.pcolormesh(np.exp(xxx), np.exp(yyy), zzz, shading='nearest',alpha=0.2,cmap='seismic')
            ax.scatter(np.exp(X.reshape(-1)), np.exp(Y.reshape(-1)) , c=  flat_grid_error, cmap = 'seismic', vmin=0, vmax=1, s = 95)
            ax.plot(np.exp(theLogIM[0]), np.exp(theLogIM[1]) ,'.',markersize=10,markerfacecolor='y',color='k')

            plt.xlabel(im_names[0] + myunits[0]);
            plt.ylabel(im_names[1] + myunits[1]);

            #
            # minor formatting
            #
            idx1 = 0; idx2 = 1;
            tick_idx1 = np.argwhere((ax.get_xticks()>im_lb[idx1]) * (ax.get_xticks()<im_ub[idx1])).T[0]
            tick_idx2 = np.argwhere((ax.get_yticks()>im_lb[idx2]) * (ax.get_yticks()<im_ub[idx2])).T[0]
            plt.xticks(np.hstack([ax.get_xticks()[tick_idx1], np.array([im_lb[idx1], im_ub[idx1]])]))
            plt.yticks(np.hstack([ax.get_yticks()[tick_idx2], np.array([im_lb[idx2], im_ub[idx2]])]))
            plt.xlim([im_lb[idx1]/np.exp(log_im_range[idx1]*0.1), im_ub[idx1]*np.exp(log_im_range[idx1]*0.1)])
            plt.ylim([im_lb[idx2]/np.exp(log_im_range[idx2]*0.1), im_ub[idx2]*np.exp(log_im_range[idx2]*0.1)])
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            plt.grid()

        if nim==1:

            import matplotlib.pyplot as plt

            ngrid_1axis = int(im_nbins[0])
            mypoints = np.zeros((0,nim))

            for nsa in range(ngrid_1axis):

                idx1 = 0
                theLogSF1 = np.log(np.array(selected_gm_scale_list[nsa]) ** scaling_exponent[idx1])

                theLogIM1 = np.array(IM_log_data_pool[selected_gm_ID_list[nsa], idx1])

                mypoints_tmp = np.vstack([theLogIM1 + theLogSF1]).T
                mypoints = np.vstack([mypoints,mypoints_tmp])

            X = np.linspace(log_im_lb[idx1], log_im_ub[idx1], int(im_nbins[idx1]))
            IM_log_ref = np.vstack([X.reshape(-1)]).T

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.scatter(mypoints[:, 0], 0*mypoints[:, 0],s=18)
            ax.scatter(IM_log_ref[:, 0], 0*IM_log_ref[:, 0],s=5)
            plt.xlabel(im_names[idx1]);

        plt.savefig('res.png')


if __name__ == "__main__":

    errf = open("gridIM_log.err","w")
    try:
        main(sys.argv,errf)
        errf.close()

    except Exception as e:
        print("Exception occurred while code Execution: " + str(repr(e)))
        errf.write("Exception occurred while code Execution: " + str(repr(e)))
        errf.close()
        exit(-1)

