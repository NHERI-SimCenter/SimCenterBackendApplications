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

def main(inputArgs):

    curDir = os.path.dirname(__file__)
    gmDataBaseDir = os.path.join(curDir,"gmdata.json")
    inputJsonPath = inputArgs[1]

    with open(inputJsonPath,'r') as f:
        inputJson = json.load(f)

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
    numEQmax = round(ngrid/10) # Maximum number of records from the single earthquake


    #
    # Get grid of IMs - change everything to log-space
    #
    log_im_ub = np.log(im_ub)
    log_im_lb = np.log(im_lb)
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


    if nim_eff<=0:
        # ERROR
        print("number of IMs should be greater than 1")
        exit(-1)
    elif nim_eff ==1:
        X = np.linspace(log_im_lb[0], log_im_ub[0], im_nbins[0])
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
    with open(gmDataBaseDir,'r') as f:
        gmData = json.load(f)

    geomPSA = gmData["geomPSA"]
    periods = gmData["period"]
    numgm = gmData["numgm"]
    geomds = gmData["geomds"]
    eqnameID = gmData["eqnameID"]

    #
    # Define Sa(T_cond)
    #

    # compute scaling factors

    #
    # Compute SaRatio(T_lowbound,T_cond,T_highbound) and Ds575
    #

    IM_log_data_pool = np.zeros((numgm,0))
    scaling_exponent = np.zeros((nim,))
    for ni in range(nim):
        if im_names[ni].startswith("PSA"):
            Sa_T1 = np.zeros((numgm,))
            T_cond = float(im_periods[ni][0])  # central (<= 5.0)
            for ng in range(numgm):
                Sa_T1[ng] = np.interp(T_cond, periods, geomPSA[ng])

            Sa1_pool = Sa_T1[np.newaxis].T
            IM_log_data_pool = np.hstack([IM_log_data_pool, np.log(Sa1_pool)])
            scaling_exponent[ni] = 1

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

        elif im_names[ni]=="DS575":
            ds575_pool = (np.array(geomds))[np.newaxis].T
            IM_log_data_pool = np.hstack([IM_log_data_pool, np.log(ds575_pool)])
            scaling_exponent[ni] = 0

        else:
            print("unrecognized IM name "+im_names[ni])
            exit(-1)

    if id_im_scaling_ancher>=0:
        IM_log_data_scaling_anchor = IM_log_data_pool[:,id_im_scaling_ancher]
        #IM_log_ref_scaling_anchor = IM_log_ref[:,id_im_scaling_ancher]
        IM_log_ref_scaling_anchor = Scaling_ref

        IM_log_data_pool2 = np.delete(IM_log_data_pool, id_im_scaling_ancher, 1)
        IM_log_ref2 = IM_log_ref.copy()
        scaling_exponent2 = np.delete(scaling_exponent, id_im_scaling_ancher)

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

    selected_gm_ID_list =[]
    selected_gm_err_list =[]
    selected_gm_eqID_list =[]
    selected_gm_scale_list =[]
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

        err_mat = distance_matrix(IM_log_data_pool3, IM_log_ref2, p=2) ** 2 + np.tile(penalty_pool,(int(nGridPerIM), 1)).T * sf_penalty

        minerr = np.sort(err_mat,axis=0)
        minerr_tag = np.argsort(err_mat,axis=0)

        selected_gm_ID = []
        selected_gm_err = []
        selected_gm_eqID = []
        selected_gm_scale = []
        for nrep in range(npergrid):
            for ngr in np.random.permutation(int(nGridPerIM)):
                count = 0
                for ng in minerr_tag[:,ngr]:

                    cureqID = eqnameID[ng]
                    cureqID_existnum = np.sum(cureqID == eqnameID[0:ngr])

                    if (selected_gm_ID.count(ng)==0) and (cureqID_existnum<numEQmax):
                        break  # we do not consider this

                    count += 1
                    if ng==minerr_tag[-1,ngr]:
                        print("not enough ground motion to match your criteria")
                        exit(-1)


                selected_gm_ID += [ng]
                selected_gm_err += [minerr[count,ngr]]
                selected_gm_eqID += [cureqID]
                selected_gm_scale += [sf_pool[ng]]

        selected_gm_ID_list += [selected_gm_ID]
        selected_gm_err_list += [selected_gm_err]
        selected_gm_eqID_list += [selected_gm_eqID]
        selected_gm_scale_list += [selected_gm_scale]

    import matplotlib.pyplot as plt

    # nsa =0
    #
    # for nsa in range(5):
    #
    #     idx1 = 0
    #     idx2 = 2
    #
    #     theScaleFactor1 = np.array(selected_gm_scale_list[nsa])**scaling_exponent[idx1]
    #     theScaleFactor2 = np.array(selected_gm_scale_list[nsa])**scaling_exponent[idx2]
    #
    #     theIM1 = np.array(IM_log_data_pool[selected_gm_ID_list[nsa],idx1])
    #     theIM2 = np.array(IM_log_data_pool[selected_gm_ID_list[nsa],idx2])
    #
    #
    #     #plt.plot(theIM1 + np.log(theScaleFactor1),theIM2 + np.log(theScaleFactor2),'x')
    #     #plt.xlabel(im_names[idx1]);plt.ylabel(im_names[idx2]);
    #     #plt.show()
    #
    #     plt.plot(theIM1 + np.log(theScaleFactor1),theIM2 + np.log(theScaleFactor2),'x')
    #     #plt.plot(np.linspace(log_im_lb[myID[0]], log_im_ub[myID[0]], int(im_nbins[myID[1]])),(IM_log_ref_scaling_anchor[nsa]*np.ones((5,1))),'o')
    #     plt.plot((IM_log_ref[:, 0]), (IM_log_ref[:, 1]), 'o')
    #     plt.xlabel(im_names[idx1]);plt.ylabel(im_names[idx2]);
    #
    # plt.show()


    mypoints = np.zeros((nim,))

    for nsa in range(5):

        idx1 = 0
        idx2 = 2
        idx3 = 1
        theLogSF1 = np.log(np.array(selected_gm_scale_list[nsa]) ** scaling_exponent[idx1])
        theLogSF2 = np.log(np.array(selected_gm_scale_list[nsa]) ** scaling_exponent[idx2])
        theLogSF3 = np.log(np.array(selected_gm_scale_list[nsa]) ** scaling_exponent[idx3])

        theLogIM1 = np.array(IM_log_data_pool[selected_gm_ID_list[nsa], idx1])
        theLogIM2 = np.array(IM_log_data_pool[selected_gm_ID_list[nsa], idx2])
        theLogIM3 = np.array(IM_log_data_pool[selected_gm_ID_list[nsa], idx3])

        mypoints_tmp = np.vstack([theLogIM1 + theLogSF1, theLogIM2 + theLogSF2, theLogIM3 + theLogSF3]).T
        mypoints = np.vstack([mypoints,mypoints_tmp])
        plt.plot(mypoints_tmp[:, 0], mypoints_tmp[:, 1], 'x',zorder=nsa)


    LogIMref1 = np.linspace(log_im_lb[idx1], log_im_ub[idx1], int(im_nbins[idx1]))
    LogIMref2 = np.linspace(log_im_lb[idx2], log_im_ub[idx2], int(im_nbins[idx2]))
    X, Y = np.meshgrid(LogIMref2,LogIMref1)
    IM_log_ref = np.vstack([X.reshape(-1), Y.reshape(-1)]).T

    plt.scatter(IM_log_ref[:, 1], IM_log_ref[:, 0],zorder=6)

    plt.xlabel(im_names[idx1]);
    plt.ylabel(im_names[idx2]);
    plt.grid()
    
    plt.show()



    #
    #
    #
    #
    LogIMref1 = np.linspace(log_im_lb[idx1], log_im_ub[idx1], int(im_nbins[idx1]))
    LogIMref2 = np.linspace(log_im_lb[idx2], log_im_ub[idx2], int(im_nbins[idx2]))
    LogIMref3 = np.linspace(log_im_lb[idx3], log_im_ub[idx3], int(im_nbins[idx3]))
    X, Y, Z = np.meshgrid(LogIMref1,LogIMref2,LogIMref3)
    IM_log_ref = np.vstack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).T

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(mypoints[:, 0], mypoints[:, 1], mypoints[:, 2],s=5)
    ax.scatter(IM_log_ref[:, 0], IM_log_ref[:, 1], IM_log_ref[:, 2])
    plt.xlabel(im_names[idx1]);
    plt.ylabel(im_names[idx2]);
    ax.set_zlabel(im_names[idx3]);
    plt.show()



if __name__ == "__main__":
    main(sys.argv)
