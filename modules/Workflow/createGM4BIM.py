# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 The Regents of the University of California
#
# This file is part of the RDT Application.
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
# You should have received a copy of the BSD 3-Clause License along with the
# RDT Application. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Chaofeng Wang
# fmk

import numpy as np
import json
import os
import shutil
from glob import glob
import argparse
import pandas as pd
from computeResponseSpectrum import *

def createFilesForEventGrid(inputDir, outputDir, removeInputDir):

    if not os.path.isdir(inputDir):
        print(f"input dir: {inputDir} does not exist")
        return 0

    
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    
    siteFiles = glob(f"{inputDir}/*BIM.json")

    GP_file	= []
    Longitude = []
    Latitude = []
    id = []
    sites = []
    # site im dictionary
    periods = np.array([0.1,0.2,0.3,0.4,0.5,0.75,1,2,3,4,5,7.5,10])
    dict_im = {('type','loc','dir','stat'):[],
               ('PGA',0,1,'median'):[],
               ('PGA',0,1,'beta'):[],
               ('PGA',0,2,'median'):[],
               ('PGA',0,2,'beta'):[],
               ('PGV',0,1,'median'):[],
               ('PGV',0,1,'beta'):[],
               ('PGV',0,2,'median'):[],
               ('PGV',0,2,'beta'):[],
               ('PGD',0,1,'median'):[],
               ('PGD',0,1,'beta'):[],
               ('PGD',0,2,'median'):[],
               ('PGD',0,2,'beta'):[]}
    for Ti in periods:
        dict_im.update({('SA({}s)'.format(Ti),0,1,'median'):[],
                        ('SA({}s)'.format(Ti),0,1,'beta'):[],
                        ('SA({}s)'.format(Ti),0,2,'median'):[],
                        ('SA({}s)'.format(Ti),0,2,'beta'):[]})

    for site in siteFiles:

        with open(site, 'r') as f:

            All_json = json.load(f)
            generalInfo = All_json['GeneralInformation']
            Longitude.append(generalInfo['Longitude'])
            Latitude.append(generalInfo['Latitude'])
            siteID = generalInfo['BIM_id']

            id.append(int(siteID))
            
            siteFileName = f"Site_{siteID}.csv"
            sites.append(siteFileName)
            
            workdirs = glob(f"{inputDir}/{siteID}/workdir.*")
            siteEventFiles = []
            siteEventFactors = []
            siteDF = pd.DataFrame(list(zip(siteEventFiles, siteEventFactors)), columns =['TH_file', 'factor'])
            siteDF.to_csv(f"{outputDir}/{siteFileName}", index=False)

            # initialization
            psa_x = []
            psa_y = []
            pga_x = []
            pga_y = []
            pgv_x = []
            pgv_y = []
            pgd_x = []
            pgd_y = []
                        
            for workdir in workdirs:

                head, sep, sampleID = workdir.partition('workdir.')
                #print(sampleID)

                eventName = f"Event_{siteID}_{sampleID}"
                #print(eventName)
                shutil.copy(f"{workdir}/fmkEVENT", f"{outputDir}/{eventName}.json")

                siteEventFiles.append(eventName)
                siteEventFactors.append(1.0)

                # compute ground motion intensity measures
                with open(f"{outputDir}/{eventName}.json", 'r') as f:
                    cur_gm = json.load(f)
                cur_seismograms = cur_gm['Events'][0]['timeSeries']
                num_seismograms = len(cur_seismograms)
                # im_X and im_Y
                for cur_time_series in cur_seismograms:
                    dt = cur_time_series.get('dT')
                    acc = cur_time_series.get('data')
                    acc_hist = np.array([[dt*x for x in range(len(acc))],acc])
                    # get intensity measure
                    my_response_spectrum_calc = NewmarkBeta(acc, dt, periods, damping=0.05, units='g')
                    tmp, time_series, accel, vel, disp = my_response_spectrum_calc.run()
                    psa = tmp.get('Pseudo-Acceleration')
                    pga = time_series.get('PGA',0.0)
                    pgv = time_series.get('PGV',0.0)
                    pgd = time_series.get('PGD',0.0)
                    print(psa)
                    print(pga)
                    # append
                    if cur_time_series.get('name') == 'accel_X':
                        psa_x.append(psa)
                        pga_x.append(pga)
                        pgv_x.append(pgv)
                        pgd_x.append(pgd)
                    else:
                        psa_y.append(psa)
                        pga_y.append(pga)
                        pgv_y.append(pgv)
                        pgd_y.append(pgd)

            # median and dispersion
            # psa
            if len(psa_x) > 0:
                m_psa_x =  np.exp(np.mean(np.log(psa_x),axis=0))
                s_psa_x = np.std(np.log(psa_x),axis=0)
            else:
                m_psa_x = np.zeros((len(periods),))
                s_psa_x = np.zeros((len(periods),))
            if len(psa_y) > 0:
                m_psa_y =  np.exp(np.mean(np.log(psa_y),axis=0))
                s_psa_y = np.std(np.log(psa_y),axis=0)
            else:
                m_psa_y = np.zeros((len(periods),))
                s_psa_y = np.zeros((len(periods),))
            # pga
            if len(pga_x) > 0:
                m_pga_x =  np.exp(np.mean(np.log(pga_x)))
                s_pga_x =  np.std(np.log(pga_x))
            else:
                m_psa_x = 0.0
                s_pga_x = 0.0
            if len(pga_y) > 0:
                m_pga_y =  np.exp(np.mean(np.log(pga_y)))
                s_pga_y =  np.std(np.log(pga_y))
            else:
                m_psa_y = 0.0
                s_pga_y = 0.0
            # pgv
            if len(pgv_x) > 0:
                m_pgv_x =  np.exp(np.mean(np.log(pgv_x)))
                s_pgv_x =  np.std(np.log(pgv_x))
            else:
                m_pgv_x = 0.0
                s_pgv_x = 0.0
            if len(pgv_y) > 0:
                m_pgv_y =  np.exp(np.mean(np.log(pgv_y)))
                s_pgv_y =  np.std(np.log(pgv_y))
            else:
                m_pgv_y = 0.0
                s_pgv_y = 0.0
            # pgd
            if len(pgd_x) > 0:
                m_pgd_x =  np.exp(np.mean(np.log(pgd_x)))
                s_pgd_x =  np.std(np.log(pgd_x))
            else:
                m_pgd_x = 0.0
                s_pgd_x = 0.0
            if len(pgd_y) > 0:
                m_pgd_y =  np.exp(np.mean(np.log(pgd_y)))
                s_pgd_y =  np.std(np.log(pgd_y))
            else:
                m_pgd_y = 0.0
                s_pgd_y = 0.0
            # add to dictionary
            dict_im[('type','loc','dir','stat')].append(int(siteID))
            # pga
            dict_im[('PGA',0,1,'median')].append(m_pga_x)
            dict_im[('PGA',0,1,'beta')].append(s_pga_x)
            dict_im[('PGA',0,2,'median')].append(m_pga_y)
            dict_im[('PGA',0,2,'beta')].append(s_pga_y)
            # pgv
            dict_im[('PGV',0,1,'median')].append(m_pgv_x)
            dict_im[('PGV',0,1,'beta')].append(s_pgv_x)
            dict_im[('PGV',0,2,'median')].append(m_pgv_y)
            dict_im[('PGV',0,2,'beta')].append(s_pgv_y)
            # pgd
            dict_im[('PGD',0,1,'median')].append(m_pgd_x)
            dict_im[('PGD',0,1,'beta')].append(s_pgd_x)
            dict_im[('PGD',0,2,'median')].append(m_pgd_y)
            dict_im[('PGD',0,2,'beta')].append(s_pgd_y)
            for jj, Ti in enumerate(periods):
                cur_sa = 'SA({}s)'.format(Ti)
                dict_im[(cur_sa,0,1,'median')].append(m_psa_x[jj])
                dict_im[(cur_sa,0,1,'beta')].append(s_psa_x[jj])
                dict_im[(cur_sa,0,2,'median')].append(m_psa_y[jj])
                dict_im[(cur_sa,0,2,'beta')].append(s_psa_y[jj])

    # create pandas
    im_csv_path = os.path.dirname(os.path.dirname(outputDir))
    df_im = pd.DataFrame.from_dict(dict_im)
    df_im.to_csv(os.path.join(im_csv_path,'Results','IM_{}-{}.csv'.format(min(id),max(id))),index=False)


    # create the EventFile
    gridDF = pd.DataFrame(list(zip(sites, Longitude, Latitude)), columns =['GP_file', 'Longitude', 'Latitude'])

    gridDF.to_csv(f"{outputDir}/EventGrid.csv", index=False)
    

    # remove original files
    if removeInputDir:         
        shutil.rmtree(inputDir)
    
    return 0


if __name__ == "__main__":
    #Defining the command line arguments

    workflowArgParser = argparse.ArgumentParser(
        "Create ground motions for BIM.",
        allow_abbrev=False)

    workflowArgParser.add_argument("-i", "--inputDir",
                                   help="Dir containing results of siteResponseWhale.")

    workflowArgParser.add_argument("-o", "--outputDir",
                                   help="Dir where results to be stored.")

    workflowArgParser.add_argument("--removeInput", action='store_true')

    #Parsing the command line arguments
    wfArgs = workflowArgParser.parse_args()

    print(wfArgs)
    #Calling the main function
    createFilesForEventGrid(wfArgs.inputDir, wfArgs.outputDir, wfArgs.removeInput)
    

