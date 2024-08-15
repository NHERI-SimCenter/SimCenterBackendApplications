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

import argparse
import json
import os
import shutil
from glob import glob

import pandas as pd


def createFilesForEventGrid(inputDir, outputDir, removeInputDir):
    if not os.path.isdir(inputDir):
        print(f'input dir: {inputDir} does not exist')
        return 0

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    siteFiles = glob(f'{inputDir}/*BIM.json')

    GP_file = []
    Longitude = []
    Latitude = []
    id = []
    sites = []

    for site in siteFiles:
        with open(site) as f:
            All_json = json.load(f)
            generalInfo = All_json['GeneralInformation']
            Longitude.append(generalInfo['Longitude'])
            Latitude.append(generalInfo['Latitude'])
            siteID = generalInfo['BIM_id']

            id.append(siteID)

            siteFileName = f'Site_{siteID}.csv'
            sites.append(siteFileName)

            workdirs = glob(f'{inputDir}/{siteID}/workdir.*')
            siteEventFiles = []
            siteEventFactors = []

            for workdir in workdirs:
                head, sep, sampleID = workdir.partition('workdir.')
                print(sampleID)

                eventName = f'Event_{siteID}_{sampleID}.json'
                print(eventName)
                shutil.copy(f'{workdir}/fmkEVENT', f'{outputDir}/{eventName}')

                siteEventFiles.append(eventName)
                siteEventFactors.append(1)

            siteDF = pd.DataFrame(
                list(zip(siteEventFiles, siteEventFactors)),
                columns=['TH_file', 'factor'],
            )
            siteDF.to_csv(f'{outputDir}/{siteFileName}', index=False)

    # create the EventFile
    gridDF = pd.DataFrame(
        list(zip(sites, Longitude, Latitude)),
        columns=['GP_file', 'Longitude', 'Latitude'],
    )

    gridDF.to_csv(f'{outputDir}/EventGrid.csv', index=False)

    # remove original files
    if removeInputDir:
        shutil.rmtree(inputDir)

    return 0


if __name__ == '__main__':
    # Defining the command line arguments

    workflowArgParser = argparse.ArgumentParser(
        'Create ground motions for BIM.', allow_abbrev=False
    )

    workflowArgParser.add_argument(
        '-i', '--inputDir', help='Dir containing results of siteResponseWhale.'
    )

    workflowArgParser.add_argument(
        '-o', '--outputDir', help='Dir where results to be stored.'
    )

    workflowArgParser.add_argument('--removeInput', action='store_true')

    # Parsing the command line arguments
    wfArgs = workflowArgParser.parse_args()

    print(wfArgs)
    # Calling the main function
    createFilesForEventGrid(wfArgs.inputDir, wfArgs.outputDir, wfArgs.removeInput)
