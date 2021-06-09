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

import numpy as np
import json
import os
import shutil
from glob import glob
import argparse
import pandas as pd

def main(surfaceGMDir):

    originalDirs = glob(f"{surfaceGMDir}/*")

    dirList = glob(f"{surfaceGMDir}/*/*")
    GP_file	= []
    Longitude = []
    Latitude = []

    for siteID,siteDir in enumerate(dirList):


        gmlist = glob(f"{siteDir}/*.json")

        # get the location of this site
        with open(gmlist[0], 'r') as f:
            EVENT_json = json.load(f)
            location = EVENT_json['location']
            Longitude.append(location['longitude'])
            Latitude.append(location['latitude'])

        TH_file = []
        factor = []
        for gm in gmlist:
            newFileName = gm.split('EVENT-')[-1]
            TH_file.append(newFileName[:-5])
            factor.append(1.0)
            shutil.move(gm, f"{surfaceGMDir}/{newFileName}")

        sitedf = pd.DataFrame(list(zip(TH_file, factor)), columns =['TH_file', 'factor'])
        sitedf.to_csv(f"{surfaceGMDir}/site{siteID}.csv", index=False)

        GP_file.append(f"site{siteID}.csv")

    griddf = pd.DataFrame(list(zip(GP_file, Longitude, Latitude)), columns =['GP_file', 'Longitude', 'Latitude'])
    griddf.to_csv(f"{surfaceGMDir}/EventGrid.csv", index=False)

    # remove original files
    for mDir in originalDirs:
        shutil.rmtree(mDir)

    return 0


if __name__ == "__main__":
    #Defining the command line arguments

    workflowArgParser = argparse.ArgumentParser(
        "Create ground motions for BIM.",
        allow_abbrev=False)

    workflowArgParser.add_argument("-d", "--surfaceGMDir",
        default='results/surface_motions',
        help="Dir where the generated surface motions saved.")

    #Parsing the command line arguments
    wfArgs = workflowArgParser.parse_args()

    #Calling the main function 
    main(surfaceGMDir = wfArgs.surfaceGMDir)
