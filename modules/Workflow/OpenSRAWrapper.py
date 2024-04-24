# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 The Regents of the University of California
#
# This file is a part of SimCenter backend applications.
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
#
# Contributors:
# Dr. Stevan Gavrilovic
#

import argparse, os, json
from typing import Dict, Any
from pathlib import Path


def convertToOSRAFile(inputs : Dict[str, Any]) -> None :

    InfrastructureType = None
    gasInfra = inputs["Applications"]["Assets"]["GasNetwork"]
    
    if 'NaturalGasPipelines' in gasInfra :
        InfrastructureType = "below_ground"
    else :
        raise ValueError(f'The infrastructure type in {gasInfra} is not supported')
    
    # Get the site location params from the inputfile
    natGasPipelines = gasInfra['NaturalGasPipelines']
    natGasPipelinesApplicationData = natGasPipelines['ApplicationData']
    site_loc_params = natGasPipelinesApplicationData['siteLocationParams']
    siteDataFile = natGasPipelinesApplicationData['assetGISFile']
    pathToGISFiles = natGasPipelinesApplicationData['pathToGISFiles']

                            
    event = inputs["Applications"]["RegionalEvent"]
    source_for_im = None
    event_app = event["Application"]
    
    if event_app == 'UserInputShakeMap' :
        
        # Get the folder to the shakemap
        app_data = event["ApplicationData"]
        
        dir = Path(app_data["Directory"])
        
        if not dir.exists() :
            raise ValueError(f'The provided ShakeMap directory {app_data["Directory"]} does not exist')
        
        last_directory = dir.parts[-1]  # This is the path object for the last directory
        prior_directories = dir.parent
        
        source_for_im = {
                        "ShakeMap" : {
                            "Directory": str(prior_directories),
                            "Events": [ str(last_directory)]
                            }
                         }
    
    else :
        raise ValueError(f'The event application {event_app} is not supported')
        
    outpath = Path(inputs["runDir"]) / inputs["commonFileDir"]
        
    # Placeholders for converted data
    converted_json = {
        "General": {
            "AnalysisID" : inputs["Name"],
            "Directory": {
                "Working": inputs["runDir"]
            },
            "OutputFileType": "csv"
        },
        "Infrastructure": {
            "InfrastructureType": InfrastructureType,
            "DataType": "Region_Network",
            "SiteDataFile": siteDataFile,
            "SiteLocationParams": site_loc_params
        },
        "IntensityMeasure": {
            "SourceForIM": source_for_im
        },
        "EngineeringDemandParameter": inputs["Applications"]["DL"]["GasNetwork"]["OpenSRA"]["EngineeringDemandParameter"],
        "DamageMeasure": inputs["Applications"]["DL"]["GasNetwork"]["OpenSRA"]["DamageMeasure"],
        "DecisionVariable": inputs["Applications"]["DL"]["GasNetwork"]["OpenSRA"]["DecisionVariable"],
        "InputParameters": inputs["RV"]["InputParameters"],
        "UserSpecifiedData": {
            "GISDatasets": {
                "Directory": pathToGISFiles
            },
            "CPTParameters": {}
        }
    }
        
    # Save the output
    with open(outpath / "SetupConfig.json", 'w') as file:
        json.dump(converted_json, file, indent=4)
        
    print(f'Saved OpenSRA output file to {outpath}')
                
