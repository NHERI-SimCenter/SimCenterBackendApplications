{
    "Applications": {
        "EDP": {
            "Application": "StandardEarthquakeEDP",
            "ApplicationData": {
            }
        },
        "Events": [
            {
                "Application": "ExistingPEER_Events",
                "ApplicationData": {
                },
                "EventClassification": "Earthquake",
                "subtype": "PEER NGA Records"
            }
        ],
        "Modeling": {
            "Application": "OpenSeesInput",
            "ApplicationData": {
                "fileName": "MRF_4Story_Concentrated_model.tcl",
                "filePath": "C:/Users/SimCenter/Sangri/build-EE-UQ-Desktop_Qt_5_15_0_MSVC2019_64bit-Debug/Debug/Examples/eeuq-0003/src"
            }
        },
        "Simulation": {
            "Application": "OpenSees-Simulation",
            "ApplicationData": {
            }
        },
        "UQ": {
            "Application": "Dakota-UQ",
            "ApplicationData": {
            }
        }
    },
    "DefaultValues": {
        "driverFile": "driver",
        "edpFiles": [
            "EDP.json"
        ],
        "filenameAIM": "AIM.json",
        "filenameDL": "BIM.json",
        "filenameEDP": "EDP.json",
        "filenameEVENT": "EVENT.json",
        "filenameSAM": "SAM.json",
        "filenameSIM": "SIM.json",
        "rvFiles": [
            "AIM.json",
            "SAM.json",
            "EVENT.json",
            "SIM.json"
        ],
        "workflowInput": "scInput.json",
        "workflowOutput": "EDP.json"
    },
    "EDP": {
        "type": "StandardEarthquakeEDP"
    },
    "Events": [
        {
            "EventClassification": "Earthquake",
            "Events": [
                {
                    "EventClassification": "Earthquake",
                    "Records": [
                        {
                            "dirn": 1,
                            "factor": 1.2771,
                            "fileName": "RSN6_IMPVALL.I_I-ELC180.AT2",
                            "filePath": "C:/Users/SimCenter/AppData/Local/Temp.ihREpp"
                        }
                    ],
                    "name": "PEER-Record-6",
                    "type": "PeerEvent"
                },
                {
                    "EventClassification": "Earthquake",
                    "Records": [
                        {
                            "dirn": 1,
                            "factor": 2.2646,
                            "fileName": "RSN20_NCALIF.FH_H-FRN044.AT2",
                            "filePath": "C:/Users/SimCenter/AppData/Local/Temp.ihREpp"
                        }
                    ],
                    "name": "PEER-Record-20",
                    "type": "PeerEvent"
                },
                {
                    "EventClassification": "Earthquake",
                    "Records": [
                        {
                            "dirn": 1,
                            "factor": 3.2921,
                            "fileName": "RSN30_PARKF_C05085.AT2",
                            "filePath": "C:/Users/SimCenter/AppData/Local/Temp.ihREpp"
                        }
                    ],
                    "name": "PEER-Record-30",
                    "type": "PeerEvent"
                },
                {
                    "EventClassification": "Earthquake",
                    "Records": [
                        {
                            "dirn": 1,
                            "factor": 2.4137,
                            "fileName": "RSN68_SFERN_PEL090.AT2",
                            "filePath": "C:/Users/SimCenter/AppData/Local/Temp.ihREpp"
                        }
                    ],
                    "name": "PEER-Record-68",
                    "type": "PeerEvent"
                },
                {
                    "EventClassification": "Earthquake",
                    "Records": [
                        {
                            "dirn": 1,
                            "factor": 0.4925,
                            "fileName": "RSN77_SFERN_PUL164.AT2",
                            "filePath": "C:/Users/SimCenter/AppData/Local/Temp.ihREpp"
                        }
                    ],
                    "name": "PEER-Record-77",
                    "type": "PeerEvent"
                }
            ],
            "TargetSpectrum": {
                "Sd1": "0.6",
                "Sds": "1.0",
                "SpectrumType": "Design Spectrum (ASCE 7-10)",
                "Tl": "12.0"
            },
            "components": "H1",
            "distanceMax": "50.0",
            "distanceMin": "0.0",
            "distanceRange": false,
            "durationMax": "",
            "durationMin": "",
            "durationRange": false,
            "faultType": "All Types",
            "magnitudeMax": "8.0",
            "magnitudeMin": "5.0",
            "magnitudeRange": false,
            "periodPoints": "0.01,0.05,0.1,0.5,1,5,10.0",
            "pulse": "All",
            "records": "5",
            "scaling": "Single Period",
            "singlePeriod": "1.0",
            "type": "ExistingPEER_Events",
            "vs30Max": "300.0",
            "vs30Min": "150.0",
            "vs30Range": false,
            "weights": "1.0,1.0,1.0,1.0,1.0,1.0,1.0"
        }
    ],
    "GeneralInformation": {
        "NumberOfStories": 3,
        "PlanArea": 14400,
        "StructureType": "RM1",
        "YearBuilt": 1990,
        "depth": 120,
        "height": 39,
        "location": {
            "latitude": 37.8715,
            "longitude": -122.273
        },
        "name": "",
        "planArea": 14400,
        "stories": 3,
        "units": {
            "force": "kips",
            "length": "in",
            "temperature": "C",
            "time": "sec"
        },
        "width": 120
    },
    "Modeling": {
        "centroidNodes": [
            11,
            12,
            13,
            14,
            15
        ],
        "dampingRatio": 0.02,
        "ndf": 3,
        "ndm": 2,
        "randomVar": [
        ],
        "responseNodes": [
            11,
            12,
            13,
            14,
            15
        ],
        "type": "OpenSeesInput"
    },
    "Simulation": {
        "Application": "OpenSees-Simulation",
        "algorithm": "Newton",
        "analysis": "Transient -numSubLevels 2 -numSubSteps 10",
        "convergenceTest": "NormUnbalance 1.0e-2 10",
        "dampingModel": "Rayleigh Damping",
        "fileName": "MRF_4Story_Concentrated_solver.tcl",
        "filePath": "C:/Users/SimCenter/Sangri/build-EE-UQ-Desktop_Qt_5_15_0_MSVC2019_64bit-Debug/Debug/Examples/eeuq-0003/src",
        "firstMode": 1,
        "integration": "Newmark 0.5 0.25",
        "modalRayleighTangentRatio": 0,
        "numModesModal": 1,
        "rayleighTangent": "Initial",
        "secondMode": 2,
        "solver": "Umfpack"
    },
    "UQ": {
        "parallelExecution": true,
        "samplingMethodData": {
            "method": "LHS",
            "samples": 5,
            "seed": 413
        },
        "saveWorkDir": true,
        "uqType": "Forward Propagation"
    },
    "WorkflowType": "Building Simulation",
    "localAppDir": "C:/Users/SimCenter/Sangri/SimCenterBackendApplications",
    "randomVariables": [
    ],
    "remoteAppDir": "C:/Users/SimCenter/Sangri/SimCenterBackendApplications",
    "resultType": "SimCenterUQResultsSampling",
    "runDir": "C:/Users/SimCenter/Documents/EE-UQ/LocalWorkDir/tmp.SimCenter",
    "runType": "runningLocal",
    "summary": [
    ],
    "workingDir": "C:/Users/SimCenter/Documents/EE-UQ/LocalWorkDir"
}
