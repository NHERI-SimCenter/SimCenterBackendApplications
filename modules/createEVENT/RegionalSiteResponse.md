# Regional Site Response Workflow

Run the following command to perform regional site response analysis workflow. 

```
python \
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/Workflow/R2DTool_workflow.py \
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/Example-siteresponse/rWHALE_config_srt.json \
-r /Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/Workflow/WorkflowApplications.json \
--referenceDir /Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/Example-siteresponse/input_Data/ \
-w /Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/Example-siteresponse/results/
```

* R2DTool_workflow.py: Workflow main file.
* rWHALE_config_srt.json: Input file defining the regional site response workflow.
* -r: Registry file, which defines all the available applications.
* --referenceDir: Reference directory, where input data will be found.
* -w: Running directory.

***
The workflow command invokes a series of applications:
***

## 1. Create BIM file


#### Definition in the workflow configuration file:

```
    "Building": {
        "Application": "CSV_to_BIM",
        "ApplicationData": {
            "buildingSourceFile": "input_params.csv",
            "filter" : "1-2"
        }
    }
```

* buildingSourceFile: The input csv file contains building info
* filter: The filter specifying the building IDs to be simulated.

#### Workflow-generated execution code:

```
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/venv3.8.7/bin/python \
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/createBIM/CSV_to_BIM/CSV_to_BIM.py \
--buildingFile /Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/Example-siteresponse/results/buildings1-2.json \
--buildingSourceFile /Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/Example-siteresponse/input_Data/input_params.csv \
--filter 1-2 \
--getRV 
```

* --buildingFile: Path to the file that will contain a list of building ids and corresponding BIM filenames
* --buildingSourceFile: Path to the CSV file with the building inventory
* --filter: Filter applied to select a subset of buildings from the inventory
* --getRV: Identifies the preparational stage of the workflow. This app is only used in that stage, so it does not do anything if called without this flag.

#### Outputs:

The above command will generate 
* buildings1-2.json: File that contains a list of building ids and corresponding BIM filenames.
* {ID}-BIM.json: BIM files for each building.

***

## 2. Regional mapping 


#### Definition in the workflow configuration file:

```
    "RegionalMapping": {
        "Application": "NearestNeighborEvents",
        "ApplicationData": {
            "filenameEVENTgrid": "records/EventGrid.csv",
            "samples": 5,
            "neighbors": 4
        }
    }
```

#### Workflow-generated execution code:

```
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/venv3.8.7/bin/python \
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/performRegionalMapping/NearestNeighborEvents/NNE.py \
--buildingFile /Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/Example-siteresponse/results/buildings1-2.json \
--filenameEVENTgrid /Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/Example-siteresponse/input_Data/records/EventGrid.csv \
--samples 5 \
--neighbors 4 
```

* filenameEVENTgrid: Path to file containing location information on each event file
* samples: Number of event samples to assign to each building 
* neighbors: Number of neighboring event sites to sample from

#### Outputs:

The above command will result in updated {ID}-BIM.json with hazard event information added.

***

## 3. Creating files with random variables

### 3.1 Run Event app for RV

RegionalSiteResponse

#### Definition in the workflow configuration file:

```
    "Events": [
        {
            "EventClassification": "Earthquake",
            "Application": "RegionalSiteResponse",
            "ApplicationData": {
                "pathEventData": "records/",
                "mainScript": "FreeField3D_Dry.tcl",
                "modelPath": "model/",
                "ndm": 3
            }
        }
    ]
```

#### Workflow-generated execution code (for each BIM file):

```
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/venv3.8.7/bin/python \
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/createEVENT/siteResponse/RegionalSiteResponse.py \
--filenameBIM 1-BIM.json \
--filenameEVENT EVENT.json \
--pathEventData /Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/Example-siteresponse/input_Data/records/ \
--mainScript FreeField3D_Dry.tcl \
--modelPath /Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/Example-siteresponse/input_Data/model/ \
--ndm 3 \
--getRV 
```

* filenameBIM: BIM file name.
* filenameEVENT: Event file name
* pathEventData: Path to directory containing event data files in SimCenter format.
* mainScript: The name of the main model file.
* modelPath: The location of the OpenSees tcl model files.
* ndm: The number of degrees of freedom in the numerical model.


#### Outputs:

The above command will produce a template file (EVENT.json) for event, which contains the name of random variables.

***

### 3.2 Run Modeling app for RV

#### Definition in the workflow configuration file:

```
    "Modeling": {
        "Application": "OpenSeesPyInput",
        "ApplicationData": {
            "mainScript": "cantilever_light.py",
            "modelPath": "model/",
            "ndm": 3,
            "dofMap": "1,2,3"
        }
    }
```

#### Workflow-generated execution code:

```
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/venv3.8.7/bin/python \
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/createSAM/openSeesPyInput/OpenSeesPyInput.py \
--filenameBIM 1-BIM.json \
--filenameEVENT EVENT.json \
--filenameSAM SAM.json \
--mainScript cantilever_light.py \
--modelPath /Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/Example-siteresponse/input_Data/model/ \
--ndm 3 \
--dofMap 1,2,3 \
--getRV 
```

* modelPath: Defines the location of the OpenSeesPy model files.
* mainScript: Defines the name of the main model file.
* ndm: Defines the number of degrees of freedom in the numerical model.
* dofMap: The workflow assumes X=1, Y=2, Z=3 mapping between directions and degrees of freedom; XY being the horizontal directions.   This input allows you to define an alternative mapping by providing three numbers separated by commas in a string, such as '1, 3, 2' if you want to have Y as the vertical direction.
* columnLine: Defines the ids of the nodes that shall be used for EDP calculations. Only the firsts n+1 nodes will be used for an n-story building.

#### Outputs:

The above command will produce the file SAM.json, which contains the name of random variables.

***

### 3.3 Run EDP app for RV

#### Definition in the workflow configuration file:

```
    "EDP": {
      "Application": "StandardEarthquakeEDP_R",
      "ApplicationData": {}
    }
```

#### Workflow-generated execution code:

```
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/createEDP/standardEarthquakeEDP_R/StandardEarthquakeEDP \
--filenameBIM 1-BIM.json \
--filenameEVENT EVENT.json \
--filenameSAM SAM.json \
--filenameEDP EDP.json \
--getRV 
```

StandardEarthquakeEDP a binary executable. 

#### Outputs:

The above command creates the EDP.json file containing standard EDPs for earthquake hazard events, including peak interstory drift ratio (PID), peak floor acceleration (PFA), peak roof drift (PRD), and peak floor displacement (PFD).

***

### 3.4 Run Simulation app for RV


#### Definition in the workflow configuration file:

```
    "Simulation": {
        "Application": "OpenSeesPy-Simulation",
        "ApplicationData": {}
    }
```

#### Workflow-generated execution code:

```
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/venv3.8.7/bin/python \
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/performSIMULATION/openSeesPy/OpenSeesPySimulation.py \
--filenameBIM 1-BIM.json \
--filenameSAM SAM.json \
--filenameEVENT EVENT.json \
--filenameEDP EDP.json \
--filenameSIM SIM.json \
--getRV 
```

#### Outputs:

The above command does nothing at this stage.

***



## 4. Run the regional simulation

### 4.1 Creating the driver file:

```
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/performUQ/dakota/simCenterDprepro params.in bim.j 1-BIM.json
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/performUQ/dakota/simCenterDprepro params.in sam.j SAM.json
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/performUQ/dakota/simCenterDprepro params.in evt.j EVENT.json
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/performUQ/dakota/simCenterDprepro params.in edp.j EDP.json

/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/venv3.8.7/bin/python \
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/createBIM/CSV_to_BIM/CSV_to_BIM.py \
--buildingFile /Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/Example-siteresponse/results/buildings1-2.json \
--buildingSourceFile /Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/Example-siteresponse/input_Data/input_params.csv \
--filter 1-2 

/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/venv3.8.7/bin/python \
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/createEVENT/siteResponse/RegionalSiteResponse.py \
--filenameBIM 1-BIM.json \
--filenameEVENT EVENT.json \
--pathEventData /Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/Example-siteresponse/input_Data/records/ \
--mainScript FreeField3D_Dry.tcl --modelPath /Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/Example-siteresponse/input_Data/model/ \
--ndm 3 

/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/venv3.8.7/bin/python \
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/createSAM/openSeesPyInput/OpenSeesPyInput.py \
--filenameBIM 1-BIM.json \
--filenameEVENT EVENT.json \
--filenameSAM SAM.json \
--mainScript cantilever_light.py \
--modelPath /Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/Example-siteresponse/input_Data/model/ \
--ndm 3 \
--dofMap 1,2,3 

/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/createEDP/standardEarthquakeEDP_R/StandardEarthquakeEDP \
--filenameBIM 1-BIM.json \
--filenameEVENT EVENT.json \
--filenameSAM SAM.json \
--filenameEDP EDP.json 

/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/venv3.8.7/bin/python \
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/performSIMULATION/openSeesPy/OpenSeesPySimulation.py \
--filenameBIM 1-BIM.json \
--filenameSAM SAM.json \
--filenameEVENT EVENT.json \
--filenameEDP EDP.json \
--filenameSIM SIM.json 

/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/performUQ/dakota/extractEDP \
EDP.json \
results.out \
1-BIM.json \
0  

```



### 4.2 Run site response simulation from Dakota using the above driver

```
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/venv3.8.7/bin/python \
/Users/simcenter/Codes/SimCenter/SimCenterBackendApplications/applications/performUQ/dakota/DakotaFEM.py \
--filenameBIM 1-BIM.json \
--filenameSAM SAM.json \
--filenameEVENT EVENT.json \
--filenameEDP EDP.json \
--filenameSIM SIM.json \
--driverFile driver \
--method LHS \
--samples 5 \
--type UQ \
--concurrency 2 \
--keepSamples True \
--detailedLog True \
--runType run 
```

