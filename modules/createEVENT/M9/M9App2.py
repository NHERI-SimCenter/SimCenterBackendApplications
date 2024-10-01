#%%
import os 
import json
from datetime import datetime
import time
from tapipy.tapis import Tapis


# change the directory to the current directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def Submit_tapis_job(username, password):
    t = Tapis(
        base_url="https://designsafe.tapis.io",
        username=username,
        password=password)

    t.get_tokens()

    t.authenticator.list_profiles()

    with open("TapisFiles/information.json", "r") as file:
        information = json.load(file)	
    file.close()

    # %%
    savingDirectory = information["directory"]
    if not os.path.exists(savingDirectory):
        os.makedirs(savingDirectory)


    #print("Uploading files to designsafe storage")
    t.files.mkdir(systemId="designsafe.storage.default", path=f"{t.username}/physics_based")
    t.files.mkdir(systemId="designsafe.storage.default", path=f"{t.username}/physics_based/M9")
    with open("TapisFiles/M9.py", 'rb') as file:
        contents = file.read()
    result = t.files.insert(systemId="designsafe.storage.default", path=f"{t.username}/physics_based/M9/M9.py", file=contents)
    with open("TapisFiles/information.json", 'rb') as file:
       contents = file.read()
    result = t.files.insert(systemId="designsafe.storage.default", path=f"{t.username}/physics_based/M9/information.json", file=contents)
    with open("TapisFiles/selectedSites.csv", 'rb') as file:
        contents = file.read()
    result = t.files.insert(systemId="designsafe.storage.default", path=f"{t.username}/physics_based/M9/selectedSites.csv", file=contents)

    # %%
    # -------------------------------------------------------------------------------
    # Define Inputs
    input_Directory = f"tapis://designsafe.storage.default/{t.username}/physics_based/M9"
    fileInputs = [{"name":"Input Directory","sourceUrl":input_Directory}]

    # -------------------------------------------------------------------------------
    # Define parameterSet
    input_filename = "M9.py"
    input_uri = f"tapis://designsafe.storage.default/{t.username}/physics_based/M9/"
    # parameterSet = {"envVariables": [{"key": "inputScript", "value": input_filename},
    #                                  {"key": "dataDirectory", "value": input_uri}]}

    parameterSet = {"envVariables": [{"key": "inputScript", "value": input_filename},
                                     {"key": "_UserProjects", "value": "4127798437512801810-242ac118-0001-012,PRJ-4603"}]}
    

    jobdict = {
        'name': '',
        'appId': 'SimCenter-DesignSafeVM',
        'appVersion': '0.0.1',
        'execSystemId': 'wma-exec-01',
        'nodeCount': 1,
        'coresPerNode': 1,
        'maxMinutes': 30,
        'archiveOnAppError': True,
        'archiveSystemId': 'designsafe.storage.default',
        'archiveSystemDir': f"{t.username}/physics_based/M9/",
        'fileInputs': fileInputs,
        'parameterSet': parameterSet,
        'tags': ['portalName: DESIGNSAFE']
    }


    # Generate a timestamp to append to the job name an
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    jobname = f"PhysicsBasedMotion_M9_{t.username}_{timestamp}"

    print("Submitting job")
    # submit the job 
    jobdict['name'] = jobname

    # %%
    res = t.jobs.submitJob(**jobdict)
    mjobUuid=res.uuid

    tlapse = 1
    status =t.jobs.getJobStatus(jobUuid=mjobUuid).status
    previous = status
    while True:
        if status in ["FINISHED","FAILED","STOPPED"]:
            break
        status = t.jobs.getJobStatus(jobUuid=mjobUuid).status
        if status == previous:
            continue
        else :
            previous = status
        print(f"\tStatus: {status}")
        time.sleep(tlapse)    

    # %%
    print("Downloading extracted motions")
    archivePath = t.jobs.getJob(jobUuid=mjobUuid).archiveSystemDir
    archivePath = f"{archivePath}/M9/Events"
    files = t.files.listFiles(systemId="designsafe.storage.default", path=archivePath)

   # %%
    if len(files) <= 1:
        print("No files in the archive")
    else :
        for file in files:
            filename = file.name
            if filename == ".":
                continue
            path = f"{archivePath}/{filename}"
            res = t.files.getContents(systemId="designsafe.storage.default", path=path)

            with open(f"{savingDirectory}/{filename}", "w") as f:
                f.write(res.decode("utf-8"))
            f.close()
        print("Files downloaded")
        print("Please check the directory for the extracted motions")

   # %%
