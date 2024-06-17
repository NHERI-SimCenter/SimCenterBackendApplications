#%%
import os 
import json
from datetime import datetime
import time
from agavepy.agave import Agave

# change the directory to the current directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def Submit_tapis_job():
    ag = Agave.restore()
    with open("TapisFiles/information.json", "r") as file:
        information = json.load(file)	
    file.close()


    # %%
    profile         = ag.profiles.get()
    username        = profile['username']
    savingDirectory = information["directory"]
    if not os.path.exists(savingDirectory):
        os.makedirs(savingDirectory)


    print("Uploading files to designsafe storage")
    ag.files.manage(systemId="designsafe.storage.default", filePath=f"{username}/", body={'action': 'mkdir','path': "physics_based"})
    ag.files.manage(systemId="designsafe.storage.default", filePath=f"{username}/physics_based", body={'action': 'mkdir','path': "Istanbul"})
    # ag.files_mkdir(systemId="designsafe.storage.default", filePath=f"{username}/physics_based/Istanbul2")
    with open("TapisFiles/Istanbul.py", 'rb') as file:
        result = ag.files.importData(filePath= f"{username}/physics_based/Istanbul/",fileToUpload=file,systemId='designsafe.storage.default')
    with open("TapisFiles/information.json", 'rb') as file:
        result = ag.files.importData(filePath= f"{username}/physics_based/Istanbul/",fileToUpload=file,systemId='designsafe.storage.default')
    with open("TapisFiles/selectedSites.csv", 'rb') as file:
        result = ag.files.importData(filePath= f"{username}/physics_based/Istanbul/",fileToUpload=file,systemId='designsafe.storage.default')

    # %%
    jobdict = {
        'name': '',
        'appId': 'physicsBasedMotionApp-0.0.1',
        'nodeCount': 1,
        'processorsPerNode': 1,
        'archive': True,
        'archiveOnAppError':True,
        'inputs': {'inputDirectory': ''},
        'parameters' : {'inputScript':'Istanbul.py'},
        'maxRunTime': '00:01:00',
        'memoryPerNode': '1GB',
        'archiveSystem':'designsafe.storage.default',
    }

    # Generate a timestamp to append to the job name an
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    jobname = f"PhysicsBasedMotion_Istanbul_{username}_{timestamp}"

    print("Submitting job")
    # submit the job 
    jobdict['name'] = jobname
    jobdict['inputs']['inputDirectory'] = f"agave://designsafe.storage.default/{username}/physics_based/Istanbul/"

    # %%
    res = ag.jobs.submit(body=jobdict)
    jobid = res['id']
    status = ""
    last_status = ""
    count = 0
    while status != "FINISHED":
        status = ag.jobs.getStatus(jobId=jobid)['status']
        if count == 0:
            last_status = status
            print("Job status: ", status)
        count += 1
        if last_status != status:
            print("Job status: ", status)
            last_status = status
        if status == "FAILED":
            print("Job failed")
            break

        time.sleep(10)


    # %%
    print("Downloading extracted motions")
    archivePath = ag.jobs.get(jobId=jobid)["archivePath"]
    archivePath = f"{archivePath}/Istanbul/Events/"

    files = ag.files.list(filePath=archivePath, systemId="designsafe.storage.default")
    # %%
    if len(files) <= 1:
        print("No files in the archive")
    else :
        for file in files:
            filename = file['name']
            if filename == ".":
                continue
            path = f"{archivePath}/{filename}"
            res = ag.files.download(filePath=path, systemId="designsafe.storage.default")
            with open(f"{savingDirectory}/{filename}", "wb") as f:
                f.write(res.content)
    # %%
