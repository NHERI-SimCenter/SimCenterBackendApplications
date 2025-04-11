import os  
import json
import time
from datetime import datetime
from tapipy.tapis import Tapis

# change the directory to the current directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))  


def Submit_tapis_job(username=None, password=None):  
    # Initialize Tapis client with credentials
    if username is not None and password is not None:
        t = Tapis(
            base_url='https://designsafe.tapis.io', username=username, password=password
        )
        t.get_tokens()
    else:
        # Use existing authentication if no credentials provided
        t = Tapis(base_url='https://designsafe.tapis.io')
        t.get_tokens()

    with open('TapisFiles/information.json') as file: 
        information = json.load(file)
    file.close()

    savingDirectory = information['directory']  
    if not os.path.exists(savingDirectory): 
        os.makedirs(savingDirectory)  

    print('Uploading files to designsafe storage')  
    try:
        t.files.mkdir(
            systemId='designsafe.storage.default', path=f'{t.username}/physics_based'
        )
    except:  
        # Directory might already exist
        pass
    
    try:
        t.files.mkdir(
            systemId='designsafe.storage.default', path=f'{t.username}/physics_based/Istanbul'
        )
    except:  
        # Directory might already exist
        pass

    # Upload Istanbul.py
    with open('TapisFiles/Istanbul.py', 'rb') as file:  
        contents = file.read()
    result = t.files.insert(
        systemId='designsafe.storage.default',
        path=f'{t.username}/physics_based/Istanbul/Istanbul.py',
        file=contents,
    )

    # Upload information.json
    with open('TapisFiles/information.json', 'rb') as file:  
        contents = file.read()
    result = t.files.insert(
        systemId='designsafe.storage.default',
        path=f'{t.username}/physics_based/Istanbul/information.json',
        file=contents,
    )

    # Upload selectedSites.csv
    with open('TapisFiles/selectedSites.csv', 'rb') as file:  
        contents = file.read()
    result = t.files.insert(  
        systemId='designsafe.storage.default',
        path=f'{t.username}/physics_based/Istanbul/selectedSites.csv',
        file=contents,
    )

    # Define inputs
    input_Directory = (  
        f'tapis://designsafe.storage.default/{t.username}/physics_based/Istanbul'
    )
    fileInputs = [{'name': 'Input Directory', 'sourceUrl': input_Directory}] 

    # Define parameterSet
    input_filename = 'Istanbul.py'
    parameterSet = {  
        'envVariables': [
            {'key': 'inputScript', 'value': input_filename},
        ]
    }

    # Generate a timestamp to append to the job name
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')  
    jobname = f'PhysicsBasedMotion_Istanbul_{t.username}_{timestamp}'

    jobdict = {
        'name': jobname,
        'appId': 'SimCenter-DesignSafeVM',
        'appVersion': '0.0.1',
        'execSystemId': 'wma-exec-01',
        'nodeCount': 1,
        'coresPerNode': 1,
        'maxMinutes': 30,
        'archiveOnAppError': True,
        'archiveSystemId': 'designsafe.storage.default',
        'archiveSystemDir': f'{t.username}/physics_based/Istanbul/',
        'fileInputs': fileInputs,
        'parameterSet': parameterSet,
        'tags': ['portalName: DESIGNSAFE'],
    }

    print('Submitting job')  
    res = t.jobs.submitJob(**jobdict)
    job_uuid = res.uuid

    # Monitor job status
    tlapse = 10
    status = t.jobs.getJobStatus(jobUuid=job_uuid).status
    previous = status
    print(f'Job status: {status}') 
    
    while True:
        if status in ['FINISHED', 'FAILED', 'STOPPED', 'CANCELLED']:
            break
            
        time.sleep(tlapse)
        status = t.jobs.getJobStatus(jobUuid=job_uuid).status
        
        if status != previous:
            print(f'Job status: {status}') 
            previous = status

    # Download results
    print('Downloading extracted motions')  
    archivePath = t.jobs.getJob(jobUuid=job_uuid).archiveSystemDir  
    archivePath = f'{archivePath}/Istanbul/Events'  
    
    try:
        files = t.files.listFiles(
            systemId='designsafe.storage.default', path=archivePath
        )

        if len(files) <= 1:
            print('No files in the archive') 
        else:
            for file in files:
                filename = file.name
                if filename == '.':
                    continue
                    
                path = f'{archivePath}/{filename}'
                res = t.files.getContents(
                    systemId='designsafe.storage.default', path=path
                )

                with open(f'{savingDirectory}/{filename}', 'w') as f:  
                    f.write(res.decode('utf-8'))
                
            print('Files downloaded') 
            print('Please check the directory for the extracted motions')  
    except Exception as e:
        print(f'Error downloading files: {e}')  

    return res