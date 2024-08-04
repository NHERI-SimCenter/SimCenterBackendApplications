# %%  # noqa: CPY001, D100, INP001
import json
import os
import time
from datetime import datetime

from agavepy.agave import Agave

# change the directory to the current directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))  # noqa: PTH120


def Submit_tapis_job():  # noqa: N802, D103
    ag = Agave.restore()
    with open('TapisFiles/information.json') as file:  # noqa: PLW1514, PTH123
        information = json.load(file)
    file.close()

    # %%
    profile = ag.profiles.get()
    username = profile['username']
    savingDirectory = information['directory']  # noqa: N806
    if not os.path.exists(savingDirectory):  # noqa: PTH110
        os.makedirs(savingDirectory)  # noqa: PTH103

    print('Uploading files to designsafe storage')  # noqa: T201
    ag.files.manage(
        systemId='designsafe.storage.default',
        filePath=f'{username}/',
        body={'action': 'mkdir', 'path': 'physics_based'},
    )
    ag.files.manage(
        systemId='designsafe.storage.default',
        filePath=f'{username}/physics_based',
        body={'action': 'mkdir', 'path': 'M9'},
    )
    # ag.files_mkdir(systemId="designsafe.storage.default", filePath=f"{username}/physics_based/Istanbul2")
    with open('TapisFiles/M9.py', 'rb') as file:  # noqa: PTH123
        result = ag.files.importData(
            filePath=f'{username}/physics_based/M9/',
            fileToUpload=file,
            systemId='designsafe.storage.default',
        )
    with open('TapisFiles/information.json', 'rb') as file:  # noqa: PTH123
        result = ag.files.importData(
            filePath=f'{username}/physics_based/M9/',
            fileToUpload=file,
            systemId='designsafe.storage.default',
        )
    with open('TapisFiles/selectedSites.csv', 'rb') as file:  # noqa: PTH123
        result = ag.files.importData(  # noqa: F841
            filePath=f'{username}/physics_based/M9/',
            fileToUpload=file,
            systemId='designsafe.storage.default',
        )

    # %%
    jobdict = {
        'name': '',
        'appId': 'physicsBasedMotionApp-0.0.1',
        'nodeCount': 1,
        'processorsPerNode': 1,
        'archive': True,
        'archiveOnAppError': True,
        'inputs': {'inputDirectory': ''},
        'parameters': {'inputScript': 'M9.py'},
        'maxRunTime': '00:01:00',
        'memoryPerNode': '1GB',
        'archiveSystem': 'designsafe.storage.default',
    }

    # Generate a timestamp to append to the job name an
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')  # noqa: DTZ005
    jobname = f'PhysicsBasedMotion_M9_{username}_{timestamp}'

    print('Submitting job')  # noqa: T201
    # submit the job
    jobdict['name'] = jobname
    jobdict['inputs']['inputDirectory'] = (
        f'agave://designsafe.storage.default/{username}/physics_based/M9/'
    )

    # %%
    res = ag.jobs.submit(body=jobdict)
    jobid = res['id']
    status = ''
    last_status = ''
    count = 0
    while status != 'FINISHED':
        status = ag.jobs.getStatus(jobId=jobid)['status']
        if count == 0:
            last_status = status
            print('Job status: ', status)  # noqa: T201
        count += 1
        if last_status != status:
            print('Job status: ', status)  # noqa: T201
            last_status = status
        if status == 'FAILED':
            print('Job failed')  # noqa: T201
            break

        time.sleep(10)

    # %%
    print('Downloading extracted motions')  # noqa: T201
    archivePath = ag.jobs.get(jobId=jobid)['archivePath']  # noqa: N806
    archivePath = f'{archivePath}/M9/Events/'  # noqa: N806

    files = ag.files.list(
        filePath=archivePath, systemId='designsafe.storage.default'
    )
    # %%
    if len(files) <= 1:
        print('No files in the archive')  # noqa: T201
    else:
        for file in files:
            filename = file['name']
            if filename == '.':
                continue
            path = f'{archivePath}/{filename}'
            res = ag.files.download(
                filePath=path, systemId='designsafe.storage.default'
            )
            with open(f'{savingDirectory}/{filename}', 'wb') as f:  # noqa: FURB103, PTH123
                f.write(res.content)
    # %%
