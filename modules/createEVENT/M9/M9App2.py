# %%  # noqa: INP001, D100
import os  # noqa: I001
import json
from datetime import datetime
import time
from tapipy.tapis import Tapis


# change the directory to the current directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))  # noqa: PTH120


def Submit_tapis_job(username, password):  # noqa: N802, D103
    t = Tapis(
        base_url='https://designsafe.tapis.io', username=username, password=password
    )

    t.get_tokens()

    t.authenticator.list_profiles()

    with open('TapisFiles/information.json', 'r') as file:  # noqa: PTH123, UP015
        information = json.load(file)
    file.close()

    # %%
    savingDirectory = information['directory']  # noqa: N806
    if not os.path.exists(savingDirectory):  # noqa: PTH110
        os.makedirs(savingDirectory)  # noqa: PTH103

    # print("Uploading files to designsafe storage")
    t.files.mkdir(
        systemId='designsafe.storage.default', path=f'{t.username}/physics_based'
    )
    t.files.mkdir(
        systemId='designsafe.storage.default', path=f'{t.username}/physics_based/M9'
    )
    with open('TapisFiles/M9.py', 'rb') as file:  # noqa: PTH123
        contents = file.read()
    result = t.files.insert(
        systemId='designsafe.storage.default',
        path=f'{t.username}/physics_based/M9/M9.py',
        file=contents,
    )
    with open('TapisFiles/information.json', 'rb') as file:  # noqa: PTH123
        contents = file.read()
    result = t.files.insert(
        systemId='designsafe.storage.default',
        path=f'{t.username}/physics_based/M9/information.json',
        file=contents,
    )
    with open('TapisFiles/selectedSites.csv', 'rb') as file:  # noqa: PTH123
        contents = file.read()
    result = t.files.insert(  # noqa: F841
        systemId='designsafe.storage.default',
        path=f'{t.username}/physics_based/M9/selectedSites.csv',
        file=contents,
    )

    # %%
    # -------------------------------------------------------------------------------
    # Define Inputs
    input_Directory = (  # noqa: N806
        f'tapis://designsafe.storage.default/{t.username}/physics_based/M9'
    )
    fileInputs = [{'name': 'Input Directory', 'sourceUrl': input_Directory}]  # noqa: N806

    # -------------------------------------------------------------------------------
    # Define parameterSet
    input_filename = 'M9.py'
    input_uri = f'tapis://designsafe.storage.default/{t.username}/physics_based/M9/'  # noqa: F841
    # parameterSet = {"envVariables": [{"key": "inputScript", "value": input_filename},
    #                                  {"key": "dataDirectory", "value": input_uri}]}

    parameterSet = {  # noqa: N806
        'envVariables': [
            {'key': 'inputScript', 'value': input_filename},
            {
                'key': '_UserProjects',
                'value': '4127798437512801810-242ac118-0001-012,PRJ-4603',
            },
        ]
    }

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
        'archiveSystemDir': f'{t.username}/physics_based/M9/',
        'fileInputs': fileInputs,
        'parameterSet': parameterSet,
        'tags': ['portalName: DESIGNSAFE'],
    }

    # Generate a timestamp to append to the job name an
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')  # noqa: DTZ005
    jobname = f'PhysicsBasedMotion_M9_{t.username}_{timestamp}'

    print('Submitting job')  # noqa: T201
    # submit the job
    jobdict['name'] = jobname

    # %%
    res = t.jobs.submitJob(**jobdict)
    mjobUuid = res.uuid  # noqa: N806

    tlapse = 1
    status = t.jobs.getJobStatus(jobUuid=mjobUuid).status
    previous = status
    while True:
        if status in ['FINISHED', 'FAILED', 'STOPPED', 'STAGING_INPUTS']:
            break
        status = t.jobs.getJobStatus(jobUuid=mjobUuid).status
        if status == previous:
            continue
        else:  # noqa: RET507
            previous = status
        print(f'\tStatus: {status}')  # noqa: T201
        time.sleep(tlapse)

    # %%
    print('Downloading extracted motions')  # noqa: T201
    archivePath = t.jobs.getJob(jobUuid=mjobUuid).archiveSystemDir  # noqa: N806
    archivePath = f'{archivePath}/M9/Events'  # noqa: N806
    files = t.files.listFiles(
        systemId='designsafe.storage.default', path=archivePath
    )

    # %%
    if len(files) <= 1:
        print('No files in the archive')  # noqa: T201
    else:
        for file in files:
            filename = file.name
            if filename == '.':
                continue
            path = f'{archivePath}/{filename}'
            res = t.files.getContents(
                systemId='designsafe.storage.default', path=path
            )

            with open(f'{savingDirectory}/{filename}', 'w') as f:  # noqa: PTH123
                f.write(res.decode('utf-8'))
            f.close()
        print('Files downloaded')  # noqa: T201
        print('Please check the directory for the extracted motions')  # noqa: T201


# %%
