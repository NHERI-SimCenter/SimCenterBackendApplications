# %%  # noqa: INP001, D100
import os  # noqa: I001
import json
from datetime import datetime
import time
from tapipy.tapis import Tapis
from tapipy import errors


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

    # Poll until job reaches a terminal state; avoid busy-looping.
    tlapse = 2
    previous = None
    while True:
        status = t.jobs.getJobStatus(jobUuid=mjobUuid).status
        if status in ['FINISHED', 'FAILED', 'STOPPED', 'CANCELLED']:
            print(f'\tStatus: {status}')  # noqa: T201
            break
        if status != previous:
            print(f'\tStatus: {status}')  # noqa: T201
            previous = status
        time.sleep(tlapse)

    # %%
    print('Downloading extracted motions')  # noqa: T201
    archiveRoot = t.jobs.getJob(jobUuid=mjobUuid).archiveSystemDir  # noqa: N806
    # The Events directory may be directly under archive root or under M9/Events.
    candidate_paths = [f'{archiveRoot}/M9/Events', f'{archiveRoot}/Events']
    archivePath = None
    files = []
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        for cpath in candidate_paths:
            try:
                files = t.files.listFiles(
                    systemId='designsafe.storage.default', path=cpath
                )
                archivePath = cpath
                break
            except errors.NotFoundError:  # noqa: PERF203
                continue
        if archivePath is not None:
            break
        time.sleep(3)
    if archivePath is None:
        print('Could not locate Events directory in job archive after retries.')  # noqa: T201
        return

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
