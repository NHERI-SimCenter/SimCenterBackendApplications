# %%  # noqa: INP001, D100
import json
import os
import time
from datetime import datetime
from subprocess import PIPE, run

# change the directory to the current directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))  # noqa: PTH120


# %%
# helper function to call the tapis command
def call(command):  # noqa: ANN001, ANN201, D103
    command = command.split()
    command.append('-f')
    command.append('json')
    result = run(command, stdout=PIPE, stderr=PIPE, text=True, check=False)  # noqa: S603, UP022
    result = json.loads(result.stdout)
    return result  # noqa: RET504


# %%
def Submit_tapis_job():  # noqa: ANN201, N802, D103, PLR0915
    with open('TapisFiles/information.json') as file:  # noqa: PTH123
        information = json.load(file)
    file.close()

    profile = call('tapis profiles show self')
    username = profile['username']
    email = profile['email']
    savingDirectory = information['directory']  # noqa: N806

    if not os.path.exists(savingDirectory):  # noqa: PTH110
        os.makedirs(savingDirectory)  # noqa: PTH103

    print('Uploading files to designsafe storage')  # noqa: T201
    call(
        f'tapis files mkdir agave://designsafe.storage.default/{username}/  physics_based'  # noqa: E501
    )
    call(
        f'tapis files mkdir agave://designsafe.storage.default/{username}/physics_based  M9'  # noqa: E501
    )

    call(
        f'tapis files upload agave://designsafe.storage.default/{username}/physics_based/M9/  TapisFiles/M9.py '  # noqa: E501
    )
    call(
        f'tapis files upload agave://designsafe.storage.default/{username}/physics_based/M9/  TapisFiles/information.json '  # noqa: E501
    )
    call(
        f'tapis files upload agave://designsafe.storage.default/{username}/physics_based/M9/  TapisFiles/selectedSites.csv '  # noqa: E501
    )

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
        'notifications': [{'url': '', 'event': '*'}],
    }

    # Generate a timestamp to append to the job name an
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')  # noqa: DTZ005
    jobname = f'PhysicsBasedMotion_M9_{username}_{timestamp}'

    print('Submitting job')  # noqa: T201
    jobdict['name'] = jobname
    jobdict['inputs']['inputDirectory'] = (
        f'agave://designsafe.storage.default/{username}/physics_based/M9/'
    )
    jobdict['notifications'][0]['url'] = f'{email}'

    # submit the job
    jobfile = './TapisFiles/job.json'
    json.dump(jobdict, open(jobfile, 'w'), indent=2)  # noqa: SIM115, PTH123
    res = call(f'tapis jobs submit -F {jobfile}')

    # delete the job file
    # os.remove(jobfile)  # noqa: ERA001

    res = call(f'tapis jobs search --name eq {jobname}')
    jobid = res[0]['id']
    status = ''
    last_status = ''
    count = 0
    while status != 'FINISHED':
        status = call(f'tapis jobs status {jobid} ')['status']
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
    # # %%

    # # %%
    print('Downloading extracted motions')  # noqa: T201
    archivePath = call(f'tapis jobs show {jobid}')['archivePath']  # noqa: N806
    archivePath = f'agave://designsafe.storage.default/{archivePath}/M9'  # noqa: N806

    files = call(f'tapis files list {archivePath}/Events/')
    if len(files) == 0:
        print('No files in the archive')  # noqa: T201
    else:
        command = f'tapis files download {archivePath}/Events/ -W {savingDirectory}/'
        command = command.split()
        run(command, stdout=PIPE, stderr=PIPE, text=True, check=False)  # noqa: S603, UP022

    return res


if __name__ == '__main__':
    Submit_tapis_job()
