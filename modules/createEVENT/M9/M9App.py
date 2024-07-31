# %%
import os
from subprocess import PIPE, run
import json
from datetime import datetime
import time

# change the directory to the current directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))


# %%
# helper function to call the tapis command
def call(command):
    command = command.split()
    command.append('-f')
    command.append('json')
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    result = json.loads(result.stdout)
    return result


# %%
def Submit_tapis_job():
    with open('TapisFiles/information.json', 'r') as file:
        information = json.load(file)
    file.close()

    profile = call('tapis profiles show self')
    username = profile['username']
    email = profile['email']
    savingDirectory = information['directory']

    if not os.path.exists(savingDirectory):
        os.makedirs(savingDirectory)

    print('Uploading files to designsafe storage')
    call(
        f'tapis files mkdir agave://designsafe.storage.default/{username}/  physics_based'
    )
    call(
        f'tapis files mkdir agave://designsafe.storage.default/{username}/physics_based  M9'
    )

    call(
        f'tapis files upload agave://designsafe.storage.default/{username}/physics_based/M9/  TapisFiles/M9.py '
    )
    call(
        f'tapis files upload agave://designsafe.storage.default/{username}/physics_based/M9/  TapisFiles/information.json '
    )
    call(
        f'tapis files upload agave://designsafe.storage.default/{username}/physics_based/M9/  TapisFiles/selectedSites.csv '
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
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    jobname = f'PhysicsBasedMotion_M9_{username}_{timestamp}'

    print('Submitting job')
    jobdict['name'] = jobname
    jobdict['inputs']['inputDirectory'] = (
        f'agave://designsafe.storage.default/{username}/physics_based/M9/'
    )
    jobdict['notifications'][0]['url'] = f'{email}'

    # submit the job
    jobfile = './TapisFiles/job.json'
    json.dump(jobdict, open(jobfile, 'w'), indent=2)
    res = call(f'tapis jobs submit -F {jobfile}')

    # delete the job file
    # os.remove(jobfile)

    res = call(f'tapis jobs search --name eq {jobname}')
    jobid = res[0]['id']
    status = ''
    last_status = ''
    count = 0
    while status != 'FINISHED':
        status = call(f'tapis jobs status {jobid} ')['status']
        if count == 0:
            last_status = status
            print('Job status: ', status)
        count += 1
        if last_status != status:
            print('Job status: ', status)
            last_status = status
        if status == 'FAILED':
            print('Job failed')
            break

        time.sleep(10)
    # # %%

    # # %%
    print('Downloading extracted motions')
    archivePath = call(f'tapis jobs show {jobid}')['archivePath']
    archivePath = f'agave://designsafe.storage.default/{archivePath}/M9'

    files = call(f'tapis files list {archivePath}/Events/')
    if len(files) == 0:
        print('No files in the archive')
    else:
        command = f'tapis files download {archivePath}/Events/ -W {savingDirectory}/'
        command = command.split()
        run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    return res


if __name__ == '__main__':
    Submit_tapis_job()
