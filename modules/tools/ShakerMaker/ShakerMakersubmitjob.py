# %%
from tapipy.tapis import Tapis
import os
import time
import json
import argparse

#get the number of cores from argparser and runtime
parser = argparse.ArgumentParser(description='Submit a job to DesignSafe')
parser.add_argument('--metadata', type=str, help='metadatafile for submitting the job')
parser.add_argument('--tapisfolder', type=str, help='folder to upload the files to')
parser.add_argument('--username', type=str, help='username for DesignSafe')
parser.add_argument('--password', type=str, help='password for DesignSafe')


args = parser.parse_args()
# load the metadata file
with open(args.metadata + "/metadata.json") as json_file:
    metadata = json.load(json_file)

jobinfo       = metadata['jobdata']
maxminutes    = jobinfo['maxruntime']
corespernode  = jobinfo['numCores']
numnodes      = jobinfo['numNodes']
totalcores    = str(corespernode*numnodes)
queue         = jobinfo['queue']


# chane hh:mm:ss to minutes
maxminutes = int(maxminutes.split(":")[0]) * 60 + int(maxminutes.split(":")[1]) + int(maxminutes.split(":")[2]) / 60
maxminutes = int(maxminutes)


systemcodepath   = args.tapisfolder
faultinfo        = metadata['faultdata']
faultfiles       = faultinfo["Faultfilenames"]

filenames = []
for filename in faultfiles:
    # // seperate the filename from the whole path
    filename = os.path.basename(filename)
    filenames.append(filename)
faultinfo["Faultfilenames"] = filenames


# Write the faualt info in the faultinfo folder
with open(args.metadata + "/faultInfo.json", "w") as outfile:
    json.dump(faultinfo, outfile, indent=4)

# write the metadata file
metadata["faultdata_path"] = "";
# delete the faultdata key
del metadata["faultdata"]
del metadata["jobdata"]


with open(args.metadata + "/metadata.json", "w") as outfile:
    json.dump(metadata, outfile, indent=4)

# print("username: ", args.username)
# print("password: ", args.password)
# exit()

print("Connecting to DesignSafe")
t = Tapis(base_url= "https://designsafe.tapis.io",
          username=args.username,
          password=args.password)

# Call to Tokens API to get access token
t.get_tokens()

# %%
# =============================================================================
# uploding files
# =============================================================================
print("Uploading files to DesignSafe")
# create shakermaker directory if it does not exist
codepath = f"{t.username}/Shakermaker"
t.files.mkdir(systemId="designsafe.storage.default", path=codepath)
for filename in os.listdir(systemcodepath):
    filedata = open(f"{systemcodepath}/{filename}", "r").read()
    t.files.insert(systemId="designsafe.storage.default", path=f"{codepath}/{filename}", file=filedata)
for filename in faultfiles:
    filedata = open(filename, "r").read()
    t.files.insert(systemId="designsafe.storage.default", path=f"{codepath}/{os.path.basename(filename)}", file=filedata)
# upload the source time function
t.files.insert(systemId="designsafe.storage.default", path=f"{codepath}/SourceTimeFunction.py", file=open(f"{args.metadata}fault/SourceTimeFunction.py", "r").read())
t.files.insert(systemId="designsafe.storage.default", path=f"{codepath}/metadata.json", file=open(args.metadata + "metadata.json", "r").read())
t.files.insert(systemId="designsafe.storage.default", path=f"{codepath}/faultInfo.json", file=open(args.metadata + "faultInfo.json", "r").read())

# =============================================================================
# submit a tapi job
# =============================================================================
print("Submitting a job to DesignSafe")
day = time.strftime("%Y_%m_%d")
time_ = time.strftime("%H_%M_%S")
# attach the date and time to the username
jobname = f"EEUQ_DesignSafe_{t.username}_{day}_{time_}"
archivepath = f"{t.username}/tapis-jobs-archive/{jobname}"
urls = [
    f"tapis://designsafe.storage.default/{t.username}/Shakermaker/ShakerMakermodel.py",
    f"tapis://designsafe.storage.default/{t.username}/Shakermaker/metadata.json",
    f"tapis://designsafe.storage.default/{t.username}/Shakermaker/faultInfo.json",
    f"tapis://designsafe.storage.default/{t.username}/Shakermaker/SourceTimeFunction.py"
]
for filename in filenames:
    urls.append(f"tapis://designsafe.storage.default/{t.username}/Shakermaker/{filename}")

my_job = {
    "name": jobname,
    "appId": "Shakermaker-app-amnp95",
    "appVersion": "1.0.0",
    "jobType": "BATCH",
    "execSystemId": "frontera",
    "execSystemExecDir":"${JobWorkingDir}/jobs/",
    "execSystemInputDir":"${JobWorkingDir}/inputs/",
    "execSystemOutputDir":"${JobWorkingDir}/inputs/results/",
    "archiveSystemId": "designsafe.storage.default",
    "archiveSystemDir": archivepath,
    "nodeCount": numnodes,
    "coresPerNode": corespernode,
    "maxMinutes": maxminutes,
    "execSystemLogicalQueue": queue,
    "archiveOnAppError": False,
    "fileInputArrays":[{"sourceUrls":urls,"targetDir":"*"}],
    "parameterSet": {
        "schedulerOptions": [{ "arg": "-A DesignSafe-SimCenter" }],
        "envVariables": [
            {"key":"inputFile","value":"ShakerMakermodel.py"},
            {"key":"numProcessors","value":totalcores}
        ]
    }
}
job_info = t.jobs.submitJob(**my_job)
# =============================================================================
# get the job status
# =============================================================================
jobs = t.jobs.getJobList(limit=1, orderBy='lastUpdated(desc),name(asc)', computeTotal=True)
job_info = jobs[0]
uuid = job_info.uuid
# ge the status every 10 seconds
job_info = t.jobs.getJob(jobUuid=uuid)
previous_status  = job_info.status
while job_info.status not in['FINISHED','FAILED', 'CANCELLED','CANCELLED', 'QUEUED', 'RUNNING']:
    job_info = t.jobs.getJob(jobUuid=uuid)
    if job_info.status != previous_status:
        print(f"Job status: {job_info.status}")
        previous_status = job_info.status
    time.sleep(10.0)

if job_info.status == 'FINISHED':
    print("Job finished successfully")
if job_info.status == 'FAILED':
    print("Job failed")
    print("please check the job logs and contact the developers")
if job_info.status == 'CANCELLED':
    print("Job cancelled")
if job_info.status == 'QUEUED':
    print("Job is submitted and is in the queue")
    print("This can take several days according to the queue")
    print("please wait for the job to finish")
    print("you can check the job status through the designsafe portal")
if job_info.status == 'RUNNING':
    print("Job is running")
    print("This can take several hours")
    print("please wait for the job to finish")
    print("you can check the job status through the designsafe portal")
# %%

