import sys
import math
import os

count = int(sys.argv[1])
configFile = sys.argv[2]
outDir = sys.argv[3]
taskSize = int(sys.argv[4])
tasksCount = int(math.ceil(count/taskSize))
jobId = os.getenv('SLURM_JOB_ID')
pythonDir = '/tmp/{}/python'.format(jobId)
firstBuilding = int(sys.argv[5])
workflowScript = "/tmp/rWHALE/applications/Workflow/RDT_workflow.py"

with open('WorkflowTasks.txt', 'w+') as tasksFile:
    subfolder = 0
    for i in range(0, tasksCount):
        if (i%500) == 0:
            subfolder = subfolder + 1
        min = i * taskSize + firstBuilding
        max = (i + 1) * taskSize + firstBuilding - 1
        runDir = "/tmp/rWHALE/applications/Workflow/RunDir{}-{}".format(min,max)
        logPath = "{}/logs/{}/log{}-{}.txt".format(outDir, subfolder, min, max)
        tasksFile.write('mkdir -p {}/logs/{}/ && '.format(outDir, subfolder))
        tasksFile.write('python3 {} {} -Min {} -Max {} -d /tmp/rWHALE/applications/Workflow/data -w {} -l {} && '.format(workflowScript, configFile, min, max, runDir, logPath))
        tasksFile.write('mkdir -p {}/results/DV/{}/ && '.format(outDir, subfolder))
        tasksFile.write('mkdir -p {}/results/DM/{}/ && '.format(outDir, subfolder))
        tasksFile.write('mkdir -p {}/results/EDP/{}/ && '.format(outDir, subfolder))
        tasksFile.write('mkdir -p {}/results/realizations/{}/ && '.format(outDir, subfolder))
        tasksFile.write('cp -f {}/DM*.csv {}/results/DM/{}/ && '.format(runDir, outDir, subfolder))
        tasksFile.write('cp -f {}/DV*.csv {}/results/DV/{}/ && '.format(runDir, outDir, subfolder))
        tasksFile.write('cp -f {}/EDP*.csv {}/results/EDP/{}/ && '.format(runDir, outDir, subfolder))
        tasksFile.write('cp -f {}/realizations*.csv {}/results/realizations/{}/ && '.format(runDir, outDir, subfolder))
        #tasksFile.write('cp -f {}/log.txt {}/logs/{}/log{}-{}.txt && '.format(outDir, subfolder, runDir, outDir, subfolder, min, max))
        tasksFile.write('( cp -f {}/pelicun_log*.txt {}/logs/{} 2> /dev/null || : ) && '.format(runDir, outDir, subfolder))
        #tasksFile.write("(cd {0} && realpath `find {0} -type 'f' -path '*dakota*.out'` --relative-to {0} | cpio -pdm {1}/logs/{2} ) && ".format(runDir, outDir, subfolder))
        tasksFile.write("rm -rf {} \n".format(runDir))


