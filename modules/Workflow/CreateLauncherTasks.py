import sys
import math

count = int(sys.argv[1])
configFile = sys.argv[2]
outDir = sys.argv[3]
taskSize = int(sys.argv[4])
tasksCount = int(math.ceil(count/taskSize))

with open('WorkflowTasks', 'w+') as tasksFile:
    subfolder = 0
    for i in range(0, tasksCount):
        if (i%500) == 0:
            subfolder = subfolder + 1
        min = i * taskSize + 1
        max = (i + 1) * taskSize
        runDir = "/tmp/rWHALE/applications/Workflow/RunDir{}-{}".format(min,max)
        tasksFile.write('cd /tmp/rWHALE/applications/Workflow && ')
        tasksFile.write('export LD_PRELOAD=/home1/apps/tacc-patches/python_cacher/myopen.so && ')
        tasksFile.write('export LD_LIBRARY_PATH=/tmp/python/bin:$LD_LIBRARY_PATH && ')
        tasksFile.write('source /tmp/python/bin/activate && ')
        tasksFile.write('python3 RDT_workflow.py {} -Min {} -Max {} -d /tmp/rWHALE/applications/Workflow/data -w {} -f && '.format(configFile, min, max, runDir))
        tasksFile.write('mkdir -p {}/results/DV/{}/ && mkdir -p {}/results/DM/{}/ && '.format(outDir, subfolder, outDir, subfolder))
        tasksFile.write('cp -f {}/DM*.csv {}/results/DM/{}/ && '.format(runDir, outDir, subfolder))
        tasksFile.write('cp -f {}/DV*.csv {}/results/DV/{}/ && '.format(runDir, outDir, subfolder))
        tasksFile.write('mkdir -p {}/logs/{}/ && cp -f {}/log.txt {}/logs/{}/log{}-{}.txt && '.format(outDir, subfolder, runDir, outDir, subfolder, min, max))
        tasksFile.write('cp -f {}/pelicun_log*.txt {}/logs/{}\n'.format(runDir, outDir, subfolder))

