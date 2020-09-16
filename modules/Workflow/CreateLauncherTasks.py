# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
#
# This file is part of the SimCenter Backend Applications
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# this file. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay
# Wael Elhaddad
# 

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
        logPath = "{}/logs/{}/log{:07d}-{:07d}.txt".format(outDir, subfolder, min, max)
        tasksFile.write('mkdir -p {}/logs/{}/ && '.format(outDir, subfolder))
        tasksFile.write('python3 {} {} -Min {} -Max {} -d /tmp/rWHALE/applications/Workflow/data -w {} -l {} && '.format(workflowScript, configFile, min, max, runDir, logPath))
        tasksFile.write('mkdir -p {}/results/DV/{}/ && '.format(outDir, subfolder))
        tasksFile.write('mkdir -p {}/results/DM/{}/ && '.format(outDir, subfolder))
        tasksFile.write('mkdir -p {}/results/EDP/{}/ && '.format(outDir, subfolder))
        tasksFile.write('mkdir -p {}/results/realizations/{}/ && '.format(outDir, subfolder))
        tasksFile.write('cp -f {}/DM*.csv {}/results/DM/{}/ && '.format(runDir, outDir, subfolder))
        tasksFile.write('cp -f {}/DV*.csv {}/results/DV/{}/ && '.format(runDir, outDir, subfolder))
        tasksFile.write('cp -f {}/EDP*.csv {}/results/EDP/{}/ && '.format(runDir, outDir, subfolder))
        tasksFile.write('cp -f {}/realizations*.hd5 {}/results/realizations/{}/ && '.format(runDir, outDir, subfolder))
        #tasksFile.write('cp -f {}/log.txt {}/logs/{}/log{}-{}.txt && '.format(outDir, subfolder, runDir, outDir, subfolder, min, max))
        tasksFile.write('( cp -f {}/pelicun_log*.txt {}/logs/{} 2> /dev/null || : ) && '.format(runDir, outDir, subfolder))
        #tasksFile.write("(cd {0} && realpath `find {0} -type 'f' -path '*dakota*.out'` --relative-to {0} | cpio -pdm {1}/logs/{2} ) && ".format(runDir, outDir, subfolder))
        tasksFile.write("rm -rf {} \n".format(runDir))


