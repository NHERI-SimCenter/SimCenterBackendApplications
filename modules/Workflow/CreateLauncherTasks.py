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
# Adam Zsarn�czay
# Wael Elhaddad
#

import sys, os
import math
import json
import argparse

def generate_workflow_tasks(count, config_file, out_dir, task_size,
                            first_building):

    rWHALE_dir = 'rWHALE'

    #count = int(sys.argv[1])
    #config_file = sys.argv[2]
    #out_dir = sys.argv[3]
    #task_size = int(sys.argv[4])
    #first_building = int(sys.argv[5])

    #pythonDir = '/tmp/{}/python'.format(jobId)

    tasksCount = int(math.ceil(count/task_size))
    last_bldg = count + first_building - 1
    jobId = os.getenv('SLURM_JOB_ID')
    workflowScript = f"/tmp/{rWHALE_dir}/applications/Workflow/RDT_workflow.py"

    # get the type of outputs requested
    with open(config_file, 'r') as f:
        settings = json.load(f)
    output_types = [out_type for out_type, val in settings['outputs'].items()
                    if val==True]

    subfolder = 0
    for i in range(0, tasksCount):

        min_ID = i * task_size + first_building
        max_ID = min((i + 1)*task_size + first_building - 1, last_bldg)

        # do not try to run sims if we are beyond the last bldg
        if min_ID <= last_bldg:

            if (i%500) == 0:
                subfolder = subfolder + 1

            run_dir = (f"/tmp/{rWHALE_dir}"
                       f"/applications/Workflow/RunDir{min_ID}-{max_ID}")

            log_path = (f"{out_dir}/logs/{subfolder}"
                       f"/log{min_ID:07d}-{max_ID:07d}.txt")

            task_list = ""

            # create the subfolder to store log files
            task_list += f'mkdir -p {out_dir}/logs/{subfolder}/ && '

            # run the simulation
            task_list += (f'python3 {workflowScript} {config_file} '
                          f'-Min {min_ID} -Max {max_ID} '
                          f'-d /tmp/{rWHALE_dir}/applications/Workflow/data '
                          f'-w {run_dir} -l {log_path} && ')

            # copy the results from the task for aggregation
            for out_type in output_types:

                if out_type in ['EDP', 'DM', 'DV']:
                    res_type = out_type
                    file_name = f'{res_type}*.csv'
                elif out_type == 'every_realization':
                    res_type = 'realizations'
                    file_name = f'{res_type}*.hd5'

                task_list += (f'mkdir -p {out_dir}/results/{res_type}'
                              f'/{subfolder}/ && ')

                task_list += (f'cp -f {run_dir}/{file_name} {out_dir}/results'
                              f'/{res_type}/{subfolder}/ && ')

            # remove the results after the simulation is completed
            task_list += f"rm -rf {run_dir} \n"

            # write the tasks to the output file
            with open('WorkflowTasks.txt', 'a+') as tasksFile:
                tasksFile.write(task_list)

if __name__ == "__main__":

    #Defining the command line arguments

    workflowArgParser = argparse.ArgumentParser(
        "Create the workflow tasks for rWHALE.")

    workflowArgParser.add_argument("-buildingsCount", "-B", type=int,
        help="Number of buildings to include simulate.")
    workflowArgParser.add_argument("-configFile", "-c", type=str,
        help="The file used to configure the simulation.")
    workflowArgParser.add_argument("-outputDir", "-o", type=str,
        help="The directory to save the final results to.")
    workflowArgParser.add_argument("-buildingsPerTask", "-b", type=int,
        help="Number of buildings to run in each task.")
    workflowArgParser.add_argument("-firstBuilding", "-f", type=int,
        help="The building ID to start counting from.")

    #Parsing the command line arguments
    line_args = workflowArgParser.parse_args()


    main(line_args.threads)
