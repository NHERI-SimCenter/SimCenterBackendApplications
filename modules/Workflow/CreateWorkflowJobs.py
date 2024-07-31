#  # noqa: INP001, D100
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

import argparse
import json
import math
import os

import numpy as np


def generate_workflow_tasks(  # noqa: ANN201, C901, D103, PLR0912, PLR0915
    bldg_filter,  # noqa: ANN001
    config_file,  # noqa: ANN001
    out_dir,  # noqa: ANN001
    task_size,  # noqa: ANN001
    rWHALE_dir,  # noqa: ANN001, N803
):
    jobId = os.getenv('SLURM_JOB_ID')  # We might need this later  # noqa: N806, F841

    # get the type of outputs requested
    with open(f'{rWHALE_dir}/{config_file}') as f:  # noqa: PTH123
        settings = json.load(f)
    output_types = [
        out_type
        for out_type, val in settings['outputs'].items()
        if val == True  # noqa: E712
    ]

    # KZ@220324: check if regional site response is requested
    run_regional_event = settings['Applications'].get('RegionalEvent', None)
    if (
        run_regional_event
        and run_regional_event.get('Application', None) == 'RegionalSiteResponse'
    ):
        generate_workflow_tasks_regionalsiteresponse(
            '', config_file, out_dir, task_size, rWHALE_dir
        )

    # get the list of buildings requested to run
    if bldg_filter == '':
        # we pull the bldg_filter from the config file
        bldg_filter = settings['Applications']['Building']['ApplicationData'].get(
            'filter', ''
        )

        if bldg_filter == '':
            raise ValueError(  # noqa: TRY003
                'Running a regional simulation on DesignSafe requires either '  # noqa: EM101
                "the 'buildingFilter' parameter to be set for the workflow "
                "application or the 'filter' parameter set for the Building "
                'application in the workflow configuration file. Neither was '
                'provided in the current job. If you want to run every building '
                "in the input file, provide the filter like '#min-#max' where "
                '#min is the id of the first building and #max is the id of the '
                'last building in the inventory.'
            )

    # note: we assume that there are no gaps in the indexes
    bldgs_requested = []
    for bldgs in bldg_filter.split(','):
        if '-' in bldgs:
            bldg_low, bldg_high = bldgs.split('-')
            bldgs_requested += list(range(int(bldg_low), int(bldg_high) + 1))
        else:
            bldgs_requested.append(int(bldgs))

    count = len(bldgs_requested)

    tasksCount = int(math.ceil(count / task_size))  # noqa: N806

    workflowScript = f'/tmp/{rWHALE_dir}/applications/Workflow/rWHALE.py'  # noqa: S108, N806

    subfolder = 0
    for i in range(tasksCount):
        bldg_list = np.array(bldgs_requested[i * task_size : (i + 1) * task_size])

        # do not try to run sims if there are no bldgs to run
        if len(bldg_list) > 0:
            min_ID = bldg_list[0]  # noqa: N806
            max_ID = bldg_list[-1]  # noqa: N806

            max_ids = np.where(np.diff(bldg_list) != 1)[0]
            max_ids = np.append(
                max_ids,
                [
                    len(bldg_list) - 1,
                ],
            ).astype(int)

            min_ids = np.zeros(max_ids.shape, dtype=int)
            min_ids[1:] = max_ids[:-1] + 1

            filter = ''  # noqa: A001
            for i_min, i_max in zip(min_ids, max_ids):
                if i_min == i_max:
                    filter += f',{bldg_list[i_min]}'  # noqa: A001
                else:
                    filter += f',{bldg_list[i_min]}-{bldg_list[i_max]}'  # noqa: A001
            filter = filter[1:]  # to remove the initial comma  # noqa: A001

            if (i % 500) == 0:
                subfolder = subfolder + 1

            run_dir = (
                f'/tmp/{rWHALE_dir}'  # noqa: S108
                f'/applications/Workflow/RunDir{min_ID}-{max_ID}'
            )

            log_path = (
                f'{out_dir}/logs/{subfolder}' f'/log{min_ID:07d}-{max_ID:07d}.txt'
            )

            task_list = ''

            # create the subfolder to store log files
            task_list += f'mkdir -p {out_dir}/logs/{subfolder}/ && '

            # run the simulation
            task_list += (
                f'python3 {workflowScript} '
                f'/tmp/{rWHALE_dir}/{config_file} '  # noqa: S108
                f'-d /tmp/{rWHALE_dir}/input_data '
                f'-w {run_dir} -l {log_path} '
                f'--filter {filter} '
                f'-s parallel && '
            )

            # copy the results from the task for aggregation
            for out_type in output_types:
                res_type = None

                if out_type in ['BIM', 'EDP', 'DM', 'DV']:
                    res_type = out_type
                    file_name = f'{res_type}*.csv'
                elif out_type == 'every_realization':
                    res_type = 'realizations'
                    file_name = f'{res_type}*.hdf'

                if res_type is not None:
                    task_list += (
                        f'mkdir -p {out_dir}/results/{res_type}' f'/{subfolder}/ && '
                    )

                    task_list += (
                        f'cp -f {run_dir}/{file_name} {out_dir}/results'
                        f'/{res_type}/{subfolder}/ && '
                    )

            # remove the results after the simulation is completed
            task_list += f'rm -rf {run_dir} \n'

            # write the tasks to the output file
            with open('WorkflowJobs.txt', 'a+') as tasksFile:  # noqa: PTH123, N806
                tasksFile.write(task_list)


def generate_workflow_tasks_siteresponse(  # noqa: ANN201, D103
    bldg_filter,  # noqa: ANN001
    config_file,  # noqa: ANN001
    out_dir,  # noqa: ANN001
    task_size,  # noqa: ANN001
    rWHALE_dir,  # noqa: ANN001, N803
):
    jobId = os.getenv('SLURM_JOB_ID')  # We might need this later  # noqa: N806, F841

    # get the type of outputs requested
    with open(f'{rWHALE_dir}/{config_file}') as f:  # noqa: PTH123
        settings = json.load(f)
    output_types = [  # noqa: F841
        out_type
        for out_type, val in settings['outputs'].items()
        if val == True  # noqa: E712
    ]

    # get the list of buildings requested to run
    if bldg_filter == '':
        # we pull the bldg_filter from the config file
        bldg_filter = settings['Applications']['Building']['ApplicationData'].get(
            'filter', ''
        )

        if bldg_filter == '':
            raise ValueError(  # noqa: TRY003
                'Running a regional simulation on DesignSafe requires either '  # noqa: EM101
                "the 'buildingFilter' parameter to be set for the workflow "
                "application or the 'filter' parameter set for the Building "
                'application in the workflow configuration file. Neither was '
                'provided in the current job. If you want to run every building '
                "in the input file, provide the filter like '#min-#max' where "
                '#min is the id of the first building and #max is the id of the '
                'last building in the inventory.'
            )

    # note: we assume that there are no gaps in the indexes
    bldgs_requested = []
    for bldgs in bldg_filter.split(','):
        if '-' in bldgs:
            bldg_low, bldg_high = bldgs.split('-')
            bldgs_requested += list(range(int(bldg_low), int(bldg_high) + 1))
        else:
            bldgs_requested.append(int(bldgs))

    count = len(bldgs_requested)

    tasksCount = int(math.ceil(count / task_size))  # noqa: N806

    print(f'tasksCount = {tasksCount}')  # noqa: T201

    workflowScript = (  # noqa: N806
        f'/tmp/{rWHALE_dir}/applications/Workflow/SiteResponse_workflow.py'  # noqa: S108
    )

    subfolder = 0
    for i in range(tasksCount):
        bldg_list = np.array(bldgs_requested[i * task_size : (i + 1) * task_size])

        # do not try to run sims if there are no bldgs to run
        if len(bldg_list) > 0:
            min_ID = bldg_list[0]  # noqa: N806
            max_ID = bldg_list[-1]  # noqa: N806

            max_ids = np.where(np.diff(bldg_list) != 1)[0]
            max_ids = np.append(
                max_ids,
                [
                    len(bldg_list) - 1,
                ],
            ).astype(int)

            min_ids = np.zeros(max_ids.shape, dtype=int)
            min_ids[1:] = max_ids[:-1] + 1

            filter = ''  # noqa: A001
            for i_min, i_max in zip(min_ids, max_ids):
                if i_min == i_max:
                    filter += f',{bldg_list[i_min]}'  # noqa: A001
                else:
                    filter += f',{bldg_list[i_min]}-{bldg_list[i_max]}'  # noqa: A001
            filter = filter[1:]  # to remove the initial comma  # noqa: A001

            if (i % 500) == 0:
                subfolder = subfolder + 1

            run_dir = (
                f'/tmp/{rWHALE_dir}'  # noqa: S108
                f'/applications/Workflow/RunDir{min_ID}-{max_ID}'
            )

            log_path = (
                f'{out_dir}/logs/{subfolder}' f'/log{min_ID:07d}-{max_ID:07d}.txt'
            )

            task_list = ''

            # create the subfolder to store log files
            task_list += f'mkdir -p {out_dir}/logs/{subfolder}/ && '

            # run the simulation
            task_list += (
                f'python3 {workflowScript} '
                f'/tmp/{rWHALE_dir}/{config_file} '  # noqa: S108
                f'-d /tmp/{rWHALE_dir}/input_data '
                f'-w {run_dir} -l {log_path} '
                f'--filter {filter} && '
            )

            # copy the results from the task for aggregation
            file_name = 'surface_motions/*'
            task_list += (
                f'mkdir -p {out_dir}/results/surface_motions' f'/{subfolder}/ && '
            )
            task_list += (
                f'cp -Rf {run_dir}/{file_name} {out_dir}/results'
                f'/surface_motions/{subfolder}/ && '
            )

            task_list += "echo 'cmd generated. Currend dir: '$PWD \n"

            # write the tasks to the output file
            with open('WorkflowJobs_siteResponse.txt', 'a+') as tasksFile:  # noqa: PTH123, N806
                tasksFile.write(task_list)


def generate_workflow_tasks_regionalsiteresponse(  # noqa: ANN201, C901, D103, PLR0912, PLR0915
    site_filter,  # noqa: ANN001
    config_file,  # noqa: ANN001
    out_dir,  # noqa: ANN001
    task_size,  # noqa: ANN001
    rWHALE_dir,  # noqa: ANN001, N803
):
    jobId = os.getenv('SLURM_JOB_ID')  # We might need this later  # noqa: N806, F841

    # KZ@220324: currently only EDP is valid output as it's just soil column response in this step
    output_valid = ['IM']

    # get the type of outputs requested
    with open(f'{rWHALE_dir}/{config_file}') as f:  # noqa: PTH123
        settings = json.load(f)
    output_types = [
        out_type
        for out_type, val in settings['outputs'].items()
        if (val == True and out_type in output_valid)  # noqa: E712
    ]

    # get the list of sites requested to run
    if site_filter == '':
        # we pull the site_filter from the config file
        site_filter = settings['Applications']['RegionalEvent'][
            'ApplicationData'
        ].get('filter', '')

        if site_filter == '':
            raise ValueError(  # noqa: TRY003
                'Running a regional simulation on DesignSafe requires either '  # noqa: EM101
                "the 'buildingFilter' parameter to be set for the workflow "
                "application or the 'filter' parameter set for the Building "
                'application in the workflow configuration file. Neither was '
                'provided in the current job. If you want to run every building '
                "in the input file, provide the filter like '#min-#max' where "
                '#min is the id of the first building and #max is the id of the '
                'last building in the inventory.'
            )

    # note: we assume that there are no gaps in the indexes
    sites_requested = []
    for sites in site_filter.split(','):
        if '-' in sites:
            site_low, site_high = sites.split('-')
            sites_requested += list(range(int(site_low), int(site_high) + 1))
        else:
            sites_requested.append(int(sites))

    count = len(sites_requested)

    tasksCount = int(math.ceil(count / task_size))  # noqa: N806

    workflowScript = f'/tmp/{rWHALE_dir}/applications/Workflow/siteResponseWHALE.py'  # noqa: S108, N806

    subfolder = 0
    for i in range(tasksCount):
        site_list = np.array(sites_requested[i * task_size : (i + 1) * task_size])

        # do not try to run sims if there are no bldgs to run
        if len(site_list) > 0:
            min_ID = site_list[0]  # noqa: N806
            max_ID = site_list[-1]  # noqa: N806

            max_ids = np.where(np.diff(site_list) != 1)[0]
            max_ids = np.append(
                max_ids,
                [
                    len(site_list) - 1,
                ],
            ).astype(int)

            min_ids = np.zeros(max_ids.shape, dtype=int)
            min_ids[1:] = max_ids[:-1] + 1

            filter = ''  # noqa: A001
            for i_min, i_max in zip(min_ids, max_ids):
                if i_min == i_max:
                    filter += f',{site_list[i_min]}'  # noqa: A001
                else:
                    filter += f',{site_list[i_min]}-{site_list[i_max]}'  # noqa: A001
            filter = filter[1:]  # to remove the initial comma  # noqa: A001

            if (i % 500) == 0:
                subfolder = subfolder + 1

            run_dir = (
                f'/tmp/{rWHALE_dir}'  # noqa: S108
                f'/applications/Workflow/RunDirSite{min_ID}-{max_ID}'
            )

            log_path = (
                f'{out_dir}/logs/{subfolder}'
                f'/logSite{min_ID:07d}-{max_ID:07d}.txt'
            )

            task_list = ''

            # create the subfolder to store log files
            task_list += f'mkdir -p {out_dir}/logs/{subfolder}/ && '

            # run the simulation
            task_list += (
                f'python3 {workflowScript} '
                f'-i /tmp/{rWHALE_dir}/{config_file} '
                f'-d /tmp/{rWHALE_dir}/input_data '
                f'-w {run_dir} -l {log_path} '
                f'-o /tmp/{rWHALE_dir}/input_data/siteResponseOutputMotions '
                f'--filter {filter} && '
            )

            # copy the results from the task for aggregation
            for out_type in output_types:
                res_type = None

                if out_type in ['IM', 'BIM', 'EDP', 'DM', 'DV']:
                    res_type = out_type
                    file_name = f'{res_type}*.csv'
                elif out_type == 'every_realization':
                    res_type = 'realizations'
                    file_name = f'{res_type}*.hdf'

                if res_type is not None:
                    task_list += (
                        f'mkdir -p {out_dir}/results/{res_type}' f'/{subfolder}/ && '
                    )

                    task_list += (
                        f'cp -f {run_dir}/{file_name} {out_dir}/results'
                        f'/{res_type}/{subfolder}/ && '
                    )

            # remove the results after the simulation is completed
            task_list += f'rm -rf {run_dir} \n'

            # write the tasks to the output file
            with open('WorkflowJobs_SiteResponse.txt', 'a+') as tasksFile:  # noqa: PTH123, N806
                tasksFile.write(task_list)


if __name__ == '__main__':
    # Defining the command line arguments

    workflowArgParser = argparse.ArgumentParser(  # noqa: N816
        'Create the workflow tasks for rWHALE.'
    )

    workflowArgParser.add_argument(
        '-buildingFilter',
        '-F',
        type=str,
        default='',
        nargs='?',
        const='',
        help='Filter a subset of the buildings to run',
    )
    workflowArgParser.add_argument(
        '-configFile',
        '-c',
        type=str,
        help='The file used to configure the simulation.',
    )
    workflowArgParser.add_argument(
        '-outputDir',
        '-o',
        type=str,
        help='The directory to save the final results to.',
    )
    workflowArgParser.add_argument(
        '-buildingsPerTask',
        '-b',
        type=int,
        help='Number of buildings to run in each task.',
    )
    workflowArgParser.add_argument(
        '-rWHALE_dir',
        '-W',
        type=str,
        default='rWHALE',
        help='The path to the rWHALE files on the compute nodes',
    )
    workflowArgParser.add_argument(
        '-workflowName',
        '-N',
        type=str,
        default='building',
        help='building or siteResponse',
    )

    # Parsing the command line arguments
    line_args = workflowArgParser.parse_args()

    if line_args.workflowName == 'building':
        generate_workflow_tasks(
            line_args.buildingFilter,
            line_args.configFile,
            line_args.outputDir,
            line_args.buildingsPerTask,
            line_args.rWHALE_dir,
        )

    elif line_args.workflowName == 'siteResponse':
        generate_workflow_tasks_siteresponse(
            line_args.buildingFilter,
            line_args.configFile,
            line_args.outputDir,
            line_args.buildingsPerTask,
            line_args.rWHALE_dir,
        )
    else:
        # currently supporting building and siteresponse
        print('-workflowName has to be building or siteResponse')  # noqa: T201
