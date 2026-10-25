#  # noqa: INP001
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
#
# This file is part of pelicun.
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
# pelicun. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay
# fmk - modifictions to allow cases 2 and 3 (see below)

"""
Script to run the pelicun.tools.DL_calculation.main() in a number of ways:


Cases:
1: num DL_Realizations > 1 & num_possible_worlds == 1 (runSingle = false), this is default case

   Runs default behavior, i.e. calls pelicun dl_main() as before and now combines results to DL.json

2: num DL_Realizations == 1 and num_possible_worlds >= 1 (runSingle = true)

   the DL calculation is now run after FEM simulation as part of per sample workflow
   (sample being UQ samples) in forward propogation problem .. whale now includes
   call to this wrapper in sc_driver file .. script running in workdir.x ..
   needs to peform dl calculation with single EDP values and output both a results.out
   (for uq engine) and DL.json (which whale will collect into one DL.json containnig all info)

   functions utilized:
      create_world_running_workflow() - to create valid input file for dl_main(), updates EDP in file
      run_edp_to_csv() - to take EDP results from FEM sim, updated EDP and create response.csv file
      run_dl_calculation() - to run the dl_calculation()
      run_parse_results - to take the results from dl calculations and create a results.out for workflow
      combine_dl_results - places more detailed results in DL.json

3: num DL_Realizations > 1 & num_possible_worlds > 1 (runSingle = false)

   the DL calculation is running with no FEM simulation and we have a number of possible worlds. for
     all possible valid worlds we create input file, run dl_calculation with same response.,csv file for
   all and then smartly pull in results for the problem (round robin for now as that s how global RV
   being oresently handled)

   functions utilized:
      create_world() - to create valid input file for a possible world
      run_dl_calculation() - to run th dl calculation for a world
      combine_sampled_dl_results() - to combine the results from each world_x directory in round robin fashion

   NOTES: user error will cause a failure, e.g. cannot run an FEM simulation in workflow with case 3.

"""

import argparse
import copy
import csv
import json
import shutil
import statistics
import sys
from pathlib import Path

import pandas as pd
from pelicun.tools.DL_calculation import main as dl_main

def run_dl_calculation(dl_args):
    """
    Run pelicun's DL_calculation.main() with a temporary argv override,
    since it reads its arguments from sys.argv rather than accepting a list.
    """
    old_argv = sys.argv
    sys.argv = [old_argv[0], *dl_args]
    try:
        dl_main()
    finally:
        sys.argv = old_argv

def get_num_possible_worlds(input_json):
    
    """
    Open inputfile and check for num_possible_worlds. 

    Returns (num_possible_worlds, input data).
    """

    num_possible_worlds = 1
    
    #
    # open file & check for simple return, no multiple worlds or equal to 1
    #
    
    input_path = Path(input_json)
    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    pws = data.get('possible_world_summary', {})
    if pws:
        num_possible_worlds = int(pws.get('num_possible_worlds', 1))
    else:
        gi = data.get('GeneralInformation', {})
    
        num_possible_worlds = int(gi.get('num_possible_worlds', 1))
        
    return num_possible_worlds, data
    

def create_world_running_workflow(data, num_worlds, input_json, input_edp):
    
    """
    If the AIM describes multiple possible worlds, change
    field given in changing_features to the single value for this realization
    with the realization index beiing taken from the working directory name
    (workdir.X); the world used is X % num_possible_worlds and update SIM_EDP
    data and num_worlds come from get_num_possible_worlds(), so the input
    file is not re-read; it is rewritten to input_json if a world is selected.
    Returns the (possibly updated) input data.
    """

    #
    # simple return, no multiple worlds or equal to 1
    #
    
    if num_worlds <= 1:
        return data

    gi = data.get('GeneralInformation', {})

    #
    # change those attributes given in changing_fields to have single value
    #
    
    changing = gi.get('changing_fields', gi.get('changing_features', []))

    try:
        sample_id = int(Path.cwd().name.split('.')[-1])
    except ValueError:
        print(
            f'WARNING: could not get sample number from {Path.cwd().name}, '
            f'possible world not selected'
        )
        return data
    world = sample_id % num_worlds

    for field in changing:
        values = gi.get(field)
        if isinstance(values, list):
            gi[field] = values[world]

    # file now describes a single world; drop the list of changing features
    # as pelicun auto-populate scripts expect scalar GI values
    gi['num_possible_worlds'] = 1
    gi.pop('changing_features', None)
    gi.pop('changing_fields', None)

    #
    # update SIM_EDP
    #

    input_edp_path = Path(input_edp)
    with open(input_edp_path, encoding='utf-8') as f:
        data_edp = json.load(f)

    file_edps = data_edp.get('EngineeringDemandParameters')

    new_edps = gather_edp(file_edps)
    data['SIM_EDP'] = new_edps
    
    #
    # rewrite the input file for DL calculation
    #
    
    with open(input_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    return data

def run_edp_to_csv(data, results_path, out_path):

    """
    function needed to take the results of FEM simulation (results.out), updated EDP
    and and create the response.csv file needed by pelicun.

    Notes: Code copied and modified from whale.py
    """

    # Vector-valued EDPs (length > 1) get one column per component,
    # e.g. "PFA" with length 3 becomes PFA_1, PFA_2, PFA_3.
    names = []
    for edp in data['SIM_EDP']:
        length = edp.get('length', 1)
        if length == 1:
            names.append(edp['name'])
        else:
            names.extend(f"{edp['name']}_{i + 1}" for i in range(length))

    with open(results_path) as f:
        tokens = f.read().split()

    # results.out sometimes has a leading run-id token before the values.
    if len(tokens) == len(names) + 1:
        tokens = tokens[1:]
    elif len(tokens) != len(names):
        raise ValueError(
            f'Expected {len(names)} values in {results_path}, found {len(tokens)}'
        )

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(names)
        writer.writerow(tokens)

def csv_to_dl_dict(csv_path):

    """
    Read a pelicun output .csv (or .zip containing a .csv) and convert it
    to the same dict structure pelicun writes to the corresponding .json,
    i.e. column names 'a-b' become nested {'a': {'b': [values]}} and a
    trailing empty level ('repair_cost-') is dropped.
    """

    df = pd.read_csv(csv_path, index_col=0)

    # drop units row, if present (case-insensitive like pelicun)
    units_rows = [idx for idx in df.index if str(idx).lower() == 'units']
    df = df.drop(units_rows).astype(float)

    out = {}
    for col in df.columns:
        keys = str(col).split('-')
        if len(keys) > 1 and keys[-1] == '':
            keys = keys[:-1]
        node = out
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = df[col].tolist()

    return out


def load_dl_file(cwd, name):

    """
    Load pelicun output 'name' from cwd, using name.json if it exists,
    otherwise name.csv or name.zip (pelicun --output_format csv).
    """

    json_path = cwd / f'{name}.json'
    if json_path.exists():
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        data.pop('Units', None)
        return data

    for ext in ('.csv', '.zip'):
        csv_path = cwd / f'{name}{ext}'
        if csv_path.exists():
            return csv_to_dl_dict(csv_path)

    raise FileNotFoundError(
        f'No {name}.json, {name}.csv or {name}.zip found in {cwd}'
    )


def load_dl_outputs(cwd=None):

    """
    Read DL_summary and DMG_grp (written by DL_calculation) once, in either
    .json or .csv/.zip format, into dicts matching pelicun's json layout.
    results passed to run_parse_results() and combine_dl_results().
    Returns (dl_summary, dmg_grp, most_likely_critical_damage_state).
    """

    cwd = Path(cwd) if cwd else Path.cwd()

    dl_summary = load_dl_file(cwd, 'DL_summary')
    dmg_grp = load_dl_file(cwd, 'DMG_grp')

    # For each realization take the worst damage state over all component
    # groups (None/NaN groups are skipped), then the most frequent of these
    # (ties averaged).
    # note: most_likely_critical_damage determination from whale/main.py
    most_likely_critical_damage_state = float(
        pd.DataFrame(dmg_grp).max(axis=1).mode().mean()
    )

    return dl_summary, dmg_grp, most_likely_critical_damage_state


def run_parse_results(out_path, dl_summary, most_likely_critical_damage_state):

    """
    function needed for single-run case to produce results.out for workflow
    Reduces DL outputs to a single-line summary
    (repair cost, collapse, most likely critical damage state, irreparable)
    NOTE: order must match the DL EDP list in whale/main.py
    """

    repair_cost = dl_summary['repair_cost'][0] # [0] as only single run
    collapse = dl_summary['collapse'][0]
    irreparable = dl_summary['irreparable'][0]

    with open(out_path, 'w') as f:
        f.write(
            f'{repair_cost} {collapse} '
            f'{most_likely_critical_damage_state} {irreparable}\n'
        )


def combine_dl_results(dl_summary, dv_grp, most_likely_critical_damage_state,
                       cwd=None):

    """
    Merge DL_summary.json and DMG_grp.json contents into a single DL.json,
    the required output of any performDL application in workflow!
    """

    cwd = Path(cwd) if cwd else Path.cwd()

    summary_stats = {}
    for item, values in dl_summary.items():
        if isinstance(values, list) and all(
            isinstance(v, (int, float)) for v in values
        ):
            summary_stats[f'{item}_mean'] = statistics.mean(values)
            summary_stats[f'{item}_stddev'] = statistics.pstdev(values)

    summary_stats['most_likely_critical_damage_state'] = (
        most_likely_critical_damage_state
    )

    combined = {
        'decision_variables': dl_summary,
        'summary_stats': summary_stats,
        'damage_measures': dv_grp,
    }

    out_path = cwd / 'DL.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2)

    print(f'Wrote {out_path}')


def create_world(sc_input, filenameDL, possible_world):

    """
    function used if no FEM to crate a possible_world, no feature varibale is a list
    """
    
    #
    # create a deep copy of input data
    #  & set any feature that changes to the possible_world
    #
    
    new_input = copy.deepcopy(sc_input)
    gi_data = new_input["GeneralInformation"]
    changing_features = gi_data.get(
        'changing_features', gi_data.get('changing_fields', [])
    )
    for feature in changing_features:
        values = gi_data.get(feature)
        if isinstance(values, list):
            gi_data[feature] = values[possible_world]

    # file now describes a single world; drop the list of changing features
    # as pelicun auto-populate scripts expect scalar GI values
    gi_data['num_possible_worlds'] = 1
    gi_data.pop('changing_features', None)
    gi_data.pop('changing_fields', None)

    #
    # write file
    #
    
    with open(filenameDL, 'w', encoding='utf-8') as f:
        json.dump(new_input, f, indent=2)                                   

    
    
def combine_sampled_dl_results(num_possible_world, base_dir):

    """
    Combine the DL results of base_dir/world_X (X = 0..num_possible_world-1)
    into a single base_dir/DL.json. Every world has the same N realizations
    and the combined result also has N: realization r is realization r of
    world_(r % num_possible_world), e.g. with 2 worlds sample 0 from world_0,
    sample 1 from world_1, sample 2 from world_0, ...
    Keys (e.g. damage groups) may differ between worlds; a key missing from
    the world a realization is taken from gets None (null in DL.json) for
    that realization, matching the workdir merge in whale/main.py.

    note: Claude contributed most of this!
    """

    base_dir = Path(base_dir)

    world_summaries = []
    world_dmg_grps = []
    for world in range(num_possible_world):
        dl_summary, dmg_grp, _ = load_dl_outputs(base_dir / f'world_{world}')
        world_summaries.append(dl_summary)
        world_dmg_grps.append(dmg_grp)

    num_realizations = len(next(iter(world_summaries[0].values())))
    for world in range(num_possible_world):
        for values in (*world_summaries[world].values(),
                       *world_dmg_grps[world].values()):
            if len(values) != num_realizations:
                raise ValueError(
                    f'world_{world} has {len(values)} realizations, '
                    f'expected {num_realizations}'
                )

    def round_robin(world_dicts):
        # union of keys over all worlds, keeping first-seen order
        keys = list(dict.fromkeys(k for d in world_dicts for k in d))
        combined = {key: [] for key in keys}
        for r in range(num_realizations):
            world_dict = world_dicts[r % num_possible_world]
            for key in keys:
                values = world_dict.get(key)
                combined[key].append(values[r] if values is not None else None)
        return combined

    dl_summary = round_robin(world_summaries)
    dmg_grp = round_robin(world_dmg_grps)

    # same most_likely_critical_damage_state calc as load_dl_outputs()
    most_likely_critical_damage_state = float(
        pd.DataFrame(dmg_grp).max(axis=1).mode().mean()
    )

    combine_dl_results(
        dl_summary, dmg_grp, most_likely_critical_damage_state, base_dir
    )

# following taken from updateEDP
# response type -> (acronym, key holding the floor)
RESPONSE_TYPES = {
    'max_abs_acceleration': ('PFA', 'floor'),
    'rms_acceleration': ('RMSA', 'floor'),
    'max_drift': ('PID', 'floor2'),
    'residual_disp': ('RD', 'floor'),
    'max_pressure': ('PSP', 'floor2'),
    'max_rel_disp': ('PFD', 'floor'),
    'max_roof_drift': ('PRD', None),  # floor is always "1"
    'peak_wind_gust_speed': ('PWS', 'floor'),
}

# load type -> acronym
LOAD_TYPES = {
    'peak_pressure': 'PP',
    'mean_pressure': 'MP',
    'rms_pressure': 'RP',
    'peak_force': 'PF',
    'mean_force': 'MF',
    'rms_force': 'RF',
}

def scalar_edp(name):
    return {'length': 1, 'type': 'scalar', 'name': name}

def gather_edp(file_edps):
    """Python version of gatherEDP() for one "EngineeringDemandParameters" list."""
    
    edps = []

    for i, event in enumerate(file_edps):
        event_id = str(i + 1)

        # EDP in "responses"
        for edp in event.get('responses', []):
            edp_type = edp['type']
            if edp_type in RESPONSE_TYPES:
                acronym, floor_key = RESPONSE_TYPES[edp_type]
                floor = '1' if floor_key is None else edp.get(floor_key)
                for dof in edp.get('dofs', []):
                    edps.append(
                        scalar_edp(f'{event_id}-{acronym}-{floor}-{dof}')
                    )
            else:
                # not a standard edp, use name as defined by user
                edps.append(scalar_edp(edp_type))

        # EDP in "loads"
        for edp in event.get('loads', []):
            edp_type = edp['type']
            if edp_type in LOAD_TYPES:
                acronym = LOAD_TYPES[edp_type]
                edps.append(scalar_edp(f'{event_id}-{acronym}-{edp.get("name")}'))
            else:
                edps.append(scalar_edp(edp_type))

    return edps

def str2bool(value):
    """Parse an argparse boolean flag (accepts yes/no/true/false/1/0)."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def main():

    #
    # new parser, an fmk change to allow additional args for single_run case
    #
    
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--run_single', type=str2bool, default=False)
    parser.add_argument('--filenameDL', default=None)
    parser.add_argument('--dirnameOutput', default=None)    
    parser.add_argument('--filenameEDP', default='EDP.json')    
    parser.add_argument('--edpResultsOut', default='results.out')
    parser.add_argument('--edpOutputCsv', default='response.csv')

    # 
    # other options (--demandFile, ...) are forwarded to DL_calculation.main() unchanged.
    #
    
    known, dl_args = parser.parse_known_args()

    # open input & get num possible worlds
    num_possible_worlds, sc_input = get_num_possible_worlds(known.filenameDL)
    
    #
    # If no run_single .. run dl_main and exit
    #

    
    if not known.run_single:
        
        if num_possible_worlds == 1:

            #
            # case 1:
            #  origial behaviour
            
            dl_args = ['--filenameDL', known.filenameDL, *dl_args]
            if known.dirnameOutput:
                dl_args = ['--dirnameOutput', known.dirnameOutput, *dl_args]
            run_dl_calculation(dl_args)
            combine_dl_results(*load_dl_outputs())
            
        else:

            #
            # case 3:
            #  foreach possible world create a valid input file & run dl_main()
            #  combine results, 
            #
            
            input_path = Path(known.filenameDL)
            output_root = Path(known.dirnameOutput)
            
            for possible_world in range(num_possible_worlds):

                valid_world_input_filename = output_root / (
                    f'{input_path.stem}_{possible_world}{input_path.suffix}'
                )
                output_dir = output_root / f'world_{possible_world}'
                world_args = ['--filenameDL', str(valid_world_input_filename), *dl_args,
                              '--dirnameOutput', str(output_dir)]
                
                create_world(sc_input, valid_world_input_filename, possible_world)
                run_dl_calculation(world_args)

            # combine all the world results
            combine_sampled_dl_results(num_possible_worlds, output_root)

        return

    #
    # Case 2:
    #   The code below is used when running as part of UQ engines workflow, i.e. we are running 
    #     in a workdir.X directory, the FEM has run and created results.out.
    #     we may need to modify input for particular world, we do create results.csv and run dl_main()
    #     once dl_main() finished we need to proccess results and revise results.out with Dl data
    #     finally we create DL.json (which is used in rwhale/main.py to gather into a top-level DL.json)
    #

    # --filenameDL & --dirnameOutput were consumed by parse_known_args; pass them back
    dl_args = ['--filenameDL', known.filenameDL, *dl_args]
    if known.dirnameOutput:
        dl_args = ['--dirnameOutput', known.dirnameOutput, *dl_args]

    if '--output_format' not in dl_args:
        dl_args += ['--output_format', 'json']

    # create a valid input file for dl_main() for specific world
    sc_input = create_world_running_workflow(
        sc_input, num_possible_worlds, known.filenameDL, known.filenameEDP
    )

    # create the response.csv file needed for dl_main()
    run_edp_to_csv(sc_input, known.edpResultsOut, known.edpOutputCsv)

    # Keep a copy of the raw simulation output before it gets overwritten.
    sim_backup = Path(known.edpResultsOut).with_suffix('.sim')
    shutil.copy(known.edpResultsOut, sim_backup)

    # Run dl_main() & create a results.out for the UQ engine
    run_dl_calculation(dl_args)
    dl_summary, dmg_grp, most_likely = load_dl_outputs()
    run_parse_results(known.edpResultsOut, dl_summary, most_likely)

    # combine DL results in multiple files to DL.json
    combine_dl_results(dl_summary, dmg_grp, most_likely)

if __name__ == '__main__':
    main()
