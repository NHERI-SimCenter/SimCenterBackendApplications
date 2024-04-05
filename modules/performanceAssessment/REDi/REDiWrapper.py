# -*- coding: utf-8 -*-
# 
# Copyright (c) 2019 The Regents of the University of California
# Copyright (c) 2019 Leland Stanford Junior University
#
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
# whale. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Stevan Gavrilovic

import json, io, os, sys, time, math, argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from io import StringIO
import zipfile
import pandas as pd

from REDi.go_redi import go_redi

this_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
main_dir = this_dir.parents[1]
sys.path.insert(0, str(main_dir / 'common'))

from simcenter_common import get_scale_factors

class NumpyEncoder(json.JSONEncoder) :
    # Encode the numpy datatypes to json
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_stats(arr : np.array) -> dict : 
    # Returns a dictionary of summary stats from the array

    if np.min(arr) > 0.0:
        log_std = np.std(np.log(arr))
    else:
        log_std = ""
    
    return {
        'mean' : np.mean(arr),
        'std' : np.std(arr),
        'log_std' : log_std,
        'count' : len(arr),
        'min' : np.min(arr),
        'max' : np.max(arr),
        '0.10%' : np.percentile(arr, 0.1),
        '2.3%' : np.percentile(arr, 2.3),
        '10%' : np.percentile(arr, 10),
        '15.9%' : np.percentile(arr, 15.9),
        '50%' : np.percentile(arr, 50),
        '84.1%' : np.percentile(arr, 84.1),
        '90%' : np.percentile(arr, 90),
        '97.7%' : np.percentile(arr, 97.7),
        '99.9%' : np.percentile(arr, 99.9)
    }


def clean_up_results(res : dict, keys_to_remove : List[str]) -> dict :    
    # Remove extra keys not needed here
    
    for key in keys_to_remove:
        if key in res:
            del res[key]

    return res


def clean_up_nistr(nistr : str) -> str :
    # helper function to convert from Pelicun tag to REDi tag
    
    indices_to_remove = [1, 4]

    for index in sorted(indices_to_remove, reverse=True):
        nistr = nistr[:index] + nistr[index+1:]

    return nistr


def get_replacement_response(replacement_time : float) : 

    return {
            'repair_class': 'replacement',
            'damage_by_component_all_DS': None,
            'repair_schedule': 'replacement',
            'component_qty': None,
            'consequence_by_component_by_floor': None,
            'impeding_delays': None,
            'max_delay': 0.0,
            'building_total_downtime': [replacement_time, replacement_time, replacement_time]
            }


def get_first_value(val : dict, num_levels : int) -> int :
    # Get the number of samples that pelicun returns
    
    next_val =next(iter(val.items()))[1]

    if num_levels > 0:
        return get_first_value(val=next_val,num_levels=num_levels-1)
    else :
        return next_val


def main(args):

    print("***Running REDi Seismic Downtime engine***\n")

    pelicun_results_dir = Path(args.dirnameOutput)

    redi_input_dir = pelicun_results_dir/'REDi_input'
    redi_output_dir = pelicun_results_dir/'REDi_output'
    
    
    # Create the directory if it doesn't exist
    if not redi_input_dir.exists():
        redi_input_dir.mkdir(parents=True)
        
        
    # Create the directory if it doesn't exist
    if not redi_output_dir.exists():
            redi_output_dir.mkdir(parents=True)
        
        
    # dictionary to hold the base input parameter that do not change with every pelicun iteration
    rediInputDict = dict()

    # load the risk parameters
    pathRiskParams = Path(args.riskParametersPath)
    with open(pathRiskParams, encoding="utf-8") as f:
        risk_param_dict = json.load(f)

    rediInputDict['risk_parameters']=risk_param_dict

    # import SimCenter's AIM.json file
    pathAim = pelicun_results_dir/'AIM.json'
    with open(pathAim, encoding="utf-8") as f:
        AIM = json.load(f)

    # Get the CMP_sample json from Pelicun
    pathComponent = pelicun_results_dir/'CMP_sample.json'
    with open(pathComponent, encoding="utf-8") as f:
        CMP = json.load(f)

        # remove Units information - for now
        if "Units" in CMP:
            del CMP['Units']

    # Get the DMG_sample json from Pelicun
    pathComponentDmg = pelicun_results_dir/'DMG_sample.json'
    with open(pathComponentDmg, encoding="utf-8") as f:
        CMP_DMG = json.load(f)

        # remove Units information - for now
        if "Units" in CMP_DMG:
            del CMP_DMG['Units']

    # Get the DV_repair_sample json from Pelicun
    pathComponentDV = pelicun_results_dir/'DV_repair_sample.json'
    with open(pathComponentDV, encoding="utf-8") as f:
        CMP_DV = json.load(f)

        # remove Units information - for now
        if "Units" in CMP_DV:
            del CMP_DV['Units']

    # Load the csv version of the decision vars
    with zipfile.ZipFile(pelicun_results_dir/'DV_repair_sample.zip', 'r') as zip_ref:
        # Read the CSV file inside the zip file into memory
        with zip_ref.open('DV_repair_sample.csv') as csv_file:
            # Load the CSV data into a pandas DataFrame
            data = pd.read_csv(io.TextIOWrapper(csv_file, encoding='utf-8'), index_col=0)
            # Drop Units row for now to avoid breaking the code - would be good to use this info in the future
            data = data.drop("Units").astype(float)
            
    # Get the number of samples
    num_samples = data.shape[0]

    # Define a list of keywords to search for in column names
    keywords = ['replacement-collapse', 'replacement-irreparable']
    DVs = ['Cost','Time']

    DVReplacementDict = {}
    for DV in DVs : 
        columns_to_check = [col for col in data.columns if any(f'{DV}-{keyword}' in col for keyword in keywords)]
        # Create a boolean vector indicating whether non-zero values are present in each column
        result_vector = data[columns_to_check].apply(max, axis=1)

        DVReplacementDict[DV] = result_vector
        

    # Find columns containing replace or collapse keywords
    buildingirreparableOrCollapsed = (data[columns_to_check] != 0).any(axis=1)

    sum_collapsed_buildings = sum(buildingirreparableOrCollapsed)
    
    print(f"There are {sum_collapsed_buildings} collapsed or irreparable buildings from Pelicun")

    # Get some general information
    gen_info = AIM['DL']['Asset']

    nStories = int(gen_info['NumberOfStories'])
    rediInputDict['nFloor'] = nStories

    # Get the plan area
    plan_area = float(gen_info['PlanArea'])

    # Get the units
    input_units = {'length': AIM['GeneralInformation']['units']['length']}
    output_units = {'length':'ft'}
    
    # scale the input data to the length unit used internally
    f_scale_units = get_scale_factors(input_units, output_units)['length']

    # Scale the plan area
    plan_area = plan_area*f_scale_units*f_scale_units

    floor_areas = [plan_area for i in range(nStories + 1)]
    rediInputDict['floor_areas'] = floor_areas

    # Get the total building area
    total_building_area = plan_area * nStories

    # Estimate the number of workers
    # PACT provides a default setting of 0.001 which corresponds to one worker per 1000 square feet of floor area. Users should generally execute their assessment with this default value,
    num_workers = max(int(total_building_area/1000), 1)
    
    # Get the replacement cost and time
    DL_info =  AIM['DL']['Losses']['Repair']
    
    # Note these are not the random
    replacementCost = DL_info['ReplacementCost']['Median']
    rediInputDict['replacement_cost'] = float(replacementCost)/1e6 #Needs to be in the millions of dollars

    replacementTime = float(DL_info['ReplacementTime']['Median'])

    # convert replacement time to days from worker_days
    replacementTime = replacementTime / num_workers

    rediInputDict['replacement_time'] = replacementTime

    final_results_dict = dict()
    log_output : List[str] = []

    for sample in range(num_samples) :

        if buildingirreparableOrCollapsed[sample] :
            
            # Convert the replacement time coming out of Pelicun (worker-days) into days by dividing by the number of workers
            replacement_time = DVReplacementDict['Time'][sample] / num_workers
                        
            final_results_dict[sample] = get_replacement_response(replacement_time=replacement_time)
            continue 
        
        ### REDi input map ###
        # Assemble the component quantity vector
        # components object is a list of lists where each item is a list of component on a particular floor.
        # The components are a dictionary containing the component tag (NISTR) and an array of quantities (Qty) in each direction, i.e., [dir_1, dir_2]
        # [floor_1, floor_2, ..., floor_n]
        # where floor_n = [{'NISTR' : nistr_id_1,
        #                   'Qty' : [dir_1, dir_2]},
        #                   ...,
        #                  {'NISTR' : nistr_id_n,
        #                   'Qty' : [dir_1, dir_2]}]
        components : List[List[Dict[str,Any]]]= [[] for i in range(nStories + 1)]

        ### Pelicun output map ###
        #   "B1033.061b": { <- component nistr
        #     "4": {        <- floor
        #       "1": [      <- direction
        CMP = clean_up_results(res=CMP, keys_to_remove = ["collapse", "excessiveRID", "irreparable"])
        for nistr, floors in CMP.items() :

            nistr = clean_up_nistr(nistr=nistr)

            for floor, dirs in floors.items() :

                floor = int(floor)

                dir_1 = 0.0
                dir_2 = 0.0

                # If no directionality, i.e., direction is 0, divide the components evenly in the two directions
                if '0' in dirs :
                    qnty = float(dirs['0'][sample])
                    dir_1 = 0.5 * qnty
                    dir_2 = 0.5 * qnty

                elif '1'  in dirs or '2' in dirs :

                    if '1'  in dirs :
                        dir_1 = float(dirs['1'][sample])

                    if '2'  in dirs :
                        dir_2 = float(dirs['2'][sample])

                else :
                    raise ValueError('Could not parse the directionality in the Pelicun output.')

                cmp_dict = {
                    'NISTR' : nistr,
                    'Qty' : [dir_1, dir_2]
                }
                components[floor-1].append(cmp_dict)


        ### REDi input map ###
        # total_consequences = dict()
        # Assemble the component damage vector
        # component_damage object is a dictionary where each key is a component tag (NISTR) and the values is a list of a list.
        # The highest level, outer list is associated with the number of damage states while the inner list corresponds to the number of floors
        # [ds_1, ds_2, ..., ds_n]
        # where ds_n = [num_dmg_units_floor_1, num_dmg_units_floor_2, ..., num_dmg_units_floor_n]
        component_damage : Dict[str,List[List[float]]] = {}

        ### Pelicun output map ###
        #   "B1033.061b": { <- component nistr
        #     "4": {        <- floor
        #       "1": {      <- direction
        #         "0": [    <- damage state  -> Note that zero.. means undamaged
        CMP_DMG = clean_up_results(res=CMP_DMG, keys_to_remove = ["collapse", "excessiveRID", "irreparable"])
        collapse_flag = False
        for nistr, floors in CMP_DMG.items() :
            
            nistr = clean_up_nistr(nistr=nistr)

            # Get the number of damage states
            num_ds = len(get_first_value(val=floors,num_levels=1))

            floor_qtys = [0.0 for i in range(nStories + 1)]
            ds_qtys = [floor_qtys for i in range(num_ds)]
            
            for floor, dirs in floors.items() :

                floor = int(floor)

                for dir, dir_qty in dirs.items() :

                    for ds, qtys in dir_qty.items() :

                        ds = int(ds)
                        qty = float(qtys[sample])

                        if math.isnan(qty) :
                            log_output.append(f'Collapse detected sample {sample}. Skipping REDi run.\n')
                            collapse_flag = True
                            break

                        # Sum up the damage states
                        ds_qtys[ds][floor-1] += qty

                    if collapse_flag :
                        break

                if collapse_flag :
                    break

            if collapse_flag :
                break

            component_damage[nistr] = ds_qtys

            # total_consequences[nistr] = component_damage

        if collapse_flag :
            continue

        # Assemble the component decision variable vector
        cost_dict = CMP_DV['Cost']
        cost_dict = clean_up_results(res=cost_dict, keys_to_remove = ["replacement"])

        time_dict = CMP_DV['Time']
        time_dict = clean_up_results(res=time_dict, keys_to_remove = ["replacement"])

        ### REDi input map ###
        # Total_consequences is a list of lists of lists.
        # The highest-level list (always length 4) corresponds to the 4 types of consequences at the component level: (1) repair cost [dollars], (2) repair time [worker days], (3) injuries, (4) fatalities.
        # The second level list contains the number of stories, so a list with length 5 will be a 4-story building with a roof.
        # The third-level list is based on the number of damage states (not including Damage State 0).

        total_consequences : Dict[str,List[List[float]]] = {}

        ### Pelicun output map ###
        #   "COST": {           <- cost/time key
        #     "B1033.061b": {   <- component nistr  *special case - this one to evaluate consequences (depends on occupancy type). Note that Component name will match in FEMA P-58 analysis (this case)
        #       "B1033.061b": { <- component nistr  *special case - this one tells you about damage (depends on perhaps location in building or something else). Note that Component name will match in FEMA P-58 analysis (this case)
        #         "1": {        <- damage state
        #           "4": {      <- floor
        #             "1": [    <- direction
        for nistr in cost_dict.keys() :

            # Handle the case of the nested nistr which will be the same for FEMA P-58
            cost_res = cost_dict[nistr][nistr]
            time_res = time_dict[nistr][nistr]

            num_ds = len(cost_res)

            ds_list = np.array([ 0.0 for i in range(num_ds)])
            floor_list = np.array([ ds_list for i in range(nStories+1)])

            cost_floor_list = floor_list.copy()
            time_floor_list = floor_list.copy()

            for ds in cost_res.keys() :

                cost_floor_dict = cost_res[ds]
                time_floor_dict = time_res[ds]

                ds = int(ds)

                for floor in cost_floor_dict.keys() :
                    
                    cost_dirs_dict = cost_floor_dict[floor]
                    time_dirs_dict = time_floor_dict[floor]

                    floor = int(floor)

                    total_cost=0.0
                    total_time=0.0
                    for dir in cost_dirs_dict.keys() :
                        total_cost += float(cost_dirs_dict[dir][sample])
                        total_time += float(time_dirs_dict[dir][sample])

                    cost_floor_list[floor-1][ds-1] = total_cost
                    time_floor_list[floor-1][ds-1] = total_time

            nistr = clean_up_nistr(nistr=nistr)

            # Last two items are empty because pelicun does not return injuries and fatalities.
            total_consequences[nistr] = [cost_floor_list,time_floor_list,floor_list,floor_list]

        # Save the building input file
        this_it_input = rediInputDict

        this_it_input['components']=components
        this_it_input['component_damage']=component_damage
        this_it_input['total_consequences']=total_consequences

        rediInputDict['_id'] = f'SimCenter_{sample}'

        # Save the dictionary to a JSON file
        with open(redi_input_dir/f'redi_{sample}.json', 'w', encoding="utf-8") as f:
            json.dump(this_it_input, f, indent=4, cls=NumpyEncoder)

        # Create a StringIO object to capture the stdout
        captured_output = StringIO()
        
        # Redirect sys.stdout to the captured_output stream
        sys.stdout = captured_output
        
        try:
            res = go_redi(building_dict=this_it_input)
        finally:
            # Reset sys.stdout to the original stdout
            sys.stdout = sys.__stdout__

        # Get the captured output as a string
        output = captured_output.getvalue()
        log_output.append(output)
        captured_output.close()

        final_results_dict[sample] = res


    # Create a high-level json with detailed results
    print(f'Saving all samples to: {redi_output_dir}/redi_results_all_samples.json')
    with open(redi_output_dir/f'redi_results_all_samples.json', 'w', encoding="utf-8") as f:
        json.dump(final_results_dict, f, cls=NumpyEncoder)

    # Create a smaller summary stats json for recovery time and max delay
    dt_all_samples = [[]for i in range(3)]
    max_delay_list = []
    for sample, res in final_results_dict.items() :

        total_downtime = res['building_total_downtime']
        # full recovery - functional recovery - immediate occupancy
        for i in range(3) :
            dt_all_samples[i].append(total_downtime[i])

        max_delay_list.append(res['max_delay'])

    max_delay_list = np.array(max_delay_list)
    full_recovery_list = np.array(dt_all_samples[0])
    functional_recovery_list = np.array(dt_all_samples[1])
    immediate_occupancy_list = np.array(dt_all_samples[2])

    summary_stats = {"Max delay" : get_stats(max_delay_list),
                     "Full Recovery" : get_stats(full_recovery_list),
                     "Functional Recovery" : get_stats(functional_recovery_list),
                     "Immediate Occupancy" : get_stats(immediate_occupancy_list)
                     }

    print(f'Saving all samples to: {redi_output_dir}/redi_summary_stats.json')
    with open(redi_output_dir/f'redi_summary_stats.json', 'w', encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=4, cls=NumpyEncoder)

    # Write the log file
    print(f'Saving REDi log file at: {redi_output_dir}/redi_log.txt')
    with open(redi_output_dir/f'redi_log.txt', 'w', encoding="utf-8") as file:
        # Iterate through the list of strings and write each one to the file
        for string in log_output:
            file.write(string + '\n')



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='REDi-Pelicun Python Package Wrapper')
    parser.add_argument('-w','--dirnameOutput', type=str, default=None, help='Path to the working directory containing the Pelicun results [str]')
    parser.add_argument('-r','--riskParametersPath', type=str, default=None, help='Path to the risk parameters json file [str]')

    args = parser.parse_args()
    
    
    # Check for the required arguments
    if not args.dirnameOutput :
        print("Path to the working directory containing the Pelicun results is required")
        exit()
    else :
        if not Path(args.dirnameOutput).exists() :
            print(f"Provided path to the working directory {args.dirnameOutput} does not exist")
            exit()

    if not args.riskParametersPath :
        print("Path to the risk parameters JSON file is required")
        exit()
    else :
        if not Path(args.riskParametersPath).exists() :
            print(f"Provided path to the risk parameters JSON file {args.riskParametersPath} does not exist")
            exit()

    start_time = time.time()

    main(args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"REDi finished. Elapsed time: {elapsed_time:.2f} seconds")


"/opt/homebrew/anaconda3/envs/simcenter/bin/python" "/Users/stevan.gavrilovic/Desktop/SimCenter/SimCenterBackendApplications/applications/performanceAssessment/REDi/REDiWrapper.py" "--riskParametersPath" "/Users/stevan.gavrilovic/Desktop/SimCenter/build-PBE-Qt_6_5_1_for_macOS-Debug/PBE.app/Contents/MacOS/Examples/pbdl-0003/src/risk_params.json" "--dirnameOutput" "/Users/stevan.gavrilovic/Documents/PBE/LocalWorkDir/tmp.SimCenter" 
