# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 09:37:26 2024

@author: snaeimi
This is the main file for preprocessing data from 
"""

import os
import argparse
import random
import string
from pathlib import Path
import pandas as pd

import damage_convertor
import preprocessorIO

try:
    import REWET
except:
    # This is only for now
    import sys
    sys.path.insert(1, "C:\\Users\\naeim\\Desktop\\REWET")
    import Input.Settings as Settings
from initial import Starter

def createScnearioList(run_directory, scn_number):
    
    damage_input_dir = os.path.join(run_directory, "Results", "WaterDistributionNetwork",
                                    "damage_input")
    
    if not os.path.exists(damage_input_dir):
        os.makedirs(damage_input_dir)
        
    # REWET_input_data["damage_input_dir"] = damage_input_dir
    
    prefix = chooseARandomPreefix(damage_input_dir)
    
    scenario_name = f"{prefix}_scn_{scn_number}"
    pipe_file_name = f"{prefix}_pipe_{scn_number}"
    node_file_name = f"{prefix}_node_{scn_number}"
    pump_file_name = f"{prefix}_pump_{scn_number}"
    tank_file_name = f"{prefix}_tank_{scn_number}"
    
    
    scenario = {"Scenario Name": scenario_name,
        "Pipe Damage":	pipe_file_name,
        "Nodal Damage": node_file_name,
        "Pump Damage":	pump_file_name,
        "Tank Damage": tank_file_name,	
        "Probability": 1
        }
    
    scenario_list = scenario #pd.DataFrame([scenario]).set_index("Scenario Name")
    
    #REWET_input_data["scenario_list"] = scenario_list
    
    return scenario_list, prefix
    
def chooseARandomPreefix(damage_input_dir):
    """
    Choses a random prefix for sceranio and pipe, node, pump and tank damage
    file. The is important to find and unused prefix so if this script is being
    ran in parallel, then files are not being overwritten.

    Parameters
    ----------
    damage_input_dir : path
        The path to the damage input files directory.

    Returns
    -------
    random_prefix : str
        The Chosen random prefix string.

    """
    
    number_of_prefix = 4
    dir_list = os.listdir(damage_input_dir)
    
    prefix_dir_list = [dir_name for dir_name in dir_list if dir_name.find("_") > 0]
    prefix_list = [dir_name.split("_")[0] for dir_name in prefix_dir_list]
    
    random_prefix = random.choices(string.ascii_letters, k=number_of_prefix)
    s = ""
    for letter in random_prefix:
        s = s + letter
    random_prefix = s
    
    while random_prefix in prefix_list:
        random_prefix = random.choices(string.ascii_letters, k=number_of_prefix)
        s = ""
        for letter in random_prefix:
            s = s + letter
        random_prefix = s

    return random_prefix

#def setSettingsData(rwhale_data, REWET_input_data):
    """
    Sets the settings (future project file) for REWET. REWET input data
    dictionary is both used as a source and destination for settinsg data. The
    data is stored in REWET_input_data object with key='settings'. The value
    for 'settinsg' is an object of REWET's 'Settings' class. 

    Parameters
    ----------
    rwhale_data : json dict
        rwhale input file.
    REWET_input_data : dict
        REWET input data.

    Returns
    -------
    None.

    """


def getDLFileName(run_dir, dl_file_path, scn_number):
    """
    If dl_file_path is not given, the path is acquired from rwhale input data.

    Parameters
    ----------
    REWET_input_data : TYPE
        DESCRIPTION.
    damage_file_pa : TYPE
        DESCRIPTION.
    scn_number : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if dl_file_path == None:
        
        file_name = f"WaterNetworkDistribution_{scn_number}.json"
        run_dir = run_dir
        file_dir = os.path.join(run_dir, "Result", "WaterDistributionNetwork")
        file_path = os.path.join(file_dir, file_name)
    else:
        file_path = dl_file_path
        file_dir = Path(dl_file_path).parent
  
    return file_path, file_dir

def setSettingsData(input_json, REWET_input_data):
    policy_file_name = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["Policy Definition"]
    policy_file_path = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["Policy DefinitionPath"]
    
    policy_config_file = os.path.join(Path(policy_file_path), Path(policy_file_name) )
    
    REWET_input_data["settings"]["RUN_TIME"                  ] = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["simulationTime"]
    REWET_input_data["settings"]["simulation_time_step"      ] = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["simulationTimeStep"]
    
    REWET_input_data["settings"]['last_sequence_termination'    ] = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["last_sequence_termination"]
    REWET_input_data["settings"]['node_demand_temination'       ] = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["node_demand_temination"]
    REWET_input_data["settings"]['node_demand_termination_time' ] = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["node_demand_termination_time"]
    REWET_input_data["settings"]['node_demand_termination_ratio'] = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["node_demand_termination_ratio"]
    REWET_input_data["settings"]['solver'                       ] = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["Solver"]
    REWET_input_data["settings"]['Restoration_on'               ] = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["Restoration_on"]
    REWET_input_data["settings"]['minimum_job_time'             ] = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["minimum_job_time"]
    #REWET_input_data["settings"]['Restortion_config_file'       ] = policy_config_file # TODO: SINA unmark it
    
    p = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["pipe_damage_model"]
    REWET_input_data["settings"]['pipe_damage_model'] = {}
    for mat_data in p:
        REWET_input_data["settings"]['pipe_damage_model'][mat_data[0]] = \
            {"alpha":mat_data[1], "beta":mat_data[2], "gamma":mat_data[3],\
             "a":mat_data[4], "b":mat_data[5] }
    
    n = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["node_damage_model"]
    n = n[0]
    REWET_input_data["settings"]['node_damage_model'] = {'x':0.9012, 'a':n[0],\
       'aa':n[1], 'b':n[2], 'bb':n[3], 'c':n[4], 'cc':n[5], 'd':n[6],\
           'dd':n[7], 'e':n[8], 'ee1':n[9], 'ee2':n[10], 'f':n[11], 'ff1':n[12]\
               , 'ff2':n[13], "damage_node_model": "equal_diameter_emitter"} 
    
    if rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["Pipe_Leak_Based"]:
        pipe_leak_amount = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["pipe_leak_amount"]
        pipe_leak_time   = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["pipe_leak_time"]
        pipe_damage_discovery_mode = {'method': 'leak_based', 'leak_amount': pipe_leak_amount, 'leak_time': pipe_leak_time}
    else:
        pipe_time_discovery_ratio = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["pipe_time_discovery_ratio"]
        pipe_damage_discovery_mode = {'method': 'time_based', 'time_discovery_ratio': pipe_time_discovery_ratio}#pd.Series([line[0] for line in pipe_time_discovery_ratio], index = [line[1] for line in pipe_time_discovery_ratio])}
    
    if rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["Node_Leak_Based"]:
        node_leak_amount = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["node_leak_amount"]
        node_leak_time   = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["node_leak_time"]
        node_damage_discovery_mode = {'method': 'leak_based', 'leak_amount': node_leak_amount, 'leak_time': node_leak_time}
    else:
        node_time_discovery_ratio = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["node_time_discovery_ratio"]
        node_damage_discovery_mode = {'method': 'time_based', 'time_discovery_ratio': node_time_discovery_ratio } # pd.Series([line[0] for line in node_time_discovery_ratio], index = [line[1] for line in node_time_discovery_ratio])}    
    
    pump_time_discovery_ratio = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["pump_time_discovery_ratio"]
    tank_time_discovery_ratio = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["tank_time_discovery_ratio"]
    pump_damage_discovery_model = {'method': 'time_based', 'time_discovery_ratio': pump_time_discovery_ratio } # pd.Series([line[0] for line in pump_time_discovery_ratio], index = [line[1] for line in pump_time_discovery_ratio])}
    tank_damage_discovery_model = {'method': 'time_based', 'time_discovery_ratio': tank_time_discovery_ratio } # pd.Series([line[0] for line in tank_time_discovery_ratio], index = [line[1] for line in tank_time_discovery_ratio])}
    
    REWET_input_data["settings"]['pipe_damage_discovery_model'] = pipe_damage_discovery_mode
    REWET_input_data["settings"]['node_damage_discovery_model'] = node_damage_discovery_mode
    REWET_input_data["settings"]['pump_damage_discovery_model'] = pump_damage_discovery_model
    REWET_input_data["settings"]['tank_damage_discovery_model'] = tank_damage_discovery_model
    REWET_input_data["settings"]['minimum_pressure' ] = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["minimum_pressure"]
    REWET_input_data["settings"]['required_pressure'] = rwhale_input_Data["Recovery"]["WaterDistributionNetwork"]["required_pressure"]
    
    ############ Not Supposed to be in R2DTool GUI ############
    REWET_input_data["settings"]["minimum_simulation_time"] = 0 # TODO : HERE #REWET_input_data["event_time"] + REWET_input_data["settings"]["simulation_time_step"]
    REWET_input_data["settings"]["save_time_step"               ] = True
    REWET_input_data["settings"]['record_restoration_agent_logs'] = True
    REWET_input_data["settings"]['record_damage_table_logs'     ] = True
    REWET_input_data["settings"]["simulation_time_step"] = 3600
    REWET_input_data["settings"]["number_of_proccessor"] = 1
    REWET_input_data["settings"]['demand_ratio'              ] = 1
    REWET_input_data["settings"]['dmg_rst_data_save'         ] = True
    REWET_input_data["settings"]['Parameter_override'        ] = True 
    REWET_input_data["settings"]['mpi_resume'                ] = True #ignores the scenarios that are done
    REWET_input_data["settings"]['ignore_empty_damage'       ] = False
    REWET_input_data["settings"]['result_details'            ] = 'extended'
    REWET_input_data["settings"]['negative_node_elmination'  ] = True
    REWET_input_data["settings"]['nne_flow_limit'            ] = 0.5
    REWET_input_data["settings"]['nne_pressure_limit'        ] = -5
    REWET_input_data["settings"]['Virtual_node'              ] = True
    REWET_input_data["settings"]['damage_node_model'         ] = 'equal_diameter_emitter' #"equal_diameter_reservoir" 
    REWET_input_data["settings"]['default_pipe_damage_model'  ] = {"alpha":-0.0038, "beta":0.1096, "gamma":0.0196, "a":2, "b":1 }
    
    REWET_input_data["settings"]['limit_result_file_size'    ] = -1 #in Mb. 0 means no limit 
    REWET_input_data["settings"]['Pipe_damage_input_method'   ] = 'pickle'
        











if __name__ == '__main__':
    argParser = argparse.ArgumentParser(
        "Preprocess rwhale workflow to REWET input.")
    
    argParser.add_argument("--input", "-i",  default="inputRWHALE.json", 
                           help="rwhale input file json file")
    
    argParser.add_argument("--damage", "-d",
                           default="water_damage_input_structure.json", 
                           help="water damage input json file. If provided, number of realization is ignored if prvided and number of realization is set to 1.")
    
    argParser.add_argument("--number", "-n",
                           default=None,
                           help="numebr of realizations. If not provided, number of realization is acquired from rwhale file.")
    
    parser_data = argParser.parse_args()
    
    event_time = 2 * 3600 # this is the time of the event # TODO: Add it to teh front end
    
    REWET_input_data = {}
    REWET_input_data["settings"] = {}
    
    rwhale_input_Data = preprocessorIO.readJSONFile(parser_data.input)
    setSettingsData(rwhale_input_Data, REWET_input_data)
    
    
    # Set R2D enviroment and parameters
    water_asset_data = rwhale_input_Data["Applications"]\
        ["Assets"]["WaterDistributionNetwork"]
    inp_file_addr = water_asset_data["ApplicationData"]["inpFile"]
    if '{Current_Dir}' in inp_file_addr:
        inp_file_addr = inp_file_addr.replace("{Current_Dir}", ".")
    
    run_directory = rwhale_input_Data["runDir"]
    number_of_realization = rwhale_input_Data["Applications"]\
        ["DL"]["WaterDistributionNetwork"]["ApplicationData"]["Realizations"]
        
    if parser_data.number != None:
        number_of_realization = 1
    else:
        number_of_realization = rwhale_input_Data["Applications"]\
            ["DL"]["WaterDistributionNetwork"]["ApplicationData"]["Realizations"]

    
    for scn_number in range(number_of_realization):
        dl_file_path = getDLFileName(run_directory, parser_data.damage,
                                     scn_number)
        
        damage_data = damage_convertor.readDamagefile(
            dl_file_path,  run_directory, event_time )
    
        scenario_list, prefix = createScnearioList(run_directory, scn_number)
        
        damage_save_path = Path(run_directory) / "Results" / "WaterDistributionNetwork" / "damage_input"
        scneario_list_path = preprocessorIO.saveScenarioData(damage_data, damage_save_path, scenario_list)
    
        #setSettingsData(rwhale_input_Data, REWET_input_data)
        REWET_input_data["settings"]["result_directory"] = os.path.join(\
            run_directory,"Results","WaterDistributionNetwork", "REWET_Result")
    
        REWET_input_data["settings"]["temp_directory"] = os.path.join(\
            run_directory,"Results", "WaterDistributionNetwork", "REWET_RunFiles")
    
        REWET_input_data["settings"]["temp_directory"] = os.path.join(\
            run_directory,"Results", "WaterDistributionNetwork", "REWET_RunFiles")
    
        REWET_input_data["settings"]['WN_INP'] = inp_file_addr
        
        REWET_input_data["settings"]["pipe_damage_file_list"] = scneario_list_path
        REWET_input_data["settings"]["pipe_damage_file_directory"] = str( damage_save_path ) 
        
        settings_json_file_path = preprocessorIO.saveSettingsFile(REWET_input_data, run_directory, prefix)
        REWET_starter = Starter()
        REWET_starter.run(settings_json_file_path)
        
    
    
    