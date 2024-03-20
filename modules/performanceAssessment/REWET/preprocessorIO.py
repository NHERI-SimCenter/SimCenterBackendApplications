# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:53:49 2024

@author: naeim
"""

import json
import os
from pathlib import Path
import pandas as pd

    
def readJSONFile(file_addr):
    """
    Reads a json file.

    Parameters
    ----------
    file_addr : Path
        JSON file address.

    Raises
    ------
    ValueError
        If the file is not found on the address.

    Returns
    -------
    data : dict
        JSON File data as a dict.

    """

    if not os.path.exists(file_addr):
        raise ValueError("INPUT WHALE FILE is not found.", repr(file_addr) )
    
    with open(file_addr, "rt") as f:
        data = json.load(f)
    
    return data

# =============================================================================
# def readRWHALEFileForREWET(file_addr, REWET_input_data):
#     """
#     Reads rwhile input file and returns the data as a dict and updates REWET
#     input file.
# 
#     Parameters
#     ----------
#     file_addr : Path
#         rwhale input file path.
#     REWET_input_data : dict
#         REWET input data.
# 
#     Returns
#     -------
#     rwhale_data : dict
#         rwhale inoput data as a dict.
# 
#     """
#     
#     
#     water_asset_data = rwhale_data["Applications"]\
#         ["Assets"]["WaterDistributionNetwork"]
#     inp_file_addr = water_asset_data["ApplicationData"]["inpFile"]
#     run_directory = rwhale_data["runDir"]
#     number_of_realization = rwhale_data["Applications"]\
#         ["DL"]["WaterDistributionNetwork"]["ApplicationData"]["Realizations"]
#     
#     REWET_input_data["inp_file" ] = inp_file_addr
#     REWET_input_data["run_dir"] = run_directory
#     REWET_input_data["number_of_realizations"] = number_of_realization
#     
#     return rwhale_data
# =============================================================================

def saveScenarioData(damage_data, damage_save_path, scenario_list):
    """
    Saves the scneario data including scneario table and damaghe data acording
    to the table data/

    Parameters
    ----------
    REWET_input_data : Dict
        REWET input data.

    Returns
    -------
    None.

    """
    
    scenario_list = pd.DataFrame([scenario_list]).set_index("Scenario Name")
    pipe_damage_data = damage_data["Pipe"]
    node_damage_data = damage_data["Node"]
    pump_damage_data = damage_data["Pump"]
    tank_damage_data = damage_data["Tank"]
    
    scenario_list_file_name = f"{scenario_list.index[0]}.xlsx"
    pipe_damage_file_name   = scenario_list.iloc[0]["Pipe Damage"]
    node_damage_file_name   = scenario_list.iloc[0]["Nodal Damage"]
    pump_damage_file_name   = scenario_list.iloc[0]["Pump Damage"]
    tank_damage_file_name   = scenario_list.iloc[0]["Tank Damage"]
    
    scenario_list_file_path = os.path.join(damage_save_path, scenario_list_file_name)
    pipe_damage_file_path   = os.path.join(damage_save_path, pipe_damage_file_name)
    node_damage_file_path   = os.path.join(damage_save_path, node_damage_file_name)
    pump_damage_file_path   = os.path.join(damage_save_path, pump_damage_file_name)
    tank_damage_file_path   = os.path.join(damage_save_path, tank_damage_file_name)
    
    scenario_list.to_excel(scenario_list_file_path)
    pipe_damage_data.to_pickle(pipe_damage_file_path)
    node_damage_data.to_pickle(node_damage_file_path)
    pump_damage_data.to_pickle(pump_damage_file_path)
    tank_damage_data.to_pickle(tank_damage_file_path)
    
    return scenario_list_file_path

def saveSettingsFile(REWET_input_data, run_dir, prefix):
    """
    Saves seetings data that REWET NEEDs.

    Parameters
    ----------
    REWET_input_data : Dict
        REWET input data.

    Returns
    -------
    None.

    """
    
    settings = REWET_input_data["settings"]
    
    settings_file_name = prefix + "_" + "settings.json"
    damage_save_path = Path(run_dir) / "Results" / "WaterDistributionNetwork" / "damage_input" / settings_file_name
    with open(damage_save_path, "w") as f:
        json.dump(settings , f, indent=4)
    
    return damage_save_path
    
    
    
    
    