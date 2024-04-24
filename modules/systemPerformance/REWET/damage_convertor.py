# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 09:40:15 2024

@author: naeim
"""

import os
from pathlib import Path
import pandas as pd
import preprocessorIO

CBIG_int = int(1e9)


   

def createPipeDamageInputForREWET(pipe_damage_data, run_dir, event_time, sc_geojson):
    """
    Creates REWET-style piep damage file.

    Parameters
    ----------
    pipe_damage_data : dict
        Pipe damage data from PELICUN.
    REWET_input_data : dict
        REWET input data.

    Raises
    ------
    ValueError
        If damage type is not what it should be.

    Returns
    -------
    pipe_damage_list : Pandas Series
        REWET-style pipe damage file.

    """
    pipe_id_list = [key for key in pipe_damage_data]
    
    damage_list = []
    damage_time = event_time
    sc_geojson_file = preprocessorIO.readJSONFile(sc_geojson)
    pipe_data = [ss for ss in sc_geojson_file["features"] if ss["properties"]["type"]=="Pipe"]
    pipe_index = [str(ss["id"]) for ss in pipe_data]
    pipe_id = [ss["properties"]["InpID"] for ss in pipe_data]
    pipe_index_to_id = dict(zip(pipe_index, pipe_id))
    
    for pipe_id in pipe_id_list:
        cur_data = pipe_damage_data[pipe_id]

        cur_damage = cur_data["Damage"]
        cur_demand = cur_data["Demand"]
        
        aim_data = findAndReadAIMFile(pipe_id,os.path.join(
            "Results", "WaterDistributionNetwork", "Pipe"), run_dir)
        
        material = aim_data["GeneralInformation"].get("Material", None)
        
        if material == None:
            #raise ValueError("Material is none")
            material = "CI"
        
        aggregates_list = [cur_agg for cur_agg in list( cur_damage.keys() ) if "aggregate" in cur_agg]
        segment_sizes = len(aggregates_list )
        segment_step = 1 / segment_sizes
        c = 0
        
        for cur_agg in aggregates_list: #cur_damage["aggregate"]:
            damage_val = cur_damage[cur_agg]
            if damage_val > 0:
                if damage_val == 1:
                    damage_type = "leak"
                elif damage_val == 2:
                    damage_type = "break"
                else:
                    raise ValueError("The damage type must be eother 1 or 2")
            else:
                continue
            

            cur_loc = c * segment_step + segment_step / 2
            #print(cur_loc)
            c += 1
            damage_list.append( {"pipe_id": pipe_index_to_id[pipe_id], "damage_loc": cur_loc,
                          "type": damage_type, "Material": material}
                               )
    damage_list.reverse()
    pipe_damage_list = pd.Series(data=damage_list,
                                 index=[damage_time for val in damage_list], dtype="O")
    
    #REWET_input_data["Pipe_damage_list"] =  pipe_damage_list 
    #REWET_input_data["AIM"] =  aim_data       
    
    
    return pipe_damage_list

def createNodeDamageInputForREWET(node_damage_data, run_dir, event_time):
    """
    Creates REWET-style node damage file.

    Parameters
    ----------
    node_damage_data : dict
        Node damage data from PELICUN.
    REWET_input_data : dict
        REWET input data.

    Returns
    -------
    node_damage_list : Pandas Series
        REWET-style node damage file.

    """
    node_id_list = [key for key in node_damage_data]
    
    damage_list = []
    damage_time = event_time
    
    for node_id in node_id_list:
        cur_damage = node_damage_data[node_id]
        aggregates_list = [cur_agg for cur_agg in list( cur_damage.keys() ) if "aggregate" in cur_agg]
        
        if len(aggregates_list) == 0:
            continue
        
        cur_data = node_damage_data[node_id]

        cur_damage = cur_data["Damage"]
        cur_demand = cur_data["Demand"]
        
        aim_data = findAndReadAIMFile(node_id,os.path.join(
            "Results", "WaterDistributionNetwork", "Node"),
                                           run_dir)
        
        total_length = aim_data["GeneralInformation"].get("Total_length", None)
        total_number_of_damages = cur_damage["aggregate"]
        
        damage_list.append( {"node_name": node_id,
                             "number_of_damages": total_number_of_damages,
                             "node_Pipe_Length": total_length}
                               )
    
    node_damage_list = pd.Series(data=damage_list,
                                 index=[damage_time for val in damage_list], dtype="O")
    
    return node_damage_list
    
def createPumpDamageInputForREWET(pump_damage_data, REWET_input_data):
    """
     Creates REWET-style pump damage file.

    Parameters
    ----------
    pump_damage_data : dict
        Pump damage data from PELICUN.
    REWET_input_data : dict
        REWET input data.

    Returns
    -------
    pump_damage_list : Pandas Series
        REWET-style pump damage file.

    """
    pump_id_list = [key for key in pump_damage_data]
    
    damage_list = []
    damage_time = REWET_input_data["event_time"]
    
    for pump_id in pump_id_list:
        cur_data = pump_damage_data[pump_id]

        cur_damage = cur_data["Damage"]
        cur_repair_time = cur_data["Repair"]
        
        if cur_damage == 0:
            continue # cur_damage_state = 0 means undamaged pump
        
        # I'm not sure if we need any data about the pump at this point
        
        #aim_data = findAndReadAIMFile(tank_id, os.path.join(
            #"Results", "WaterDistributionNetwork", "Pump"),
                                           #REWET_input_data["run_dir"])
        
        #We are getting this data from PELICUN
        #restore_time = getPumpRetsoreTime(cur_damage)
        damage_list.append( {"pump_id": pump_id,
                          "time": damage_time, "Restore_time": cur_repair_time}
                               )
    pump_damage_list = pd.Series(index=[damage_time for val in damage_list], data=damage_list)
    
    return pump_damage_list
    
    
    
def createTankDamageInputForREWET(tank_damage_data, REWET_input_data):
    """
    Creates REWET-style Tank damage file.

    Parameters
    ----------
    tank_damage_data : dict
        Tank damage data from PELICUN.
    REWET_input_data : dict
        REWET input data.

    Returns
    -------
    tank_damage_list : Pandas Series
        REWET-style tank damage file.
    """
    tank_id_list = [key for key in tank_damage_data]
    
    damage_list = []
    damage_time = REWET_input_data["event_time"]
    
    for tank_id in tank_id_list:
        cur_data = tank_damage_data[tank_id]

        cur_damage = cur_data["Damage"]
        cur_repair_time = cur_data["Repair"]
        
        if cur_damage == 0:
            continue # cur_damage_state = 0 meeans undamged tank
        
# =============================================================================
#         # We are getting his data from REWET
#         
#         aim_data = findAndReadAIMFile(tank_id, os.path.join(
#             "Results", "WaterDistributionNetwork", "Tank"),
#                                            REWET_input_data["run_dir"])
#         tank_type = aim_data["GeneralInformation"].get("Type", None)
#         restore_time = getTankRetsoreTime(tank_type, cur_damage)
# =============================================================================
        
        damage_list.append( {"tank_id": tank_id,
                          "time": damage_time, "Restore_time": cur_repair_time}
                               )
    
    tank_damage_list = pd.Series(index=[damage_time for val in damage_list], data=damage_list)
    
    return tank_damage_list
    
    
def findAndReadAIMFile(asset_id, asset_type, run_dir):
    """
    Finds and read the AIM file for an asset.

    Parameters
    ----------
    asset_id : int
        The asset ID.
    asset_type : str
        Asset Type (e.g., Building, WaterDistributionNetwork).
    run_dir : path
        The directory where data is stored (aka the R2dTool directory)

    Returns
    -------
    aim_file_data : dict
        AIM file data as a dict.

    """

    file_path = Path(run_dir, asset_type, str(asset_id), "templatedir", f"{asset_id}-AIM.json")
    aim_file_data = preprocessorIO.readJSONFile(str(file_path) )
    return aim_file_data
       
def getPumpRetsoreTime(damage_state):
    """
    NOT USED! WE WILL GET IT FROM PELICUN
    
    Provides the restore time based on HAZUS repair time or any other
    approach available in the future. If damage state is slight, the restore
    time is 3 days (in seconds). If damage state is 2, the restore time is 7
    days (in seconds). If damage state is 3 or 4, the restore time is
    indefinite (a big number).
    
    Parameters
    ----------
    damage_state : Int
        Specifies the damage state (1 for slightly damages, 2 for moderate,
        3 etensive, and 4 complete.

    Returns
    -------
    Retstor time : int
        

    """
    
    if damage_state == 1:
        restore_time = int(3 * 24 * 3600)
    elif damage_state == 2:
        restore_time = int(7 * 24 * 3600)
    else:
        restore_time = CBIG_int
        
    return restore_time

def getTankRetsoreTime(tank_type, damage_state):
    """
    NOT USED! WE WILL GET IT FROM PELICUN
    
    Provides the restore time based on HAZUS repair time or any other
    approach available in the future. if damage state is slight, the restore
    time is 3 days (in seconds). If damage state is 2, the restore time is 7
    days (in seconds). If damage state is 3 or 4, the restore time is
    indefinite (a big number).
    
    Parameters
    ----------
    tank_type : STR
        Tank type based on the data schema. The parametr is not used for now.
    damage_state : Int
        Specifies the damage state (1 for slightly damages, 2 for moderate,
        3 etensive, and 4 complete.

    Returns
    -------
    Retstor time : int
        

    """
    
    if damage_state == 1:
        restore_time = int(3 * 24 * 3600)
    elif damage_state == 2:
        restore_time = int(7 * 24 * 3600)
    else:
        restore_time = CBIG_int
        
    return restore_time

def readDamagefile(file_addr, run_dir, event_time, sc_geojson):
    """
    Reads PELICUN damage files and create REWET-Style damage for all
    WaterDistributionNetwork elements

    Parameters
    ----------
    file_addr : path
        PELICUN damage file in JSON format.
    REWET_input_data : dict
        REWET input data, whcih is updated in the function.
    scn_number : dict
        JSON FILE.

    Returns
    -------
    damage_data : dict
        Damage data in PELICUN dict format.

    """
    # TODO: Make reading once for each scneario
        
    #wn = wntrfr.network.WaterNetworkModel(REWET_input_data["inp_file"] )
    
    damage_data = preprocessorIO.readJSONFile(file_addr)
    
    wn_damage_data = damage_data["WaterDistributionNetwork"]

    if "Pipe" in wn_damage_data:
        pipe_damage_data = createPipeDamageInputForREWET(
            wn_damage_data["Pipe"], run_dir, event_time, sc_geojson)
    else:
        pipe_damage_data = pd.Series(dtype="O")
    
    if "Tank" in wn_damage_data:
        tank_damage_data = createTankDamageInputForREWET(
            wn_damage_data["Tank"],  run_dir, event_time)
    else:
        tank_damage_data = pd.Series(dtype="O")
    
    if "Pump" in wn_damage_data: 
        pump_damage_data = createPumpDamageInputForREWET(
            wn_damage_data["Pump"],  run_dir, event_time)
    else:
        pump_damage_data = pd.Series(dtype="O")
    
    if "Junction" in wn_damage_data: 
        node_damage_data = createNodeDamageInputForREWET(
            wn_damage_data["Junction"],  run_dir, event_time)
    else:
        node_damage_data = pd.Series(dtype="O")
    
    damage_data = {}
    damage_data["Pipe"] = pipe_damage_data
    damage_data["Tank"] = tank_damage_data
    damage_data["Pump"] = pump_damage_data
    damage_data["Node"] = node_damage_data
    
    return damage_data