import os
import pandas as pd
import pickle
import json


##################### Read files From json #####################
def read_pipe_damage_seperate_json_file(directory, pipe_file_name):
    """Read pipe damage of a single scenario.

    Args:
        directory (path): damage scnearios path
        pipe_file_name (str): pipe damage file name

    Raises:
        ValueError: _description_
        RuntimeError: _description_

    Returns:
        Pandas.Series: Pipe Damage 
    """
    pipe_damaage = []
    pipe_time = []

    file_dest = os.path.join(directory, pipe_file_name)
    
    with open(file_dest, "rt") as f:
        read_file = json.load(f)

    if not isinstance(read_file, list):
        raise ValueError("Wrong inpout in PIPE damage file")
    
    for each_damage in read_file:
        pipe_time.append(each_damage.get("time") )
        
        cur_damage = {"pipe_ID": each_damage.get("Pipe_ID"),
                      "damage_loc": each_damage.get("Loc"),
                      "type": each_damage.get("Type"),
                      "Material": each_damage.get("Material")
        }

        pipe_damaage.append(cur_damage)

    return pd.Series(index = pipe_time, data = pipe_damaage)
    
def read_node_damage_seperate_json_file(directory, node_file_name):
    """Read node damage of a single scenario.

    Args:
        directory (path): damage scnearios path
        pipe_file_name (str): node damage file name

    Raises:
        ValueError: _description_
        RuntimeError: _description_

    Returns:
        Pandas.Series: node Damage 
    """
    node_damage = []
    node_time = []

    file_dest = os.path.join(directory, node_file_name)
    
    with open(file_dest, "rt") as f:
        read_file = json.load(f)

    if not isinstance(read_file, list):
        raise ValueError("Wrong inpout in NODE damage file")
    
    for each_damage in read_file:
        node_time.append(each_damage.get("time") )
        
        cur_damage = {"node_name": each_damage.get("Node_ID"),
                      "Number_of_damages": each_damage.get("Number_of_Damages"),
                      "node_Pipe_Length": each_damage.get("Pipe Length")
        }

        node_damage.append(cur_damage)

    return pd.Series(index = node_time, data = node_damage)

def read_tank_damage_seperate_json_file(directory, tank_file_name):
    """Read tank damage of a single scenario.

    Args:
        directory (path): tank scnearios path
        pipe_file_name (str): tank damage file name

    Raises:
        ValueError: _description_
        RuntimeError: _description_

    Returns:
        Pandas.Series: tank Damage 
    """
    tank_damage = []
    tank_time = []

    file_dest = os.path.join(directory, tank_file_name)
    
    with open(file_dest, "rt") as f:
        read_file = json.load(f)

    if not isinstance(read_file, list):
        raise ValueError("Wrong inpout in TANK damage file")
    
    for each_damage in read_file:
        tank_time.append(each_damage.get("time") )
        
        cur_damage = {"Tank_ID": each_damage.get("Tank_ID"),
                      "Restore_time": each_damage.get("Restore_time"),
        }

        tank_time.append(cur_damage)

    return pd.Series(index = tank_time, data = tank_damage)

def read_pump_damage_seperate_json_file(directory, pump_file_name):
    """Read pump damage of a single scenario.

    Args:
        directory (path): pump scnearios path
        pipe_file_name (str): pump damage file name

    Raises:
        ValueError: _description_
        RuntimeError: _description_

    Returns:
        Pandas.Series: pump Damage 
    """
    pump_damage = []
    pump_time = []

    file_dest = os.path.join(directory, pump_file_name)
    
    with open(file_dest, "rt") as f:
        read_file = json.load(f)

    if not isinstance(read_file, list):
        raise ValueError("Wrong inpout in PUMP damage file")
    
    for each_damage in read_file:
        pump_time.append(each_damage.get("time") )
        
        cur_damage = {"PUMP_ID": each_damage.get("Pump_ID"),
                      "Restore_time": each_damage.get("Restore_time"),
        }

        pump_time.append(cur_damage)

    return pd.Series(index = pump_time, data = pump_damage)


##################### Read files From Pickle #####################
def read_pipe_damage_seperate_pickle_file(directory, all_damages_file_name):
    file_dest=os.path.join(directory, all_damages_file_name)
    with open(file_dest, 'rb') as f:
        _all_damages = pickle.load(f)

    return _all_damages

def read_node_damage_seperate_pickle_file(directory, all_damages_file_name):
    file_dest=os.path.join(directory, all_damages_file_name)
    with open(file_dest, 'rb') as f:
        _node_damages = pickle.load(f)
    
    return  _node_damages

def read_tank_damage_seperate_pickle_file(directory, tank_damages_file_name):
    file_dest=os.path.join(directory, tank_damages_file_name)
    with open(file_dest, 'rb') as f:
        _tank_damages = pickle.load(f)

    return _tank_damages

def read_pump_damage_seperate_pickle_file(directory, pump_damages_file_name):
    file_dest=os.path.join(directory, pump_damages_file_name)
    with open(file_dest, 'rb') as f:
        _pump_damages = pickle.load(f)

    return _pump_damages

##################### Read files From Excel #####################

def read_pipe_damage_seperate_EXCEL_file(directory, pipe_damages_file_name):
    ss=None
    file_dest=os.path.join(directory, pipe_damages_file_name)
    ss=pd.read_excel(file_dest)
    ss.sort_values(['pipe_id','time','damage_loc'],ascending=[True,True,False], ignore_index=True, inplace=True)
    unique_time = ss.groupby(['pipe_id']).time.unique()
    if 1 in [0 if len(i)<=1 else 1 for i in unique_time]: # checks if there are any pipe id with more than two unqiue time values 
        raise ValueError("All damage location for one pipe should happen at the same time")
    ss.set_index('time', inplace=True)
    ss.pipe_id = ss.pipe_id.astype(str)
    return pd.Series(ss.to_dict('records'), index=ss.index)

def read_node_damage_seperate_EXCEL_file(directory, node_damages_file_name):
    ss           = None
    file_dest    = os.path.join(directory, node_damages_file_name)
    ss           = pd.read_excel(file_dest)
    ss.set_index('time', inplace=True)
    ss.node_name = ss.node_name.astype(str)
    return pd.Series(ss.to_dict('records'), index=ss.index)

def read_tank_damage_seperate_EXCEL_file(directory, tank_damages_file_name):
    ss         = None
    file_dest  = os.path.join(directory, tank_damages_file_name)
    ss         = pd.read_excel(file_dest)
#    ss.set_index('Tank_ID', inplace=True)
    ss.set_index('time', inplace=True)
    ss.Tank_ID = ss.Tank_ID.astype(str)
    #ss = ss['Tank_ID']

    return ss

def read_pump_damage_seperate_EXCEL_file(directory, pump_damages_file_name):
    ss           = None
    file_dest    = os.path.join(directory, pump_damages_file_name)
    ss           = pd.read_excel(file_dest)
    ss.set_index('time', inplace=True)
    ss.Pump_ID = ss.Pump_ID.astype(str)
    return ss

def read_damage_list(list_file_addr, file_directory, iCheck=False):
    """
    Reads damage sceanrio list.

    Parameters
    ----------
    list_file_addr : Address to the target list file
        DESCRIPTION.
    file_directory : TYPE
        DESCRIPTION.
    iCheck : TYPE, optional
        DESCRIPTION. The default is False. Checks if all damage files are found.

    Raises
    ------
    RuntimeError
        if file not found.

    Returns
    -------
    damage_list : Pandas Dataframe
        DESCRIPTION.

    """
    damage_list=None
    error_file_name=[]

    with open(list_file_addr, 'rb') as f:
        damage_list = pd.read_excel(f)

    iError=False
    temp = damage_list['Pipe Damage'].tolist()
    
    if iCheck==False:
        return damage_list
    
    for file_name in temp:
        if not os.path.exists(file_name):
            iError=True
            error_file_name.append(file_name)
    
    if iError:
        raise RuntimeError('The Follwoing files could not be found: '+repr(error_file_name))
    return damage_list

##################### Save Results #####################

def save_single(settings, result, name, restoration_data):
    result_file_directory = settings.process['result_directory']
    print(result_file_directory)
    result_name   = name + '.res'
    settings_name = name + '.xlsx'    
    
    file_dest = os.path.join(result_file_directory, result_name)
    print("Saving: "+str(file_dest))
    with open(file_dest, 'wb') as f:
        pickle.dump(result, f)
    
    
    process_set  = pd.Series(settings.process.settings)
    scenario_set = pd.Series(settings.scenario.settings)
    _set = pd.Series(process_set.to_list()+scenario_set.to_list(), index=process_set.index.to_list()+scenario_set.index.to_list())
    file_dest = os.path.join(result_file_directory, settings_name)
    _set.to_excel(file_dest)
    
    if settings.process['dmg_rst_data_save']:
        #file_dest = os.path.join(result_file_directory, 'restoration_file.pkl')
        #rest_data_out = pd.DataFrame.from_dict(restoration_data)
        #rest_data_out.to_pickle(file_dest)
        file_dest = os.path.join(result_file_directory, name+'_registry.pkl')
        print("Saving: "+str(file_dest))
        with open(file_dest, 'wb') as f:
            pickle.dump(restoration_data, f)