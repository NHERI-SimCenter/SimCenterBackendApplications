import os
import pandas as pd
import pickle


##################### Read files From Pickle #####################
def read_pipe_damage_seperate_pickle_file(directory, all_damages_file_name):
    file_dest = os.path.join(directory, all_damages_file_name)
    with open(file_dest, 'rb') as f:
        _all_damages = pickle.load(f)

    return _all_damages


def read_node_damage_seperate_pickle_file(directory, all_damages_file_name):
    file_dest = os.path.join(directory, all_damages_file_name)
    with open(file_dest, 'rb') as f:
        _node_damages = pickle.load(f)

    return _node_damages


def read_tank_damage_seperate_pickle_file(directory, tank_damages_file_name):
    file_dest = os.path.join(directory, tank_damages_file_name)
    with open(file_dest, 'rb') as f:
        _tank_damages = pickle.load(f)

    return _tank_damages


def read_pump_damage_seperate_pickle_file(directory, pump_damages_file_name):
    file_dest = os.path.join(directory, pump_damages_file_name)
    with open(file_dest, 'rb') as f:
        _pump_damages = pickle.load(f)

    return _pump_damages


##################### Read files From Excel #####################


def read_pipe_damage_seperate_EXCEL_file(directory, pipe_damages_file_name):
    ss = None
    file_dest = os.path.join(directory, pipe_damages_file_name)
    ss = pd.read_excel(file_dest)
    ss.sort_values(
        ['pipe_id', 'damage_time', 'damage_loc'],
        ascending=[True, True, False],
        ignore_index=True,
        inplace=True,
    )
    unique_time = ss.groupby(['pipe_id']).time.unique()
    if 1 in [
        0 if len(i) <= 1 else 1 for i in unique_time
    ]:  # checks if there are any pipe id with more than two unqiue time values
        raise ValueError(
            'All damage location for one pipe should happen at the same time'
        )
    ss.set_index('time', inplace=True)
    ss.pipe_id = ss.pipe_id.astype(str)
    return pd.Series(ss.to_dict('records'), index=ss.index)


def read_node_damage_seperate_EXCEL_file(directory, node_damages_file_name):
    ss = None
    file_dest = os.path.join(directory, node_damages_file_name)
    ss = pd.read_excel(file_dest)
    ss.set_index('time', inplace=True)
    ss.node_name = ss.node_name.astype(str)
    return pd.Series(ss.to_dict('records'), index=ss.index)


def read_tank_damage_seperate_EXCEL_file(directory, tank_damages_file_name):
    ss = None
    file_dest = os.path.join(directory, tank_damages_file_name)
    ss = pd.read_excel(file_dest)
    #    ss.set_index('Tank_ID', inplace=True)
    ss.set_index('time', inplace=True)
    ss.Tank_ID = ss.Tank_ID.astype(str)
    # ss = ss['Tank_ID']

    return ss


def read_pump_damage_seperate_EXCEL_file(directory, pump_damages_file_name):
    ss = None
    file_dest = os.path.join(directory, pump_damages_file_name)
    ss = pd.read_excel(file_dest)
    ss.set_index('time', inplace=True)
    ss.Pump_ID = ss.Pump_ID.astype(str)
    return ss


##################### Save Results #####################


def save_single(settings, result, name, restoration_data):
    result_file_directory = settings.process['result_directory']
    result_name = name + '.res'
    settings_name = name + '.xlsx'

    file_dest = os.path.join(result_file_directory, result_name)
    print('Saving: ' + str(file_dest))
    with open(file_dest, 'wb') as f:
        pickle.dump(result, f)

    process_set = pd.Series(settings.process.settings)
    scenario_set = pd.Series(settings.scenario.settings)
    _set = pd.Series(
        process_set.to_list() + scenario_set.to_list(),
        index=process_set.index.to_list() + scenario_set.index.to_list(),
    )
    file_dest = os.path.join(result_file_directory, settings_name)
    _set.to_excel(file_dest)

    if settings.process['dmg_rst_data_save']:
        # file_dest = os.path.join(result_file_directory, 'restoration_file.pkl')
        # rest_data_out = pd.DataFrame.from_dict(restoration_data)
        # rest_data_out.to_pickle(file_dest)
        file_dest = os.path.join(result_file_directory, name + '_registry.pkl')
        print('Saving: ' + str(file_dest))
        with open(file_dest, 'wb') as f:
            pickle.dump(restoration_data, f)
