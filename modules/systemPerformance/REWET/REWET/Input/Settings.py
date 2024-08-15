import json
import pickle
import warnings

import numpy as np
import pandas as pd

list_default_headers = [
    'Scenario Name',
    'Pipe Damage',
    'Nodal Damage',
    'Pump Damage',
    'Tank Damage',
    'Probability',
]

acceptable_override_list = ['POINTS']


class base:
    def __init__(self):
        self.settings = {}

    def __getitem__(self, key):
        return self.settings[key]

    def __setitem__(self, key, data):
        self.settings[key] = data


class Process_Settings(base):
    def __init__(self):
        super().__init__()
        """
        simulation settings
        """
        self.settings['RUN_TIME'] = (5 + 24 * 1) * 3600  # seconds
        self.settings['minimum_simulation_time'] = (10 + 24 * 2) * 3600  # seconds
        self.settings['simulation_time_step'] = 3600  # seconds
        self.settings['number_of_damages'] = (
            'multiple'  # single or multiple. If single, indicate single damage files. If multiple, indicate "pipe_damage_file_list"
        )
        self.settings['result_directory'] = 'Result'  # "Net3//Result"
        self.settings['temp_directory'] = 'RunFiles'
        self.settings['save_time_step'] = True
        self.settings['last_sequence_termination'] = (
            True  # sina needs to be applied in GUI
        )
        self.settings['node_demand_temination'] = (
            False  # sina needs to be applied in GUI
        )
        self.settings['node_demand_termination_time'] = (
            3 * 3600
        )  # sina needs to be applied in GUI
        self.settings['node_demand_termination_ratio'] = (
            0.95  # sina needs to be applied in GUI
        )
        self.settings['record_restoration_agent_logs'] = (
            True  # sina needs to be applied in GUI
        )
        self.settings['record_damage_table_logs'] = (
            True  # sina needs to be applied in GUI
        )

        """
        Hydraulic settings
        """
        self.settings['WN_INP'] = (
            'Example/net3.inp'  # 'giraffe386-4-1.inp' #"Anytown.inp"#'giraffe386-4-1.inp' #"Net3/net3.inp"
        )
        self.settings['demand_ratio'] = 1
        self.settings['solver'] = (
            'ModifiedEPANETV2.2'  # sina needs to be implemented
        )
        # self.settings['hydraulic_time_step'] = 3600
        self.settings['solver_type'] = 'ModifiedEPANETV2.2'

        """
        Damage settings
        """

        self.settings['pipe_damage_file_list'] = (
            'Example/example_list.xlsx'  # "Nafiseh Damage Data/9_final_akhar/list_1_final.xlsx" #"preprocess/list2-3.xlsx"#"preprocess/list2-3.xlsx" #"list_akhar_with_prob_pgv_epicenter_1.xlsx"#"preprocess/list2-3.xlsx" #"Net3/list.xlsx" #"preprocess/list2-3.xlsx" #"list_W147_6.xlsx" #'Nafiseh Damage Data/list.xlsx'
        )
        self.settings['pipe_damage_file_directory'] = (
            r'Example\Damages'  # 'Nafiseh Damage Data/9_final_akhar'#"" #'Net3' #'Nafiseh Damage Data/out'"X:\\Sina Naeimi\\anytown_damage\\"
        )
        self.settings['pump_damage_relative_time'] = (
            True  # needs to be implemented in the code
        )
        self.settings['tank_damage_relative_time'] = (
            True  # needs to be implemented in the code
        )

        """
        Restoration settings
        """
        self.settings['Restoration_on'] = True
        self.settings['minimum_job_time'] = 3600  # sina needs to be implemented

        """
        None GUI settings
        """
        # self.settings['job_assign_time_limit'     ]=None # time in seconds or None
        self.settings['maximun_worker_idle_time'] = 60
        self.settings['number_of_proccessor'] = 1

        self.settings['dmg_rst_data_save'] = True
        self.settings['Parameter_override'] = (
            True  # 'starter/settings.xlsx' #this is for settings sensitivity analysis
        )
        self.settings['mpi_resume'] = True  # ignores the scenarios that are done
        self.settings['ignore_empty_damage'] = False
        self.settings['result_details'] = 'extended'
        self.settings['negative_node_elmination'] = True
        self.settings['nne_flow_limit'] = 0.5
        self.settings['nne_pressure_limit'] = -5
        self.settings['Virtual_node'] = True
        self.settings['damage_node_model'] = (
            'equal_diameter_emitter'  # "equal_diameter_reservoir"
        )

        self.settings['limit_result_file_size'] = -1  # in Mb. 0 means no limit


class Scenario_Settings(base):
    def __init__(self):
        super().__init__()
        """
        Hydraulic settings
        """
        self.settings['minimum_pressure'] = 8
        self.settings['required_pressure'] = 25
        self.settings['pressure_exponent'] = 0.75  # sina add it to the code and GUI
        # Sina also take care of the nodal damage formula in terms of exponents [Urgent]
        self.settings['hydraulic_time_step'] = 900

        """
        Damage settings
        """
        self.settings['Pipe_damage_input_method'] = 'excel'  # excel or pickle
        self.settings['pipe_damage_model'] = {
            'CI': {
                'alpha': -0.0038,
                'beta': 0.1096,
                'gamma': 0.0196,
                'a': 2,
                'b': 1,
            },
            'DI': {
                'alpha': -0.0079,
                'beta': 0.0805,
                'gamma': 0.0411,
                'a': 2,
                'b': 1,
            },
            'STL': {
                'alpha': -0.009,
                'beta': 0.0808,
                'gamma': 0.0472,
                'a': 2,
                'b': 1,
            },
            'CON': {
                'alpha': -0.0083,
                'beta': 0.0738,
                'gamma': 0.0431,
                'a': 2,
                'b': 1,
            },
            'RS': {
                'alpha': -0.0088,
                'beta': 0.0886,
                'gamma': 0.0459,
                'a': 2,
                'b': 1,
            },
        }  # sina needs to be implemented
        self.settings['default_pipe_damage_model'] = {
            'alpha': -0.0038,
            'beta': 0.1096,
            'gamma': 0.0196,
            'a': 2,
            'b': 1,
        }
        self.settings['node_damage_model'] = {
            'x': 0.9012,
            'a': 0.0036,
            'aa': 1,
            'b': 0,
            'bb': 0,
            'c': -0.877,
            'cc': 1,
            'd': 0,
            'dd': 0,
            'e': 0.0248,
            'ee1': 1,
            'ee2': 1,
            'f': 0,
            'ff1': 0,
            'ff2': 0,
            'damage_node_model': 'equal_diameter_emitter',
        }  # sina needs to be implemented
        # Sina, there is no x in the GUI. Implement it
        """
        Restoration settings 
        """
        self.settings['Restoraion_policy_type'] = (
            'script'  # sina needs to be implemented in the code
        )
        self.settings['Restortion_config_file'] = (
            'Example/exampe_config.txt'  # "config-ghab-az-tayid.txt" #'X:\\Sina Naeimi\\anytown_damage\\config-base_base.txt'#'config-base_hydsig.txt' #'Net3/config.txt' #
        )
        self.settings['pipe_damage_discovery_model'] = {
            'method': 'leak_based',
            'leak_amount': 0.025,
            'leak_time': 3600 * 12,
        }  # sina needs to be implemented
        self.settings['node_damage_discovery_model'] = {
            'method': 'leak_based',
            'leak_amount': 0.001,
            'leak_time': 3600 * 12,
        }  # sina needs to be implemented
        self.settings['pump_damage_discovery_model'] = {
            'method': 'time_based',
            'time_discovery_ratio': pd.Series([1], index=[3600 * n for n in [0]]),
        }  # sina needs to be implemented
        self.settings['tank_damage_discovery_model'] = {
            'method': 'time_based',
            'time_discovery_ratio': pd.Series([1], index=[3600 * n for n in [0]]),
        }  # sina needs to be implemented
        self.settings['Gnode_damage_discovery_model'] = {
            'method': 'time_based',
            'time_discovery_ratio': pd.Series([1], index=[3600 * n for n in [0]]),
        }  # Sina GNode Discovery is not here! Must be appleid in the GUI
        self.settings['reservoir_damage_discovery_model'] = {
            'method': 'time_based',
            'time_discovery_ratio': pd.Series([1], index=[3600 * n for n in [0]]),
        }  # Sina GNode Discovery is not here! Must be appleid in the GUI
        self.settings['crew_out_of_zone_travel'] = (
            False  # sina needs to be implemented in the code
        )
        self.settings['crew_travel_speed'] = (
            16.66666  # unit: ft/s approximately 18 km/h. The unit is [coordinate unit] per seconds. # sina needs to be implemented in the code
        )

        self.settings['equavalant_damage_diameter'] = 1
        self.settings['pipe_damage_diameter_factor'] = 1


class Settings:
    def __init__(self):
        self.process = Process_Settings()
        self.scenario = Scenario_Settings()
        self.overrides = {}

    def __setitem__(self, key, data):
        if key in self.process.settings:
            self.process.settings[key] = data
        elif key in self.scenario.settings:
            self.scenario.settings[key] = data
        else:
            raise AttributeError(repr(key) + ' is not in the Settings.')

    def __getitem__(self, key):
        if key in self.process.settings:
            if self.scenario != None:
                if key in self.scenario.settings:
                    raise ValueError(
                        str(key) + ' in both the process and scenario settings.'
                    )

            return self.process.settings[key]
        elif self.scenario != None:
            if key in self.scenario.settings:
                return self.scenario.settings[key]

        raise ValueError(str(key) + ' NOT in either process and scenario settings.')

    def __contains__(self, key):
        if key in self.process.settings:
            return True
        elif self.scenario != None:
            if key in self.scenario.settings:
                return True

        return False

    def importJsonSettings(self, json_file_path):
        """Read a settinsg json file and import the data

        Args:
        ----
            json_file_path (path): JSON file path

        """
        with open(json_file_path) as f:
            settings_data = json.load(f)

        if not isinstance(settings_data, dict):
            raise ValueError(
                'Wrong JSON file type for the settings. The settings JSOn file must be an OBJECT file type.'
            )

        for key, val in settings_data.items():
            if key not in self:
                raise ValueError(
                    f'REWET settinsg does not have "{key}" as a settings key'
                )

            print(key, val)
            if (
                key
                in [
                    'pipe_damage_discovery_model',
                    'node_damage_discovery_model',
                    'pump_damage_discovery_model',
                    'tank_damage_discovery_model',
                ]
                and val['method'] == 'time_based'
            ):
                val['time_discovery_ratio'] = pd.Series(
                    [line[0] for line in val['time_discovery_ratio']],
                    index=[line[1] for line in val['time_discovery_ratio']],
                )

            self[key] = val

    def importProject(self, project_addr):
        with open(project_addr, 'rb') as f:
            project = pickle.load(f)
        # for k in project.project_settings.scenario.settings:
        # new_value = project.project_settings.scenario[k]
        # old_value = self.scenario[k]
        # print(k + ": " + repr(new_value) + " --> " + repr(old_value) + "\n"+"-----" + repr(type(new_value)) )
        self.process = project.project_settings.process
        self.scenario = project.project_settings.scenario

    def initializeScenarioSettings(self, scenario_index):
        if self.process['Parameter_override'] == False:
            return

        list_file = pd.read_excel(self['pipe_damage_file_list'])
        columns = list_file.columns
        parametrs_list = columns.drop(list_default_headers)

        for parameter_name in parametrs_list:
            # to prevent unnamed columns appearing in the warnings
            if 'Unnamed' in parameter_name:
                continue
            override_value = list_file.loc[scenario_index, parameter_name]
            scenario_name = list_file.loc[scenario_index, 'Scenario Name']

            if parameter_name in self:
                try:
                    if type(override_value) != str and np.isnan(override_value):
                        warnings.warn(
                            'REWET Input ERROR in scenario: '
                            + repr(scenario_name)
                            + '\n'
                            + 'The value for '
                            + repr(parameter_name)
                            + ' is empty. The override is IGNORED!'
                        )
                        continue
                except:
                    pass

                if override_value == '':
                    warnings.warn(
                        'REWET Input ERROR in scenario: '
                        + repr(scenario_name)
                        + '\n'
                        + 'The value for '
                        + repr(parameter_name)
                        + ' is empty. The override is IGNORED!'
                    )
                    continue

                self[parameter_name] = override_value
            else:
                splited_parameter_name = parameter_name.split(':')
                number_of_words = len(splited_parameter_name)

                override_key1 = splited_parameter_name[0]
                override_key2 = splited_parameter_name[1]

                if number_of_words != 2:
                    raise ValueError(
                        'REWET Input ERROR in scenario: '
                        + repr(scenario_name)
                        + '\n'
                        + 'The parameter '
                        + repr(parameter_name)
                        + ' is not an acceptable parameter'
                    )

                if override_key1 == None:
                    raise ValueError(
                        'REWET Input ERROR in scenario: '
                        + repr(scenario_name)
                        + '\n'
                        + repr(parameter_name)
                        + ' is not an acceptable parameter'
                    )

                if override_key1.upper() not in acceptable_override_list:
                    warnings.warn(
                        'REWET Input ERROR in scenario: '
                        + repr(scenario_name)
                        + '\n'
                        + repr(override_key1)
                        + ' is not an acceptable parameter. The override is IGNORED!'
                        + '\n'
                        + 'Acceptable override parameters are '
                        + repr(acceptable_override_list)
                    )
                    continue

                try:
                    if type(override_value) != str and np.isnan(override_value):
                        warnings.warn(
                            'REWET Input ERROR in scenario: '
                            + repr(scenario_name)
                            + '\n'
                            + 'The value for '
                            + repr(parameter_name)
                            + ' is empty. The override is IGNORED!'
                        )
                        continue
                except:
                    pass

                if override_value == '':
                    warnings.warn(
                        'REWET Input ERROR in scenario: '
                        + repr(scenario_name)
                        + '\n'
                        + 'The value for '
                        + repr(parameter_name)
                        + ' is empty. The override is IGNORED!'
                    )
                    continue

                if override_key1.upper() == 'POINTS':
                    if override_key2 == None:
                        raise ValueError(
                            'REWET Input ERROR in scenario: '
                            + repr(scenario_name)
                            + '\n'
                            + 'You should provide a Points Group Name for POINTS override key. WARNING: If POINTS Group Name mismatch, it may not take any effect'
                            + '\n'
                        )

                    point_list = self.getOverridePointsList(
                        override_value, scenario_name
                    )
                    if len(point_list) > 0:
                        if 'POINTS' in self.overrides:
                            self.overrides['POINTS'][override_key2] = point_list
                        else:
                            self.overrides['POINTS'] = {override_key2: point_list}
                    else:
                        warnings.warn(
                            'REWET Input ERROR in scenario: '
                            + repr(scenario_name)
                            + '\n'
                            + 'The Override Point Group has no valid input; thus, the override is ignored!'
                        )

                # =============================================================================
                #                 elif override_key1.upper() == "CREWSPEED":
                #                     if override_key2 == None:
                #                         raise ValueError("REWET Input ERROR in scenario: " + repr(scenario_name) + "\n" + "You should provide a Crew Speed for CREWSPEED override key." + "\n")
                #
                #                     crew_speed = self.getOverrideCrewSpeed(override_value, scenario_name)
                #                     if crew_speed != None:
                #                         self.overrides["CREWSPEED"] = crew_speed
                #                     else:
                #                         warnings.warn("REWET Input ERROR in scenario: " + repr(scenario_name) + "\n" + "SPEEDCREW is not valid; thus, the override is ignored!")
                # =============================================================================
                else:
                    raise ValueError('Unknown overrise key')

    def getOverridePointsList(self, points_list_str, scenario_name):
        point_list = []

        points_list_str = points_list_str.strip()
        points_list_str = points_list_str.split()

        for word in points_list_str:
            if ':' not in word:
                warnings.warn(
                    'REWET Input ERROR in scenario: '
                    + repr(scenario_name)
                    + '\n'
                    + word
                    + " must be two numbeers speerated by one ':' showing X:Y coordinate. "
                    + word
                    + ' is ignored!'
                )
                continue

            splited_word = word.split(':')

            if len(splited_word) > 2:
                warnings.warn(
                    'REWET Input ERROR in scenario: '
                    + repr(scenario_name)
                    + '\n'
                    + word
                    + " must be two numbeers speerated by ONE ':' showing X:Y coordinate. "
                    + word
                    + ' is ignored!'
                )
                continue

            x_coord = splited_word[0]
            y_coord = splited_word[1]

            try:
                x_coord = float(x_coord)
            except:
                warnings.warn(
                    'REWET Input ERROR in scenario: '
                    + repr(scenario_name)
                    + '\n'
                    + x_coord
                    + ' in '
                    + word
                    + " must be a number speerated by ONE ':' showing X:Y coordinate. "
                    + word
                    + ' is ignored!'
                )
                continue

            try:
                y_coord = float(y_coord)
            except:
                warnings.warn(
                    'REWET Input ERROR in scenario: '
                    + repr(scenario_name)
                    + '\n'
                    + y_coord
                    + ' in '
                    + word
                    + " must be a number speerated by ONE ':' showing X:Y coordinate. "
                    + word
                    + ' is ignored!'
                )
                continue

            point_list.append((x_coord, y_coord))

        return point_list

    def getOverrideCrewSpeed(self, crew_speed_str, scenario_name):
        crew_speed_str = crew_speed_str.strip()

        if len(crew_speed_str.split()) > 1:
            warnings.warn(
                'REWET Input ERROR in scenario: '
                + repr(scenario_name)
                + '\n'
                + crew_speed_str
                + ' must be ONE single number. Space detected!'
            )
            return None

        try:
            crew_speed = float(crew_speed_str)
        except:
            warnings.warn(
                'REWET Input ERROR in scenario: '
                + repr(scenario_name)
                + '\n'
                + crew_speed
                + ' must be a number.'
            )
            return None

        return crew_speed
