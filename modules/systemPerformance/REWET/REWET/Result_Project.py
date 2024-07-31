# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 15:00:31 2022

@author: snaeimi
"""

import wntrfr
import pandas as pd
import numpy as np
import os
import pickle
from collections import OrderedDict
import copy

# import Report_Reading
from Output.Map import Map
from Output.Map import Helper
from Output.Raw_Data import Raw_Data
from Output.Curve import Curve
from Output.Crew_Report import Crew_Report
from Output.Result_Time import Result_Time
import Input.Input_IO as io
from Project import Project as MainProject


class Project_Result(Map, Raw_Data, Curve, Crew_Report, Result_Time):
    def __init__(
        self,
        project_file_addr,
        result_directory=None,
        ignore_not_found=False,
        to_neglect_file=None,
        node_col='',
        result_file_dir=None,
        iObject=False,
    ):
        if iObject == False:
            self.readPorjectFile(project_file_addr)
        else:
            self.project = copy.deepcopy(project_file_addr)

        if result_file_dir != None:
            self.project.project_settings.process.settings['result_directory'] = (
                result_file_dir
            )
            # self.project.scenario_list = io.read_damage_list(self.project.project_settings.process['pipe_damage_file_list'   ], self.project.project_settings.process['pipe_damage_file_directory'])
        # print(self.project.scenario_list)
        self.project.scenario_list = self.project.scenario_list.set_index(
            'Scenario Name'
        )

        self.demand_node_name_list = []
        self._list = []
        self.pipe_damages = {}
        self.node_damages = {}
        self.pump_damages = {}
        self.tank_damages = {}
        # self.time_size                = {}
        self.demand_node_size = {}
        self.index_to_scen_name = {}
        self.data = OrderedDict()
        self.registry = OrderedDict()
        self.scenario_prob = {}
        # self.scenario_set             = {}
        self.empty_scenario_name_list = set()
        self._delivery_data = None
        self._quality_data = None
        self._DLQNIndexPopulation = {}
        self._population_data = None
        self.exceedance_map_intermediate_data = None
        self.rest_data = None
        self._RequiredDemandForAllNodesandtime = {}
        self.demand_ratio = self.project.project_settings.process['demand_ratio']
        self.scn_name_list_that_result_file_not_found = []
        self.wn = wntrfr.network.WaterNetworkModel(
            self.project.project_settings.process['WN_INP']
        )

        self.result_directory = self.project.project_settings.process[
            'result_directory'
        ]

        if not isinstance(result_directory, type(None)):
            self.result_directory = result_directory

        to_neglect = []
        if to_neglect_file != None and False:  # sina hereeeee bug dadi amedane
            raise
            file_data = pd.read_excel(to_neglect_file)
            to_neglect = file_data[node_col].to_list()

        for node_name in self.wn.junction_name_list:
            node = self.wn.get_node(node_name)
            if (
                node.demand_timeseries_list[0].base_value > 0
                and node_name not in to_neglect
            ):
                self.demand_node_name_list.append(node_name)

        self.node_name_list = self.wn.node_name_list.copy()
        ret_val = self.checkForNotExistingFile(ignore_not_found)
        self.prepareData()
        return ret_val

    def readPorjectFile(self, project_file_addr):
        print(project_file_addr)
        with open(project_file_addr, 'rb') as f:
            self.project = pickle.load(f)

    def loadPopulation(self, popuation_data, node_id_header, population_header):
        pop = popuation_data.copy()
        pop = pop.set_index(node_id_header)
        pop = pop[population_header]
        self._population_data = pop

    def checkForNotExistingFile(self, ignore_not_found):
        self.scn_name_list_that_result_file_not_found = []

        result_directory = self.result_directory
        # print(self.project.scenario_list)
        for scn_name, row in self.project.scenario_list.iterrows():
            scenario_registry_file_name = scn_name + '_registry.pkl'
            # print(result_directory)
            # print(scenario_registry_file_name)
            registry_file_data_addr = os.path.join(
                result_directory, scenario_registry_file_name
            )
            if not os.path.exists(registry_file_data_addr):
                self.scn_name_list_that_result_file_not_found.append(scn_name)

        if len(self.scn_name_list_that_result_file_not_found) > 0:
            if ignore_not_found:
                # print(str(len(self.scn_name_list_that_result_file_not_found)) +" out of "+ repr(len(self.project.scenario_list)) +" Result Files are not found and ignored" )
                # print(self.scn_name_list_that_result_file_not_found)
                pass
                # self.project.scenario_list.drop(self.scn_name_list_that_result_file_not_found, inplace=True)
            else:
                raise ValueError(
                    'Res File Not Found: '
                    + repr(self.scn_name_list_that_result_file_not_found)
                    + ' in '
                    + repr(result_directory)
                )

    def prepareData(self):
        i = 0
        # result_directory = self.project.project_settings.process['result_directory']
        # self.project.scenario_list = self.project.scenario_list.iloc[0:20]
        for scn_name, row in self.project.scenario_list.iterrows():
            self._RequiredDemandForAllNodesandtime[scn_name] = None
            # settings_file_name = scn_name+'.xlsx'
            # settings_file_addr = os.path.join(result_directory, settings_file_name)
            # scenario_set       = pd.read_excel(settings_file_addr)
            # self.scenario_set[scn_name] = scenario_set

            self.data[scn_name] = None
            self.registry[scn_name] = None

            # self.time_size[scn_name]      = len(self.data[scn_name].node['demand'].index)
            self.index_to_scen_name[i] = scn_name
            i += 1

            self.scenario_prob[scn_name] = self.project.scenario_list.loc[
                scn_name, 'Probability'
            ]

            """
                ATTENTION: We need probability for any prbablistic result
            """

    def loadScneariodata(self, scn_name):
        if self.data[scn_name] != None:
            return
        print('loading scenario ' + str(scn_name))
        result_directory = self.result_directory
        # scenario_registry_file_name = scn_name+"_registry.pkl"
        # registry_file_data_addr = os.path.join(result_directory, scenario_registry_file_name)
        scenario_registry_file_name = scn_name + '_registry.pkl'
        reg_addr = os.path.join(result_directory, scenario_registry_file_name)
        try:
            with open(reg_addr, 'rb') as f:
                # print(output_addr)
                reg_file_data = pickle.load(f)
            self.registry[scn_name] = reg_file_data
            res_file_data = self.registry[scn_name].result
        except:
            scenario_registry_file_name = scn_name + '.res'
            res_addr = os.path.join(result_directory, scenario_registry_file_name)
            with open(res_addr, 'rb') as f:
                res_file_data = pickle.load(f)
        # scenario_registry_file_name = scn_name+".res"
        # res_addr = os.path.join(result_directory, scenario_registry_file_name)
        # with open(res_addr, 'rb') as f:
        # print(output_addr)
        # res_file_data = pickle.load(f)
        # res_file_data.node['head']    = None
        # res_file_data.node['quality'] = None
        # res_file_data = self.registry[scn_name].result
        self.remove_maximum_trials(res_file_data)
        self.data[scn_name] = res_file_data

    def readData(self):
        # i=0
        self.project.scenario_list = self.project.scenario_list.iloc[0:2]
        result_directory = self.result_directory

        for scn_name, row in self.project.scenario_list.iterrows():
            self._RequiredDemandForAllNodesandtime[scn_name] = None
            scenario_registry_file_name = scn_name + '_registry.pkl'
            registry_file_data_addr = os.path.join(
                result_directory, scenario_registry_file_name
            )

            with open(registry_file_data_addr, 'rb') as f:
                if not os.path.exists(registry_file_data_addr):
                    raise ValueError(
                        'Registry File Not Found: ' + str(registry_file_data_addr)
                    )
                self.registry[scn_name] = pickle.load(f)

            # self.pipe_damages[scn_name] = current_scenario_registry.damage.pipe_all_damages
            # self.node_damages[scn_name] = current_scenario_registry.node_damage
            # self.pump_damages[scn_name] = current_scenario_registry.damaged_pumps
            # self.tank_damages[scn_name] = current_scenario_registry.tank_damage

            # res_addr = os.path.join(result_directory, scn_name+'.res')

            # with open(res_addr, 'rb') as f:
            # print(output_addr)
            # res_file_data = pickle.load(f)

            # settings_file_name = scn_name+'.xlsx'
            # settings_file_addr = os.path.join(result_directory, settings_file_name)
            # scenario_set       = pd.read_excel(settings_file_addr)
            # self.scenario_set[scn_name] = scenario_set

            # res_file_data.node['head']    = None
            # res_file_data.node['quality'] = None
            res_file_data = self.registry[scn_name]
            self.remove_maximum_trials(res_file_data)
            self.data[scn_name] = res_file_data
            # self.time_size[scn_name]      = len(self.data[scn_name].node['demand'].index)
            # self.index_to_scen_name[i]    = scn_name
            # i+=1

            self.scenario_prob[scn_name] = self.project.scenario_list.loc[
                scn_name, 'Probability'
            ]

            """
                ATTENTION: We need probability for any prbablistic result
            """
            print(str(scn_name) + ' loaded')

    def remove_maximum_trials(self, data):
        all_time_list = data.maximum_trial_time
        result_time_list = data.node['demand'].index.to_list()
        result_time_max_trailed_list = [
            time for time in result_time_list if time in all_time_list
        ]

        for att in data.node:
            all_time_list = data.maximum_trial_time
            result_time_list = data.node[att].index.to_list()
            result_time_max_trailed_list = list(
                set(result_time_list).intersection(set(all_time_list))
            )
            result_time_max_trailed_list.sort()
            if len(result_time_max_trailed_list) > 0:
                # print(result_time_max_trailed_list)
                att_data = data.node[att]
                att_data.drop(result_time_max_trailed_list, inplace=True)
                data.node[att] = att_data

        for att in data.link:
            all_time_list = data.maximum_trial_time
            result_time_list = data.link[att].index.to_list()
            result_time_max_trailed_list = [
                time for time in result_time_list if time in all_time_list
            ]
            att_data = data.link[att]
            att_data.drop(result_time_max_trailed_list, inplace=True)
            data.link[att] = att_data

        flow_balance = data.node['demand'].sum(axis=1)

        time_to_drop = flow_balance[abs(flow_balance) >= 0.01].index

        # result_time_list = data.node['demand'].index.to_list()
        # = [ time for time in result_time_list if time in all_time_list]

        for att in data.node:
            # all_time_list = data.maximum_trial_time
            result_time_list = data.node[att].index.to_list()
            result_time_max_trailed_list = list(
                set(result_time_list).intersection(set(time_to_drop))
            )
            result_time_max_trailed_list.sort()
            if len(result_time_max_trailed_list) > 0:
                # print(result_time_max_trailed_list)
                att_data = data.node[att]
                att_data.drop(result_time_max_trailed_list, inplace=True)
                data.node[att] = att_data

        for att in data.link:
            # all_time_list = data.maximum_trial_time
            result_time_list = data.link[att].index.to_list()
            result_time_max_trailed_list = list(
                set(result_time_list).intersection(set(time_to_drop))
            )
            result_time_max_trailed_list.sort()
            if len(result_time_max_trailed_list) > 0:
                att_data = data.link[att]
                att_data.drop(result_time_max_trailed_list, inplace=True)
                data.link[att] = att_data

    def remove_maximum_trials_demand_flow(self, data):
        flow_balance = data.node['demand'].sum(axis=1)

        time_to_drop = flow_balance[abs(flow_balance) >= 0.01].index

        # result_time_list = data.node['demand'].index.to_list()
        # = [ time for time in result_time_list if time in all_time_list]

        for att in data.node:
            # all_time_list = data.maximum_trial_time
            result_time_list = data.node[att].index.to_list()
            result_time_max_trailed_list = [
                time for time in result_time_list if time in time_to_drop
            ]
            print(result_time_max_trailed_list)
            att_data = data.node[att]
            att_data.drop(result_time_max_trailed_list, inplace=True)
            data.node[att] = att_data

        for att in data.link:
            # all_time_list = data.maximum_trial_time
            result_time_list = data.link[att].index.to_list()
            result_time_max_trailed_list = [
                time for time in result_time_list if time in time_to_drop
            ]
            att_data = data.link[att]
            att_data.drop(result_time_max_trailed_list, inplace=True)
            data.link[att] = att_data

    def readPopulation(
        self,
        population_xlsx_addr='demandNode-Northridge.xlsx',
        demand_node_header='NodeID',
        population_header='#Customer',
    ):
        pop = pd.read_excel(population_xlsx_addr)
        pop = pop.set_index(demand_node_header)
        pop = pop['#Customer']
        self._population_data = pop
        demand_node_without_population = [
            node_name
            for node_name in self.demand_node_name_list
            if node_name not in pop.index
        ]
        if len(demand_node_without_population) > 0:
            raise ValueError(
                'The following demand nodes are not population data: '
                + repr(demand_node_without_population)
            )

    def getRequiredDemandForAllNodesandtime(self, scn_name):
        """
        **********
        ATTENTION: We Assume that all scnearios have teh same time indexing
        **********

        Calculates and return required demands for all nodes in all the times steps

        Returns
        -------
        req_node_demand : Pandas DataFrame
            Demand for all nodes and in all time

        """
        self.loadScneariodata(scn_name)
        demand_ratio = self.demand_ratio
        if type(self._RequiredDemandForAllNodesandtime[scn_name]) != type(None):
            return self._RequiredDemandForAllNodesandtime[scn_name]
        undamaged_wn = self.wn
        time_index = self.data[scn_name].node['demand'].index
        # req_node_demand   = pd.DataFrame(index=time_index.unique())
        default_pattern = undamaged_wn.options.hydraulic.pattern
        node_pattern_list = pd.Series(
            index=undamaged_wn.junction_name_list, dtype=str
        )
        _size = len(self.demand_node_name_list)
        i = 0
        # req_node_demand = req_node_demand.transpose()

        all_base_demand = []
        all_node_name_list = []

        while i < _size:
            node_name = self.demand_node_name_list[i]
            # print(i)
            i += 1
            node = undamaged_wn.get_node(node_name)
            pattern_list = node.demand_timeseries_list.pattern_list()
            if pattern_list[0] != None:
                node_pattern_list[node_name] = pattern_list[0].name
            elif pattern_list[0] == None and default_pattern != None:
                node_pattern_list[node_name] = str(default_pattern)
            else:
                node_pattern_list[node_name] = None
                base_demand = node.base_demand * demand_ratio
                all_base_demand.append([base_demand for i in time_index])
                all_node_name_list.append(node_name)
                # temp=pd.DataFrame(data = base_demand, index = time_index, columns = [node_name])
                # req_node_demand = req_node_demand.append(temp.transpose())
        # constant_base_demand = [constant_base_demand for i in time_index]
        node_pattern_list = node_pattern_list.dropna()

        patterns_list = node_pattern_list.unique()
        multiplier = pd.DataFrame(index=time_index, columns=patterns_list)

        for pattern_name in iter(patterns_list):
            cur_pattern = undamaged_wn.get_pattern(pattern_name)
            time_index = time_index.unique()
            for time in iter(time_index):
                multiplier[pattern_name].loc[time] = cur_pattern.at(time)

        variable_base_demand = []
        variable_node_name_list = []
        for node_name, pattern_name in node_pattern_list.items():
            cur_node_req_demand = (
                multiplier[pattern_name]
                * undamaged_wn.get_node(node_name)
                .demand_timeseries_list[0]
                .base_value
                * demand_ratio
            )

            all_node_name_list.append(node_name)
            all_base_demand.append(cur_node_req_demand.to_list())
            # cur_node_req_demand.name = node_name
            # cur_node_req_demand=pd.DataFrame(cur_node_req_demand).transpose()
            # req_node_demand = req_node_demand.append(cur_node_req_demand)
        # variable_base_demand = np.array(variable_base_demand).transpose().tolist()
        req_node_demand = pd.DataFrame(
            columns=time_index, index=all_node_name_list, data=all_base_demand
        )
        req_node_demand = req_node_demand.transpose()
        # constant_node_demand_df = pd.DataFrame(data = constant_base_demand, index = time_index, columns = constant_node_name_list)
        # variable_node_demand_df = pd.DataFrame(data = variable_base_demand, index = time_index, columns = variable_node_name_list)
        # if len(variable_base_demand) > 0 and len(variable_base_demand) == 0:
        # req_node_demand = constant_node_demand_df
        # elif len(variable_base_demand) == 0 and len(variable_base_demand) > 0:
        # req_node_demand = variable_base_demand
        # elif    len(variable_base_demand) == 0 and len(variable_base_demand) == 0:
        # req_node_demand = constant_node_demand_df
        # else:
        # req_node_demand = pd.concat([constant_node_demand_df.transpose(), variable_node_demand_df.transpose()]).transpose()

        # print(len(all_node_name_list))
        # print(len(constant_base_demand))
        # print(len(variant_base_demand))
        # print("************************")
        # all_base_demand = constant_base_demand

        # req_node_demand = pd.DataFrame(index=time_index, columns=all_node_name_list, data=all_base_demand)
        # req_node_demand = req_node_demand.transpose()
        self._RequiredDemandForAllNodesandtime[scn_name] = req_node_demand.filter(
            self.demand_node_name_list
        )
        return self._RequiredDemandForAllNodesandtime[scn_name]
        self._RequiredDemandForAllNodesandtime[scn_name] = req_node_demand.filter(
            self.demand_node_name_list
        )
        return self._RequiredDemandForAllNodesandtime[scn_name]

    def AS_getDLIndexPopulation(
        self, iPopulation='No', ratio=False, consider_leak=False, leak_ratio=0.75
    ):
        scenario_list = list(self.data.keys())
        all_scenario_DL_data = {}
        for scn_name in scenario_list:
            cur_scn_DL = self.getDLIndexPopulation_4(
                scn_name,
                iPopulation=iPopulation,
                ratio=ratio,
                consider_leak=consider_leak,
                leak_ratio=leak_ratio,
            )
            cur_scn_DL = cur_scn_DL.to_dict()
            all_scenario_DL_data[scn_name] = cur_scn_DL

        return pd.DataFrame.from_dict(all_scenario_DL_data)

    def AS_getQNIndexPopulation(
        self, iPopulation='No', ratio=False, consider_leak=False, leak_ratio=0.75
    ):
        scenario_list = list(self.data.keys())
        all_scenario_QN_data = {}
        for scn_name in scenario_list:
            self.loadScneariodata(scn_name)
            cur_scn_QN = self.getQNIndexPopulation_4(
                scn_name,
                iPopulation=iPopulation,
                ratio=ratio,
                consider_leak=consider_leak,
                leak_ratio=leak_ratio,
            )
            cur_scn_QN = cur_scn_QN.to_dict()
            all_scenario_QN_data[scn_name] = cur_scn_QN

        return pd.DataFrame.from_dict(all_scenario_QN_data)

    def AS_getOutage_4(
        self,
        LOS='DL',
        iConsider_leak=False,
        leak_ratio=0,
        consistency_time_window=7200,
    ):
        scenario_list = list(self.data.keys())
        all_scenario_outage_data = {}
        i = 0
        for scn_name in scenario_list:
            cur_scn_outage = self.getOutageTimeGeoPandas_4(
                scn_name,
                LOS=LOS,
                iConsider_leak=iConsider_leak,
                leak_ratio=leak_ratio,
                consistency_time_window=consistency_time_window,
            )
            cur_scn_outage = cur_scn_outage['restoration_time'].to_dict()
            all_scenario_outage_data[scn_name] = cur_scn_outage
            i += 1

        return pd.DataFrame.from_dict(all_scenario_outage_data)

    def PR_getBSCPercentageExcedanceCurce(self, data_frame, restoration_percentage):
        max_time = data_frame.max().max()

        restore_time = {}

        if type(self._population_data) == type(None):
            demand_node_name_list = data_frame.index
            population = pd.Series(index=demand_node_name_list, data=1)
        else:
            population = self._population_data
        population = population.loc[data_frame.index]

        population_dataframe = dict(
            zip(
                data_frame.columns,
                [population.to_dict() for i in data_frame.columns],
            )
        )
        population_dataframe = pd.DataFrame.from_dict(population_dataframe)
        total_population = population.sum()

        for t in range(0, int(max_time), 3600):
            satisfies_nodes_scnearios = data_frame <= t
            satisfies_nodes_scnearios = (
                satisfies_nodes_scnearios * population_dataframe
            )
            scenario_percentages = (
                satisfies_nodes_scnearios.sum() / total_population * 100
            )
            satisfied_scenarios = (
                scenario_percentages[scenario_percentages >= restoration_percentage]
            ).index
            already_recorded_scenarios = set(restore_time.keys())
            new_scenarios = set(satisfied_scenarios) - already_recorded_scenarios

            new_record = dict(
                zip(new_scenarios, [t for k in range(len(new_scenarios))])
            )
            restore_time.update(new_record)

        already_recorded_scenarios = set(restore_time.keys())
        unsatisfied_scenarios = (
            set(self.scenario_prob.keys()) - already_recorded_scenarios
        )
        new_record = dict(
            zip(
                unsatisfied_scenarios, [t for k in range(len(unsatisfied_scenarios))]
            )
        )
        restore_time.update(new_record)

        restore_data = pd.DataFrame.from_dict({'restore_time': restore_time})

        restore_data['restore_time'] = restore_data.loc[
            list(self.scenario_prob.keys()), 'restore_time'
        ]
        restore_data['prob'] = list(self.scenario_prob.values())
        restore_data.sort_values('restore_time', ascending=False, inplace=True)
        ep_mat = Helper.EPHelper(restore_data['prob'].to_numpy())
        restore_data['EP'] = ep_mat

        return restore_data

    def PR_getCurveExcedence(
        self,
        data_frame,
        result_type='mean',
        daily=False,
        min_time=0,
        max_time=24 * 3600 * 1000,
    ):
        data_size = len(data_frame.columns)
        table_temp = []

        for i in np.arange(data_size):
            scn_name = data_frame.columns[i]
            prob = self.scenario_prob[scn_name]
            cur_scn_data = data_frame[scn_name]
            dmg_index_list = []

            cur_scn_data = cur_scn_data[cur_scn_data.index >= min_time]
            cur_scn_data = cur_scn_data[cur_scn_data.index <= max_time]

            if daily == True:
                cur_scn_data = self.getResultSeperatedDaily(cur_scn_data)

            if result_type == 'mean':
                cur_mean_res = cur_scn_data.mean()
                if type(cur_mean_res) != pd.core.series.Series:
                    temp_res = {'mean_dmg': cur_mean_res}
                    dmg_index_list.append('mean_dmg')
                else:
                    temp_res = {}
                    for day_time, value in cur_mean_res.iteritems():
                        temp_dmg_index = 'mean_dmg_' + day_time
                        temp_res.update({temp_dmg_index: value})
                        dmg_index_list.append(temp_dmg_index)
            elif result_type == 'min':
                dmg_min_res = cur_scn_data.min()
                if type(dmg_min_res) != pd.core.series.Series:
                    temp_res = {'min_dmg': dmg_min_res}
                    dmg_index_list.append('min_dmg')
                else:
                    temp_res = {}
                    for day_time, value in dmg_min_res.iteritems():
                        temp_dmg_index = 'min_dmg_' + day_time
                        temp_res.update({temp_dmg_index: value})
                        dmg_index_list.append(temp_dmg_index)
            elif result_type == 'max':
                dmg_max_res = cur_scn_data.min()
                if type(dmg_max_res) != pd.core.series.Series:
                    temp_res = {'max_dmg': dmg_max_res}
                    dmg_index_list.append('max_dmg')
                else:
                    temp_res = {}
                    for day_time, value in dmg_max_res.iteritems():
                        temp_dmg_index = 'max_dmg_' + day_time
                        temp_res.update({temp_dmg_index: value})
                        dmg_index_list.append(temp_dmg_index)
            else:
                raise ValueError('Unknown group method: ' + repr(result_type))

            loop_res = {'prob': prob, 'index': scn_name}
            loop_res.update(temp_res)
            table_temp.append(loop_res)

        table = pd.DataFrame.from_dict(table_temp).set_index('index')
        res = pd.DataFrame(
            index=[i for i in range(0, len(table.index))], dtype=np.float64
        )
        for dmg_name in dmg_index_list:
            select_columns = ['prob']
            select_columns.extend([dmg_name])
            loop_table = table[select_columns]
            loop_table.sort_values(dmg_name, inplace=True)

            ep_mat = Helper.EPHelper(loop_table['prob'].to_numpy())
            res[dmg_name] = loop_table[dmg_name].to_numpy()
            res[dmg_name + '_EP'] = ep_mat

        return res

    def getResultSeperatedDaily(self, data, begin_time=0):
        data = data[data.index >= begin_time]
        data.index = (data.index - begin_time) / (24 * 3600)

        res_data = []
        res_day = []

        for day_iter in range(0, np.int64(np.ceil(np.max(data.index)))):
            day_data = data[(data.index >= day_iter) & (data.index <= day_iter + 1)]
            res_data.append(day_data.to_list())
            res_day.append(str(day_iter) + '-' + str(day_iter + 1))

        return pd.DataFrame(res_data, index=res_day).transpose()
