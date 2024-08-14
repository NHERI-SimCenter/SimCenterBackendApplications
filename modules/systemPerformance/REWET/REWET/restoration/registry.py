"""Created on Sat Dec 26 03:22:21 2020

@author: snaeimi
"""  # noqa: CPY001, D400, INP001

import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
from restoration.restorationlog import RestorationLog

logger = logging.getLogger(__name__)


class Registry:  # noqa: D101, PLR0904
    def __init__(self, WaterNetwork, settings, demand_node_name_list, scenario_name):  # noqa: N803
        self._registry_version = 0.15
        self.wn = WaterNetwork
        self.settings = settings
        self.demand_node_name_list = demand_node_name_list
        self.scenario_name = scenario_name
        # self.EQCoordinates         = (6398403.298, 1899243.660)
        # self.proximity_points      = {'WaterSource':[(6435903.606431,1893248.592426),(6441950.711447,1897369.022871),
        # (6424377.955317,1929513.408731),(6467146.075381,1816296.452238),
        # (6483259.266246,1803209.907606),(6436359.6420960,1905761.7390040),
        # (6492204.110122,1758379.158018),(6464169.549436,1738989.098520),
        # (6504097.778564,1875687.031985),(6414434.124,1929805.346),
        # (6412947.370,1936851.950)]}
        self._pipe_break_node_coupling = {}  # for broken points that each has two nodes
        self._break_point_attached_to_mainPipe = []  # for broken points to show which node is attached to the main point. For easier and faster coding in removals of damage
        # self._occupancy = pd.Series() # for agent occupency
        # self._pipe_RepairAgentNameRegistry=[] # MAYBE NOT NEEDED for agent occupency
        self._tank_damage_table = pd.DataFrame(columns=['damage_type'])
        self._reservoir_damage_table = pd.DataFrame(columns=['damage_type'])
        self._pump_damage_table = pd.DataFrame(
            columns=['damage_type', 'element_name', 'start_node', 'end_node']
        )
        self._gnode_damage_table = pd.DataFrame(columns=['damage_type'])
        self._pipe_damage_table = pd.DataFrame(
            columns=[
                'damage_type',
                'damage_sub_type',
                'Orginal_element',
                'attached_element',
                'number',
                'LeakAtCheck',
            ]
        )
        self._pipe_data = pd.DataFrame(columns=['diameter'])
        self._node_damage_table = pd.DataFrame(
            columns=['Demand1', 'Demand2', 'Number_of_damages']
        )
        self._pipe_break_history = pd.DataFrame(
            columns=['Pipe_A', 'Pipe_B', 'Orginal_pipe', 'Node_A', 'Node_B']
        )
        self._pipe_leak_history = pd.DataFrame(
            columns=['Pipe_A', 'Pipe_B', 'Orginal_pipe', 'Node_name']
        )
        self._long_task_data = pd.DataFrame(
            columns=['Node_name', 'Action', 'Entity', 'Time', 'cur_agent_name']
        )
        self.all_node_table = pd.DataFrame(
            columns=['X_COORD', 'Y_COORD'], dtype=float
        )
        self.pre_event_demand_met = pd.DataFrame(dtype=float)
        self.hydraulic_significance = pd.Series(dtype=float)
        self.if_first_event_occured = 0
        self.restoration_log_book = RestorationLog(settings)
        self.explicit_leak_node = {}
        self.demand_node_name_list = []
        self.all_node_name_list = WaterNetwork.node_name_list.copy()
        # self.demand_node_users         = pd.Series()
        # self.minimum_time_devision     = 60*60
        self.nodal_equavalant_diameter = None
        self.original_pipe_data = {}
        self.result = None
        self.active_pipe_damages = OrderedDict()
        self.active_nodal_damages = OrderedDict()
        self.active_collectives = pd.DataFrame(columns=['action', 'Orginal_pipe'])
        self.virtual_node_data = OrderedDict()
        self._nodal_data = OrderedDict()
        self.result = None
        self.result_dump_file_list = []
        self.Pipe_Damage_restoration_report = []
        self.undamaged_link_node_list = {}

        for name, pipe in WaterNetwork.pipes():
            self._pipe_data.loc[name] = [pipe.diameter]

        for node_name, node in WaterNetwork.junctions():
            if node.demand_timeseries_list[0].base_value > 0.00000008:  # noqa: PLR2004
                self.demand_node_name_list.append(node_name)

        # for demand_node_name in self.demand_node_name_list:
        # self.demand_node_users.loc[demand_node_name]=1

        for node_name, node in WaterNetwork.nodes():
            self.all_node_table.loc[node_name, 'X_COORD'] = node.coordinates[0]
            self.all_node_table.loc[node_name, 'Y_COORD'] = node.coordinates[1]

        for link_name, link in WaterNetwork.links():
            self.undamaged_link_node_list[link_name] = (
                link.start_node_name,
                link.end_node_name,
            )

        # self._restoration_table  = pd.DataFrame(columns = ['node_name','function', 'element_name', 'element_type', 'in_function_index'])
        self._restoration_table = pd.DataFrame(
            columns=['node_name', 'function', 'record_index']
        )
        self._record_registry = []

        self._pipe_damage_table_time_series = OrderedDict()
        self._node_damage_table_time_series = OrderedDict()
        self._tank_level_time_series = OrderedDict()
        self._restoration_reservoir_name_time_series = OrderedDict()
        self.ED_history = pd.Series(dtype='O')  # Equavalant Damage Diameter

        for pipe_name, pipe in WaterNetwork.pipes():
            self.original_pipe_data[pipe_name] = {
                'diameter': pipe.diameter,
                'length': pipe.length,
                'start_node_name': pipe.start_node_name,
                'end_node_name': pipe.end_node_name,
                'roughness': pipe.roughness,
            }

    # =============================================================================
    #     def addElementToRestorationRegistry(self, damaged_node_name, function_name, element_name, elemenet_type, in_function_index):
    #         data = self.__restoration_table
    #         selected_data = data[(data[['node_name', 'element_name', 'element_type']]==[damaged_node_name,element_name,elemenet_type]).all(1))]
    #
    #         if len(selected_data)>1:
    #             raise ValueError('There are data in restroation regustry. Damaged node name: '+damaged_node_name)
    #
    #
    #         temp = pd.Series(data=[damaged_node_name, function_name, element_name, elemenet_type, in_function_index], index=['node_name','function', 'element_name', 'element_type', 'in_function_index'])
    #         self._restoration_table = self._restoration_table.append(temp, ignore_index=True)
    # =============================================================================

    def addRestorationDataOnPipe(self, damage_node_name, time, state):  # noqa: N802, D102
        if self.settings['dmg_rst_data_save'] == True:  # noqa: E712
            orginal_pipe_name = self._pipe_damage_table.loc[
                damage_node_name, 'Orginal_element'
            ]
            time = time / 3600  # noqa: PLR6104
            temp_row = {
                'time': time,
                'pipe_name': orginal_pipe_name,
                'last_state': state,
            }
            self.Pipe_Damage_restoration_report.append(temp_row)

    def addEquavalantDamageHistory(  # noqa: N802, D102
        self,
        node_name,
        new_node_name,
        new_pipe_name,
        equavalant_pipe_diameter,
        number_of_damages,
    ):
        if node_name in self.ED_history:
            raise ValueError('Node_damage already in history')  # noqa: EM101, TRY003

        self.ED_history.loc[node_name] = {
            'new_node_name': new_node_name,
            'new_pipe_name': new_pipe_name,
            'equavalant_pipe_diameter': equavalant_pipe_diameter,
            'initial_number_of_damage': number_of_damages,
            'current_number_of_damage': number_of_damages,
        }

    def getEquavalantDamageHistory(self, node_name):  # noqa: N802, D102
        temp = self.ED_history[node_name]

        if type(temp) != dict:  # noqa: E721
            raise ValueError('probably two damages with the same name: ' + node_name)

        return temp

    def removeEquavalantDamageHistory(self, node_name):  # noqa: N802, D102
        self.ED_history.drop(node_name, inplace=True)  # noqa: PD002

    def isThereSuchOngoingLongJob(self, damaged_node_name, action, entity):  # noqa: N802, D102
        data = self._long_task_data
        temp = data[['Node_name', 'Action', 'Entity']] == [
            damaged_node_name,
            action,
            entity,
        ]
        temp = data[temp.all(1)]

        if len(temp) > 1:
            raise ValueError('More job than 1 in long jobs')  # noqa: EM101, TRY003
        elif len(temp) == 1:  # noqa: RET506
            if abs(temp['Time'].iloc[0]) < 0.01:  # noqa: PLR2004
                raise ValueError('Something Wrong')  # noqa: EM101, TRY003
            else:  # noqa: RET506
                return True
        else:
            return False

    def addLongJob(  # noqa: N802, D102
        self,
        damaged_node_name,
        action,
        entity,
        job_gross_time,
        agent_name,
    ):
        data = self._long_task_data
        temp = data[['Node_name', 'Action', 'Entity', 'Time', 'cur_agent_name']] == [
            damaged_node_name,
            action,
            entity,
            job_gross_time,
            agent_name,
        ]

        if temp.all(1).any():
            raise ValueError(
                'There are currently data on: '
                + damaged_node_name
                + ','
                + action
                + ','
                + entity
            )
        # elif temp['cur_agent_name'].iloc[0]!=None:
        # raise ValueError('There is one agent: '+temp['cur_agent_name'].iloc[0]+' assigned to long job: '+damaged_node_name+','+action+','+entity)

        temp = pd.Series(
            index=['Node_name', 'Action', 'Entity', 'Time', 'cur_agent_name'],
            data=[damaged_node_name, action, entity, job_gross_time, agent_name],
        )
        self._long_task_data = data.append(temp, ignore_index=True)

    def assignAgenttoLongJob(  # noqa: N802, D102
        self,
        damaged_node_name,
        action,
        entity,
        choosed_agent_name,
    ):
        data = self._long_task_data
        temp = data[['Node_name', 'Action', 'Entity']] == [
            damaged_node_name,
            action,
            entity,
        ]
        temp = data[temp.all(1)]

        if len(temp) != 1:
            raise ValueError(
                'There must be one record: '
                + damaged_node_name
                + ','
                + action
                + ','
                + entity
            )

        ind = temp.index[0]
        if (
            self._long_task_data.loc[ind, 'cur_agent_name'] != None  # noqa: E711
            and choosed_agent_name != None  # noqa: E711
        ):
            raise ValueError(
                'Already someone is here '
                + repr(self._long_task_data.loc[ind, 'cur_agent_name'])
            )

        self._long_task_data.loc[ind, 'cur_agent_name'] = choosed_agent_name

    def deductLongJobTime(self, damaged_node_name, action, entity, deduced_time):  # noqa: N802, D102
        deduced_time = int(deduced_time)

        if deduced_time < 0:
            raise ValueError(
                'deductig time must not be less than zero: ' + repr(deduced_time)
            )

        data = self._long_task_data
        temp = data[['Node_name', 'Action', 'Entity']] == [
            damaged_node_name,
            action,
            entity,
        ]

        temp = data[temp.all(1)]

        if len(temp) == 0:
            raise ValueError(
                'There is no long task defined for: '
                + damaged_node_name
                + ', '
                + action
                + ', '
                + entity
            )
        elif len(temp) > 1:  # noqa: RET506
            raise ValueError(
                'There are MORE THAN ONE long task defined for: '
                + damaged_node_name
                + ', '
                + action
                + ', '
                + entity
            )

        ind = temp.index[0]

        if (self._long_task_data.loc[ind, 'Time'] - deduced_time) < 0:
            logger.warning(
                damaged_node_name  # noqa: G003
                + ', '
                + action
                + ', '
                + entity
                + ', '
                + str(self._long_task_data.loc[ind, 'Time'])
                + ', '
                + str(deduced_time)
                + ', '
                + str(self._long_task_data.loc[ind, 'Time'] - deduced_time)
            )
            raise ValueError('Zero reminded time for long task')  # noqa: EM101, TRY003

        self._long_task_data.loc[ind, 'Time'] -= deduced_time

    def getLongJobRemindedTime(self, damaged_node_name, action, entity):  # noqa: N802, D102
        data = self._long_task_data
        temp = data[['Node_name', 'Action', 'Entity']] == [
            damaged_node_name,
            action,
            entity,
        ]

        temp = data[temp.all(1)]

        if len(temp) == 0:
            raise ValueError(
                'There is no long task defined for: '
                + damaged_node_name
                + ','
                + action
                + ','
                + entity
            )
        elif len(temp) > 1:  # noqa: RET506
            raise ValueError(
                'There are MORE THAN ONE long task defined for: '
                + damaged_node_name
                + ','
                + action
                + ','
                + entity
            )

        return temp['Time'].iloc[0]

    def getVacantOnGoingJobs(self, action, entity):  # noqa: N802, D102
        res = []
        data = self._long_task_data
        temp = data[['Action', 'Entity']] == [action, entity]

        temp = data[temp.all(1)]

        for ind, data in temp.iterrows():  # noqa: B007
            if data['cur_agent_name'] == None:  # noqa: E711
                res.append(data['Node_name'])

        return res

    def getdamagedNodesOfPipes(self, damage_type):  # noqa: N802, D102
        if damage_type != 'break' and damage_type != 'leak':  # noqa: PLR1714
            raise ValueError('The damage for pipe is either break or leak.')  # noqa: EM101, TRY003

        if damage_type == 'break':  # noqa: RET503
            return self._pipe_break_history[['Node_A', 'Node_B']]

        elif damage_type == 'leak':  # noqa: RET505
            return self._pipe_leak_history['Node_name']

    def removeLongJob(self, damaged_node_name, action, entity):  # noqa: N802, D102
        data = self._long_task_data
        temp = data[['Node_name', 'Action', 'Entity']] == [
            damaged_node_name,
            action,
            entity,
        ]

        temp = data[temp.all(1)]

        if len(temp) == 0:
            raise ValueError(
                'There is no long task defined for: '
                + damaged_node_name
                + ','
                + action
                + ','
                + entity
            )
        elif len(temp) > 1:  # noqa: RET506
            raise ValueError(
                'There are MORE THAN ONE long task defined for: '
                + damaged_node_name
                + ','
                + action
                + ','
                + entity
            )

        ind = temp.index[0]

        self._long_task_data.drop(ind, inplace=True)  # noqa: PD002

    def addFunctionDataToRestorationRegistry(  # noqa: N802, D102
        self,
        damaged_node_name,
        history,
        function_name,
    ):
        data = self._restoration_table
        selected_data = data[
            (
                data[['node_name', 'function']] == [damaged_node_name, function_name]
            ).all(1)
        ]
        if len(selected_data) > 0:
            raise ValueError(
                'There are data in restroation registry. Damaged node name: '
                + damaged_node_name
                + '  '
                + '  '
                + function_name
            )

        self._record_registry.append(history)
        latest_index = len(self._record_registry) - 1

        temp = pd.Series(
            data=[damaged_node_name, function_name, latest_index],
            index=['node_name', 'function', 'record_index'],
        )
        self._restoration_table = self._restoration_table.append(
            temp, ignore_index=True
        )

    def addNodalDamage(self, nodal_damage, new_pipe_name_list):  # noqa: N802, D102
        if self.settings['Virtual_node'] == True:  # noqa: E712
            for ind, val in nodal_damage.items():
                val = int(val)  # noqa: PLW2901
                virtual_node_name_list = []
                for i in range(val):
                    new_virtual_node_name = ind + '_vir_' + str(i)
                    self._node_damage_table.loc[
                        new_virtual_node_name, 'Number_of_damages'
                    ] = 1
                    self._node_damage_table.loc[
                        new_virtual_node_name, 'virtual_of'
                    ] = ind
                    self._node_damage_table.loc[
                        new_virtual_node_name, 'Orginal_element'
                    ] = ind
                    self.virtual_node_data[new_virtual_node_name] = {
                        'is_damaged': True
                    }
                    self.virtual_node_data[new_virtual_node_name][
                        'new_pipe_name'
                    ] = new_pipe_name_list[ind]
                    virtual_node_name_list.append(new_virtual_node_name)
                self._nodal_data[ind] = new_pipe_name_list[ind]
        else:
            for ind, val in nodal_damage.items():
                self._node_damage_table.loc[ind, 'Number_of_damages'] = val
                self._node_damage_table.loc[
                    new_virtual_node_name, 'Orginal_element'
                ] = ind
                self._nodal_data[ind] = {
                    'real_node_name': ind,
                    'number_of_damages': val,
                }

    def isVirtualNodeDamaged(self, virtual_node_name):  # noqa: N802, D102
        return self.virtual_node_data[virtual_node_name]['is_damaged']

    def setVirtualNodeRepaired(self, virtual_node_name):  # noqa: N802, D102
        self.virtual_node_data[virtual_node_name]['is_damaged'] = False

    def addNodalDemandChange(self, node_name, demand1, demand2):  # noqa: N802, D102
        # if self.settings['Virtual_node'] == False:
        if type(node_name) == str:  # noqa: E721
            if node_name not in self._node_damage_table.index:
                raise ValueError(repr(node_name) + ' is not in the node table')
        self._node_damage_table.loc[node_name, 'Demand1'] = demand1
        self._node_damage_table.loc[node_name, 'Demand2'] = demand2
        # else:
        # node_name_vir = get_node_name(node_name, self._node_damage_table)
        # self._node_damage_table.loc[node_name_vir, 'Demand1'] = demand1
        # self._node_damage_table.loc[node_name_vir, 'Demand2'] = demand2

    def addPipeDamageToRegistry(self, node_name, data):  # noqa: N802
        """Adds damage to pipe registry

        Parameters
        ----------
        node_name : string
            Damaged node Name.
        data : Dict
            Data about Damage.

        Returns
        -------
        None.

        """  # noqa: D400, D401, DOC202
        # self._pipe_node_damage_status[name] = data

        leaking_pipe_with_pipeA_orginal_pipe = self._pipe_leak_history[  # noqa: N806
            self._pipe_leak_history.loc[:, 'Pipe_A'] == data['orginal_pipe']
        ]
        breaking_pipe_with_pipeA_orginal_pipe = self._pipe_break_history[  # noqa: N806
            self._pipe_break_history.loc[:, 'Pipe_A'] == data['orginal_pipe']
        ]

        i_leak_not_zero_length = len(leaking_pipe_with_pipeA_orginal_pipe) > 0
        i_break_not_zero_length = len(breaking_pipe_with_pipeA_orginal_pipe) > 0

        if i_leak_not_zero_length and i_break_not_zero_length:
            raise ValueError(  # noqa: DOC501, TRY003
                'There are more than 1 damage with original pipe name in pipe A. it does not make sense'  # noqa: EM101
            )
        if i_leak_not_zero_length:
            temp_node_name = leaking_pipe_with_pipeA_orginal_pipe.index[0]
            self._pipe_leak_history.loc[temp_node_name, 'Pipe_A'] = data['pipe_B']
        elif i_break_not_zero_length:
            temp_node_name = breaking_pipe_with_pipeA_orginal_pipe.index[0]
            self._pipe_break_history.loc[temp_node_name, 'Pipe_A'] = data['pipe_B']

        if data['damage_type'] == 'leak':
            self._pipe_damage_table.loc[node_name, 'damage_type'] = data[
                'damage_type'
            ]
            self._pipe_damage_table.loc[node_name, 'damage_sub_type'] = data[
                'damage_subtype'
            ]
            self._pipe_damage_table.loc[node_name, 'Orginal_element'] = data[
                'orginal_pipe'
            ]
            self._pipe_damage_table.loc[node_name, 'attached_element'] = data[
                'pipe_A'
            ]
            self._pipe_damage_table.loc[node_name, 'number'] = data['number']

            self._pipe_leak_history.loc[node_name, 'Pipe_A'] = data['pipe_A']
            self._pipe_leak_history.loc[node_name, 'Pipe_B'] = data['pipe_B']
            self._pipe_leak_history.loc[node_name, 'Orginal_pipe'] = data[
                'orginal_pipe'
            ]
            self._pipe_leak_history.loc[node_name, 'Node_name'] = node_name

        elif data['damage_type'] == 'break':
            self._pipe_damage_table.loc[node_name, 'damage_type'] = data[
                'damage_type'
            ]
            self._pipe_damage_table.loc[node_name, 'Orginal_element'] = data[
                'orginal_pipe'
            ]
            self._pipe_damage_table.loc[node_name, 'attached_element'] = data[
                'pipe_A'
            ]
            self._pipe_damage_table.loc[node_name, 'number'] = data['number']

            self._pipe_break_history.loc[node_name, 'Pipe_A'] = data['pipe_A']
            self._pipe_break_history.loc[node_name, 'Pipe_B'] = data['pipe_B']
            self._pipe_break_history.loc[node_name, 'Orginal_pipe'] = data[
                'orginal_pipe'
            ]
            self._pipe_break_history.loc[node_name, 'Node_A'] = data['node_A']
            self._pipe_break_history.loc[node_name, 'Node_B'] = data['node_B']

        else:
            raise ValueError('Undefined damage type')  # noqa: DOC501, EM101, RUF100, TRY003

    def addGeneralNodeDamageToRegistry(self, node_name, data=None):  # noqa: ARG002, N802, D102
        self._gnode_damage_table.loc[node_name, 'damage_type'] = None

    def addTankDamageToRegistry(self, node_name, data=None):  # noqa: ARG002, N802, D102
        self._tank_damage_table.loc[node_name, 'damage_type'] = None

    def addPumpDamageToRegistry(self, pump_name, data):  # noqa: N802, D102
        node_name = data.start_node.name
        self._pump_damage_table.loc[node_name, 'damage_type'] = None
        self._pump_damage_table.loc[node_name, 'element_name'] = pump_name
        self._pump_damage_table.loc[node_name, 'start_node'] = data.start_node.name
        self._pump_damage_table.loc[node_name, 'end_node'] = data.end_node.name

    def addReservoirDamageToRegistry(self, node_name, data=None):  # noqa: ARG002, N802, D102
        self._reservoir_damage_table.loc[node_name, 'damage_type'] = None

    # def assignAgentToDamage(self, element, node_name, choosed_agent_name):

    def getListAllElementOrginalName(self, element_type):  # noqa: N802, D102
        original_element_list = None
        if element_type == 'PIPE':
            original_element_list = self._pipe_damage_table['Orginal_element']

        elif element_type == 'PUMP':
            original_element_list = self._pump_damage_table['element_name']

        elif (
            element_type == 'DISTNODE'  # noqa: PLR1714
            or element_type == 'GNODE'
            or element_type == 'TANK'
            or element_type == 'RESERVOIR'
        ):
            temp = self.getDamageData(element_type, iCopy=False)
            if 'virtual_of' in temp.columns:
                original_element_list = pd.Series(
                    temp['virtual_of'], index=temp.index
                )
            else:
                original_element_list = pd.Series(temp.index, index=temp.index)

        else:
            raise ValueError('Unkown recognized element type: ' + repr(element_type))

        return original_element_list

    def getDamagedLocationListByOriginalElementList(  # noqa: N802, D102
        self,
        element_type,
        orginal_element_list,
        iCheck=False,  # noqa: FBT002, N803
    ):
        res = pd.Series()

        if element_type == 'PIPE':
            original_element_list = self._pipe_damage_table['Orginal_element']

        elif element_type == 'PUMP':
            original_element_list = self._pump_damage_table['element_name']

        elif (
            element_type == 'DISTNODE'  # noqa: PLR1714
            or element_type == 'GNODE'
            or element_type == 'TANK'
            or element_type == 'RESERVOIR'
        ):
            temp = self.getDamageData(element_type)
            original_element_list = pd.Series(temp.index, index=temp.index)

        else:
            raise ValueError('Unkown recognized element type: ' + repr(element_type))

        for element_name, group_tag in orginal_element_list.iteritems():  # noqa: B007
            temp = original_element_list[original_element_list == element_name]

            # if len(temp)!=1:
            if len(temp) != 0:
                res = res.append(temp)
            # elif len(temp)>1:
            # raise ValueError('Something wrong here')
            elif iCheck:
                raise ValueError(
                    'The element: '
                    + repr(element_name)
                    + ' does not exist in element type: '
                    + repr(element_type)
                )

        return res

    def getDamagedLocationListByOriginalElementList_2(  # noqa: N802, D102
        self,
        element_type,
        orginal_element_list,
        iCheck=False,  # noqa: FBT002, N803
    ):
        if element_type == 'PIPE':
            all_original_element_list = self._pipe_damage_table['Orginal_element']

        elif element_type == 'PUMP':
            all_original_element_list = self._pump_damage_table['element_name']

        elif (
            element_type == 'DISTNODE'  # noqa: PLR1714
            or element_type == 'GNODE'
            or element_type == 'TANK'
            or element_type == 'RESERVOIR'
        ):
            temp = self.getDamageData(element_type, iCopy=False)
            if 'virtual_of' in temp:
                all_original_element_list = temp['virtual_of']
            else:
                all_original_element_list = pd.Series(temp.index, index=temp.index)

        else:
            raise ValueError('Unkown recognized element type: ' + repr(element_type))
        temp_bool = all_original_element_list.isin(orginal_element_list.index)
        res = all_original_element_list[temp_bool]
        if iCheck == True:  # noqa: E712
            if len(res.index) < len(orginal_element_list):
                not_available_list = set(orginal_element_list) - set(res.index)
                raise ValueError(
                    'The element: '
                    + repr(not_available_list)
                    + ' does not exist in element type: '
                    + repr(element_type)
                )

        return res

    def getOriginalPipenodes(self, orginal_pipe_name):  # noqa: N802, D102
        return self.original_pipe_data[orginal_pipe_name]

    def getLeakData(self, leaking_node_name):  # noqa: N802, D102
        pipe_A = self._pipe_leak_history.loc[leaking_node_name, 'Pipe_A']  # noqa: N806
        pipe_B = self._pipe_leak_history.loc[leaking_node_name, 'Pipe_B']  # noqa: N806
        orginal_pipe = self._pipe_leak_history.loc[leaking_node_name, 'Orginal_pipe']

        return pipe_A, pipe_B, orginal_pipe

    def getCertainLeakData(self, damage_node_name, wn):  # noqa: C901, N802, D102
        pipe_name_list = []

        result_pipe_A = None  # noqa: N806
        result_pipe_B = None  # noqa: N806

        orginal_pipe = self._pipe_leak_history.loc[damage_node_name, 'Orginal_pipe']
        refined_data = self._pipe_leak_history[
            self._pipe_leak_history['Orginal_pipe'] == orginal_pipe
        ]

        for damage_point_name, data in refined_data.iterrows():  # noqa: B007
            pipe_A = data['Pipe_A']  # noqa: N806
            pipe_B = data['Pipe_B']  # noqa: N806

            if pipe_A not in pipe_name_list:
                pipe_name_list.append(pipe_A)
            if pipe_B not in pipe_name_list:
                pipe_name_list.append(pipe_B)

        # orginal_pipe = self._pipe_break_history.loc[damage_node_name, 'Orginal_pipe']
        refined_data = self._pipe_break_history[
            self._pipe_break_history['Orginal_pipe'] == orginal_pipe
        ]

        for damage_point_name, data in refined_data.iterrows():  # noqa: B007
            pipe_A = data['Pipe_A']  # noqa: N806
            pipe_B = data['Pipe_B']  # noqa: N806

            if pipe_A not in pipe_name_list:
                pipe_name_list.append(pipe_A)
            if pipe_B not in pipe_name_list:
                pipe_name_list.append(pipe_B)

        for pipe_name in pipe_name_list:
            try:
                pipe = wn.get_link(pipe_name)
            except:  # noqa: S112, E722
                continue

            if damage_node_name == pipe.start_node_name:
                result_pipe_B = pipe_name  # noqa: N806
            elif damage_node_name == pipe.end_node_name:
                result_pipe_A = pipe_name  # noqa: N806

            if result_pipe_A != None and result_pipe_B != None:  # noqa: E711
                return result_pipe_A, result_pipe_B
        raise RuntimeError(
            'There must be a pair of pipes for ' + repr(damage_node_name)
        )

    def getBreakData(self, breaking_node_name):  # noqa: N802, D102
        pipe_A = self._pipe_break_history.loc[breaking_node_name, 'Pipe_A']  # noqa: N806
        pipe_B = self._pipe_break_history.loc[breaking_node_name, 'Pipe_B']  # noqa: N806
        orginal_pipe = self._pipe_break_history.loc[
            breaking_node_name, 'Orginal_pipe'
        ]
        node_A = self._pipe_break_history.loc[breaking_node_name, 'Node_A']  # noqa: N806
        node_B = self._pipe_break_history.loc[breaking_node_name, 'Node_B']  # noqa: N806

        return pipe_A, pipe_B, orginal_pipe, node_A, node_B

    def getCertainBreakData(self, damage_node_name, wn):  # noqa: C901, N802, D102
        pipe_name_list = []

        result_pipe_A = None  # noqa: N806
        result_pipe_B = None  # noqa: N806

        node_A = self._pipe_break_history.loc[damage_node_name, 'Node_A']  # noqa: N806
        node_B = self._pipe_break_history.loc[damage_node_name, 'Node_B']  # noqa: N806

        orginal_pipe = self._pipe_break_history.loc[damage_node_name, 'Orginal_pipe']

        refined_data = self._pipe_leak_history[
            self._pipe_leak_history['Orginal_pipe'] == orginal_pipe
        ]

        for damage_point_name, data in refined_data.iterrows():  # noqa: B007
            pipe_A = data['Pipe_A']  # noqa: N806
            pipe_B = data['Pipe_B']  # noqa: N806

            if pipe_A not in pipe_name_list:
                pipe_name_list.append(pipe_A)
            if pipe_B not in pipe_name_list:
                pipe_name_list.append(pipe_B)

        # orginal_pipe = self._pipe_break_history.loc[damage_node_name, 'Orginal_pipe']
        refined_data = self._pipe_break_history[
            self._pipe_break_history['Orginal_pipe'] == orginal_pipe
        ]

        for damage_point_name, data in refined_data.iterrows():  # noqa: B007
            pipe_A = data['Pipe_A']  # noqa: N806
            pipe_B = data['Pipe_B']  # noqa: N806

            if pipe_A not in pipe_name_list:
                pipe_name_list.append(pipe_A)
            if pipe_B not in pipe_name_list:
                pipe_name_list.append(pipe_B)

        for pipe_name in pipe_name_list:
            try:
                pipe = wn.get_link(pipe_name)
            except:  # noqa: S112, E722
                continue

            if node_B == pipe.start_node_name:
                result_pipe_B = pipe_name  # noqa: N806
            elif node_A == pipe.end_node_name:
                result_pipe_A = pipe_name  # noqa: N806

            if result_pipe_A != None and result_pipe_B != None:  # noqa: E711
                return result_pipe_A, result_pipe_B, node_A, node_B
        raise RuntimeError(
            'There must be a pair of pipes for ' + repr(damage_node_name)
        )

    def getPipeDamageAttribute(self, attribute_name, damage_node_name=None):  # noqa: N802, D102
        if attribute_name not in self._pipe_damage_table.columns:
            raise ValueError('Attribute not in damage table: ' + str(attribute_name))

        if damage_node_name == None:  # noqa: E711
            return self._pipe_damage_table[attribute_name]
        else:  # noqa: RET505
            return self._pipe_damage_table.loc[damage_node_name, attribute_name]

    def getDamageData(self, element_type, iCopy=True):  # noqa: FBT002, C901, N802, N803, D102
        if element_type.upper() == 'PIPE':
            if iCopy:
                res = self._pipe_damage_table.copy()
            else:
                res = self._pipe_damage_table
        elif element_type.upper() == 'DISTNODE':
            if iCopy:
                res = self._node_damage_table.copy()
            else:
                res = self._node_damage_table

        elif element_type.upper() == 'GNODE':
            if iCopy:
                res = self._gnode_damage_table.copy()
            else:
                res = self._gnode_damage_table

        elif element_type.upper() == 'TANK':
            if iCopy:
                res = self._tank_damage_table.copy()
            else:
                res = self._tank_damage_table

        elif element_type.upper() == 'PUMP':
            if iCopy:
                res = self._pump_damage_table.copy()
            else:
                res = self._pump_damage_table

        elif element_type.upper() == 'RESERVOIR':
            if iCopy:
                res = self._reservoir_damage_table.copy()
            else:
                res = self._reservoir_damage_table

        else:
            raise ValueError('Unknown element type: ' + element_type)
        return res

    def getOrginalElement(self, damaged_node_name, element_type):  # noqa: N802, D102
        element_damage_data = self.getDamageData(element_type, iCopy=False)
        return element_damage_data.loc[damaged_node_name, 'Orginal_element']

    def getPipeData(self, attr, name=None):  # noqa: N802, D102
        if name != None:  # noqa: E711
            return self._pipe_data[attr].loc[name]
        else:  # noqa: RET505
            return self._pipe_data[attr]

    def setDamageData(self, element, col, value):  # noqa: N802, D102
        if element.upper() == 'PIPE':
            if col not in self._pipe_damage_table.columns:
                raise ValueError('Columns is not in damage table: ' + col)
            self._pipe_damage_table[col] = value
        elif element.upper() == 'DISTNODE':
            if col not in self._node_damage_table.columns:
                raise ValueError('Columns is not in damage table: ' + col)
            self._node_damage_table[col] = value
        elif element.upper() == 'GNODE':
            self._gnode_damage_table[col] = value
        elif element.upper() == 'TANK':
            self._tank_damage_table[col] = value
        elif element.upper() == 'PUMP':
            self._pump_damage_table[col] = value
        elif element.upper() == 'RESERVOIR':
            self._reservoir_damage_table[col] = value
        else:
            raise ValueError('Element is not defined: ' + element)

    def setDamageDataByRowAndColumn(self, element, index, col, value, iCheck=False):  # noqa: FBT002, N802, N803, D102
        # if element.upper() == 'PIPE':
        damage_table = self.getDamageData(element, iCopy=False)
        if col not in damage_table.columns:
            raise ValueError('Columns is not in damage table: ' + col)
        if type(index) == list or (  # noqa: E721
            (index in damage_table.index and col in damage_table.columns)
            or iCheck == True  # noqa: E712
        ):
            damage_table.loc[index, col] = value
        else:
            raise ValueError(index)

    def setDamageDataByList(self, element, index_list, col, value, iCheck=False):  # noqa: FBT002, C901, N802, N803, D102
        if type(index_list) != list:  # noqa: E721
            raise ValueError('index_list is not data type list')  # noqa: EM101, TRY003

        if element.upper() == 'PIPE':
            if col not in self._pipe_damage_table.columns:
                raise ValueError('Columns is not in damage table: ' + col)

            for damage_node_name in index_list:
                if (
                    damage_node_name in self._pipe_damage_table.index
                    or iCheck == True  # noqa: E712
                ):
                    self._pipe_damage_table.loc[damage_node_name, col] = value
                else:
                    raise ValueError(damage_node_name)

        elif element.upper() == 'DISTNODE':
            if col not in self._node_damage_table.columns:
                raise ValueError('Columns is not in damage table: ' + col)

            for damage_node_name in index_list:
                if (
                    damage_node_name in self._node_damage_table.index
                    or iCheck == True  # noqa: E712
                ):
                    self._node_damage_table.loc[damage_node_name, col] = value
                else:
                    raise ValueError(damage_node_name)

        elif element.upper() == 'GNODE':
            if col not in self._gnode_damage_table.columns:
                raise ValueError('Columns is not in damage table: ' + col)

            for gnode_name in index_list:
                if gnode_name in self._gnode_damage_table.index or iCheck == True:  # noqa: E712
                    self._gnode_damage_table.loc[gnode_name, col] = value
                else:
                    raise ValueError(gnode_name)

        elif element.upper() == 'TANK':
            if col not in self._tank_damage_table.columns:
                raise ValueError('Columns is not in damage table: ' + col)

            for _tank_damage_table in index_list:
                if (
                    _tank_damage_table in self._tank_damage_table.index
                    or iCheck == True  # noqa: E712
                ):
                    self._tank_damage_table.loc[_tank_damage_table, col] = value
                else:
                    raise ValueError(_tank_damage_table)

        elif element.upper() == 'PUMP':
            if col not in self._pump_damage_table.columns:
                raise ValueError('Columns is not in damage table: ' + col)

            for _pump_damage_table in index_list:
                if (
                    _pump_damage_table in self._pump_damage_table.index
                    or iCheck == True  # noqa: E712
                ):
                    self._pump_damage_table.loc[_pump_damage_table, col] = value
                else:
                    raise ValueError(_pump_damage_table)

        elif element.upper() == 'RESERVOIR':
            if col not in self._reservoir_damage_table.columns:
                raise ValueError('Columns is not in damage table: ' + col)

            for _reservoir_damage_table in index_list:
                if (
                    _reservoir_damage_table in self._reservoir_damage_table.index
                    or iCheck == True  # noqa: E712
                ):
                    self._reservoir_damage_table.loc[
                        _reservoir_damage_table, col
                    ] = value
                else:
                    raise ValueError(_reservoir_damage_table)
        else:
            raise ValueError('Element is not defined: ' + element)

    def updatePipeDamageTableTimeSeries(self, time):  # noqa: N802, D102
        if time in self._pipe_damage_table_time_series:
            raise ValueError('Time exist in pipe damage table time history')  # noqa: EM101, TRY003

        self._pipe_damage_table_time_series[time] = self._pipe_damage_table.copy()

    def updateNodeDamageTableTimeSeries(self, time):  # noqa: N802, D102
        if time in self._node_damage_table_time_series:
            raise ValueError('Time exist in node damage table time history')  # noqa: EM101, TRY003

        self._node_damage_table_time_series[time] = self._node_damage_table.copy()

    def updateTankTimeSeries(self, wn, time):  # noqa: N802, D102
        if time in self._tank_level_time_series:
            raise ValueError('Time exist in tank damage table time history')  # noqa: EM101, TRY003

        tank_name_list = wn.tank_name_list
        tank_level_res = pd.Series(index=tank_name_list)

        for tank_name in wn.tank_name_list:
            node = wn.get_node(tank_name)
            net_water_level = node.level - node.min_level
            if net_water_level < 0.001:  # noqa: PLR2004
                raise ValueError(
                    'Net Water Level in tank cannot be less than zero:'
                    + repr(tank_name)
                    + '  '
                    + repr(net_water_level)
                )
            tank_level_res.loc[tank_name] = net_water_level

        self._tank_level_time_series[time] = tank_level_res

    def updateRestorationIncomeWaterTimeSeries(self, wn, time):  # noqa: ARG002, N802, D102
        if time in self._restoration_reservoir_name_time_series:
            raise ValueError(  # noqa: TRY003
                'Time exist in restoration reservoir damage table time history'  # noqa: EM101
            )
        res = []
        for list_of_restoration in self._record_registry:
            for key, value in list_of_restoration.items():
                if key == 'ADDED_RESERVOIR':
                    res.append(value)

        self._restoration_reservoir_name_time_series[time] = res

    def updateElementDamageTable(self, element, attr, index, value, icheck=False):  # noqa: FBT002, C901, N802, D102
        if element == 'PIPE':
            if icheck == True:  # noqa: E712
                if self._pipe_damage_table[attr].loc[index] == value:
                    raise ValueError('the value is already set')  # noqa: EM101, TRY003

            self._pipe_damage_table.loc[index, attr] = value

        elif element == 'DISTNODE':
            if icheck == True:  # noqa: E712
                if self._node_damage_table[attr].loc[index] == value:
                    raise ValueError(
                        'the value is already set in element: '
                        + element
                        + ', attr: '
                        + attr
                        + ', index: '
                        + index
                        + ', value: '
                        + value
                    )

            self._node_damage_table.loc[index, attr] = value

        elif element == 'GNODE':
            if icheck == True:  # noqa: E712
                if self._gnode_damage_table[attr].loc[index] == value:
                    raise ValueError(
                        'the value is already set in element: '
                        + element
                        + ', attr: '
                        + attr
                        + ', index: '
                        + index
                        + ', value: '
                        + value
                    )

            self._gnode_damage_table.loc[index, attr] = value

        elif element == 'TANK':
            if icheck == True:  # noqa: E712
                if self._tank_damage_table[attr].loc[index] == value:
                    raise ValueError(
                        'the value is already set in element: '
                        + element
                        + ', attr: '
                        + attr
                        + ', index: '
                        + index
                        + ', value: '
                        + value
                    )

            self._tank_damage_table.loc[index, attr] = value

        elif element == 'PUMP':
            if icheck == True:  # noqa: E712
                if self._pump_damage_table[attr].loc[index] == value:
                    raise ValueError(
                        'the value is already set in element: '
                        + element
                        + ', attr: '
                        + attr
                        + ', index: '
                        + index
                        + ', value: '
                        + value
                    )

            self._pump_damage_table.loc[index, attr] = value

        elif element == 'RESERVOIR':
            if icheck == True:  # noqa: E712
                if self._reservoir_damage_table[attr].loc[index] == value:
                    raise ValueError(
                        'the value is already set in element: '
                        + element
                        + ', attr: '
                        + attr
                        + ', index: '
                        + index
                        + ', value: '
                        + value
                    )

            self._reservoir_damage_table.loc[index, attr] = value

        else:
            raise ValueError('Unknown element: ' + element)

    def addAttrToElementDamageTable(self, element, attr, def_data):  # noqa: N802, D102
        if element == 'PIPE':
            self.addAttrToPipeDamageTable(attr, def_data)
        elif element == 'DISTNODE':
            self.addAttrToDistNodeDamageTable(attr, def_data)
        elif element == 'GNODE':
            self.addAttrToGeneralNodeDamageTable(attr, def_data)
        elif element == 'TANK':
            self.addAttrToTankDamageTable(attr, def_data)
        elif element == 'PUMP':
            self.addAttrToPumpDamageTable(attr, def_data)
        elif element == 'RESERVOIR':
            self.addAttrToReservoirDamageTable(attr, def_data)

        else:
            raise ValueError('Undefined element: ' + element)

    def addAttrToPipeDamageTable(self, attr, def_data):  # noqa: N802, D102
        if attr in self._pipe_damage_table.columns:
            raise ValueError('attribute already in the damage table')  # noqa: EM101, TRY003

        if def_data == None:  # noqa: E711
            self._pipe_damage_table[attr] = np.nan
        else:
            self._pipe_damage_table[attr] = def_data

    def addAttrToDistNodeDamageTable(self, attr, def_data):  # noqa: N802, D102
        if attr in self._node_damage_table.columns:
            raise ValueError('attribute already in the damage table')  # noqa: EM101, TRY003

        if def_data == None:  # noqa: E711
            self._node_damage_table[attr] = np.nan
        else:
            self._node_damage_table[attr] = def_data

    def addAttrToGeneralNodeDamageTable(self, attr, def_data):  # noqa: N802, D102
        if attr in self._gnode_damage_table.columns:
            raise ValueError('attribute already in the damage table')  # noqa: EM101, TRY003

        if def_data == None:  # noqa: E711
            self._gnode_damage_table[attr] = np.nan
        else:
            self._gnode_damage_table[attr] = def_data

    def addAttrToTankDamageTable(self, attr, def_data):  # noqa: N802, D102
        if attr in self._tank_damage_table.columns:
            raise ValueError('attribute already in the damage table')  # noqa: EM101, TRY003

        if def_data == None:  # noqa: E711
            self._tank_damage_table[attr] = np.nan
        else:
            self._tank_damage_table[attr] = def_data

    def addAttrToPumpDamageTable(self, attr, def_data):  # noqa: N802, D102
        if attr in self._pump_damage_table.columns:
            raise ValueError('attribute already in the damage table')  # noqa: EM101, TRY003

        if def_data == None:  # noqa: E711
            self._pump_damage_table[attr] = np.nan
        else:
            self._pump_damage_table[attr] = def_data

    def addAttrToReservoirDamageTable(self, attr, def_data):  # noqa: N802, D102
        if attr in self._reservoir_damage_table.columns:
            raise ValueError('attribute already in the damage table')  # noqa: EM101, TRY003

        if def_data == None:  # noqa: E711
            self._reservoir_damage_table[attr] = np.nan
        else:
            self._reservoir_damage_table[attr] = def_data

    def iOccupied(self, node_name):  # noqa: N802
        """Checks if the node is occuoied

        Parameters
        ----------
        node_name : string
            Node ID.

        Returns
        -------
        bool
            result.

        """  # noqa: D400, D401
        return node_name in self._occupancy.index

    def _getDamagedPipesRegistry(self):  # noqa: N802
        """Gets the whole damage registry. Not safe to be used outside the class.

        Returns
        -------
        Pandas.Series
            damage locations by node name.

        """  # noqa: D401
        return self._pipe_node_damage_status

    def getNumberofDamagedNodes(self):  # noqa: N802
        """Gets numbers of Damaged locations. Counts two for broken pipes

        Returns
        -------
        Int
            Number of damaged locations by node name.

        """  # noqa: D400, D401
        return len(self._pipe_node_damage_status)

    def occupyNode(self, node_name, occupier_name):  # noqa: N802
        """Put adds node and its occupier in occupency list

        Parameters
        ----------
        node_name : string
            Node ID.
        occupier_name : string
            occupier's name'.

        Raises
        ------
        ValueError
            If occupier is reused, meaning that occupier is busy somewhere else.

        Returns
        -------
        None.

        """  # noqa: D400, DOC202
        if occupier_name in self._occupancy:
            # if not iNodeCoupled(node_name):
            raise ValueError(  # noqa: TRY003
                'Occupier name already in the list. Forget to remove another occupancy or double adding?'  # noqa: EM101
            )
        self._occupancy = self._occupancy.append(
            pd.Series(data=occupier_name, index=[node_name])
        )

    def removeOccupancy(self, occupier_name):  # noqa: N802
        """Removes occupency in the node by occupier's name.

        Parameters
        ----------
        occupier_name : string
            Occupier's name that is evacuating the node(s).

        Raises
        ------
        ValueError
            If the occupier's is not in occupency's name.

        Returns
        -------
        None.

        """  # noqa: D401, DOC202
        temp = self._occupancy[self._occupancy == occupier_name]

        if len(temp) == 0:
            raise ValueError('there is no node occupied with this occupier name')  # noqa: EM101, TRY003

        ind = temp.index.tolist()
        self._occupancy = self._occupancy.drop(ind)

    def whoOccupiesIn(self, node_name):  # noqa: N802
        """Gets name of the occupier

        Parameters
        ----------
        node_name : string
            node ID.

        Returns
        -------
        string
            Occupier's name.

        """  # noqa: D400, D401
        return self._occupancy[node_name]

    def whereIsOccupiedByName(self, occupier_name):  # noqa: N802
        """Gets node(s) occupied by occupier

        Parameters
        ----------
        occupier_name : string
            occupier's name.

        Raises
        ------
        ValueError
            if occupier is not occupying any node.

        Returns
        -------
        str or series
            node(s) ID.

        """  # noqa: D400, D401, DOC202
        temp = self._occupancy[self._occupancy == occupier_name]
        if len(temp) == 0:
            raise ValueError('there is no occupancy with this name')  # noqa: EM101, TRY003

    def getListofFreeRepairAgents(self):  # noqa: N802
        """MAYBE NOT NEEDED Gets a list of free agents. Not needed anymore.

        Returns
        -------
        Free_RepairAgents : TYPE
            DESCRIPTION.

        """
        working_RepairAgents = set(self._occupancy.tolist())  # noqa: N806
        RepairAgentsNameList = self._pipe_RepairAgentNameRegistry  # noqa: N806
        Free_RepairAgents = [  # noqa: N806
            name for name in RepairAgentsNameList if name not in working_RepairAgents
        ]
        return Free_RepairAgents  # noqa: RET504

    def coupleTwoBreakNodes(self, break_point_1_name, break_point_2_name):  # noqa: N802
        """Couples two nodes in registry for the time which we have a break.
        PLEASE NOTE THAT THE FIRST NODE MUST BE THE ONE CONNECTED TO THE
        MAIN(ORIGINAL) PIPE THAT IS BROKEN NOW.

        Parameters
        ----------
        break_point_1_name : STR
            First broken node(connected to the original node)
        break_point_2_name : STR
            Second node.

        Returns
        -------
        None.

        """  # noqa: D205, DOC202
        self._pipe_break_node_coupling[break_point_1_name] = break_point_2_name
        self._pipe_break_node_coupling[break_point_2_name] = break_point_1_name
        self._break_point_attached_to_mainPipe.append(break_point_1_name)

    def getCoupledBreakNode(self, break_point_name):  # noqa: N802
        """Gets the coupled node given the first coupled node, and checks if the
        given coupled node is connected to the main pipe.

        Parameters
        ----------
        break_point_name : str
            Node name

        Returns
        -------
        out1 : str
            the other coupled node name
        is_breakPoint_1_attacjedToMainPipe : bool
            If the given (first node) is the one connected to the main(original)
            pipe

        """  # noqa: D205, D401
        out1 = self._pipe_break_node_coupling[break_point_name]
        is_breakPoint_1_attacjedToMainPipe = (  # noqa: N806
            break_point_name in self._break_point_attached_to_mainPipe
        )
        return out1, is_breakPoint_1_attacjedToMainPipe

    def iNodeCoupled(self, node_name):  # noqa: N802, D102
        return node_name in self._pipe_break_node_coupling

    def iDamagedPipeReminded(self):  # noqa: N802, D102
        damaged_nodes = self._pipe_node_damage_status.index
        if len(damaged_nodes) == 0:
            return False
        is_reminded = False
        for node_name in iter(damaged_nodes):
            if node_name not in self._occupancy.index:
                is_reminded = True
                return is_reminded  # noqa: RET504
        return is_reminded

    def getOtherCoupledBreakPoint(self, node_name):  # noqa: N802, D102
        return self._pipe_break_node_coupling[node_name]

    def removeCoupledBreakNodes(self, break_point_name):  # noqa: N802
        """Removes the coupled

        Parameters
        ----------
        break_point_name : str
            Name of either one of a couple break points.

        Returns
        -------
        first : str
            Name of first node(coonected to the mnain pipe).
        second : str
            Name of second node(connected to the pipe created after break)

        """  # noqa: D400, D401
        other_coupled_break_point = self._pipe_break_node_coupling.pop(
            break_point_name
        )
        self._pipe_break_node_coupling.pop(other_coupled_break_point)
        # self._break_node_coupling.pop(break_point_name)

        i_in_list = break_point_name in self._break_point_attached_to_mainPipe
        if i_in_list:
            self._break_point_attached_to_mainPipe.remove(break_point_name)
            first = break_point_name
            second = other_coupled_break_point
        else:
            first = other_coupled_break_point
            second = break_point_name
        return first, second

    def recordPipeDamageTable(self, stop_time):  # noqa: N802, D102
        if self.settings['result_details'] == 'minimal':
            return None
        if stop_time in self._pipe_damage_table_history:
            return ValueError('Time exists in pipe damage hostry: ' + str(stop_time))
        self._pipe_damage_table_history['stop_time'] = (  # noqa: RET503
            self._pipe_damage_table_history
        )

    def getMostLeakAtCheck(self, real_node_name_list, element_type):  # noqa: N802, D102
        if element_type == 'DISTNODE':
            total_demand = self._node_damage_table.loc[
                real_node_name_list, 'Demand2'
            ]
            total_demand.loc[total_demand[total_demand.isna()].index] = 0
            return total_demand
        elif element_type == 'PIPE':  # noqa: RET505
            leak = self._pipe_damage_table.loc[real_node_name_list, 'LeakAtCheck']
            leak.loc[leak[leak.isna()].index] = 0
            return leak
        else:
            return None
