"""Created on Fri Dec 25 04:00:43 2020

@author: snaeimi
"""

import copy
import logging
import random

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_node_name(node_name, table):
    if 'virtual_of' in table.columns:
        real_node_name = table.loc[node_name, 'virtual_of']
        if (
            real_node_name == None or real_node_name == np.nan
        ):  # SINA: probably NP.NAN does not work here. Correct it.
            real_node_name = node_name
        return real_node_name
    else:
        return node_name


class Coordination:
    def __init__(self, X=None, Y=None, system=None):
        self.x = X
        self.y = Y
        self.system = system

    def set_coord(self, X, Y, system=None):
        self.x = X
        self.y = Y

    def get_coord(self):
        return (self.x, self.y)

    def set_system(self, system):
        self.system = system


class Location:
    def __init__(self, name, x, y):
        self.name = name
        self.coord = Coordination(x, y)


# =============================================================================
# class restoration_base():
#     def __init__(self):
#         self.coord = coordination()
#         self.ID = None
#         self.Object_typ = None
#
# =============================================================================


class AgentData:
    def __init__(
        self,
        agent_name,
        agent_type,
        cur_x,
        cur_y,
        shift_name,
        base_name,
        base_x,
        base_y,
        shift_obj,
        agent_speed,
    ):
        if type(agent_type) != str:
            raise ValueError('agent type must be string')
        # if type(definition) != pd.Series:
        # raise ValueError('definiton must be a Pandas series')

        if type(cur_x) != float:
            raise ValueError('cur_x must be float')
        if type(cur_y) != float:
            raise ValueError('cur_y must be float')
        if type(base_x) != float:
            raise ValueError('base_x must be float')
        if type(base_y) != float:
            raise ValueError('base_y must be float')

        self.name = agent_name
        self.agent_type = agent_type
        self.current_location = Location('current', cur_x, cur_y)
        self.base_location = Location(base_name, base_x, base_y)
        self.shift = AgentShift(self.name, shift_name)
        self._shifting = shift_obj
        self._avg_speed = agent_speed  # 20*3/3.6
        self.isWorking = False
        self.cur_job_location = None
        self.cur_job_action = None
        self.cur_job_entity = None
        self._time_of_arival = None
        self._time_of_job_done = None
        self.cur_job_ongoing = None
        self.cur_job_effect_definition_name = None
        self.cur_job_method_name = None

    def isOnShift(self, time):
        """Checks if a time is on an agent's shift

        Parameters
        ----------
        time : int
            time.

        Returns
        -------
        bool
            Is true if the time is on the agent's shift.

        """
        shift_name = self.shift._shift_name
        (time_start, time_finish) = self._shifting.getShiftTimes(shift_name)

        if type(time) != int and type(time) != float:
            raise ValueError('time must be integer ' + type(time))

        time = int(time)
        time = time % (24 * 3600)

        if time_start > time_finish:
            new_time_finish = time_finish + 24 * 3600
            time_finish = new_time_finish
            if time < time_start:
                time = time + 24 * 3600

        if time >= time_start and time < time_finish:
            return True
        else:
            return False

    def getDistanceFromCoordinate(self, destination_coordination):
        coord = self.current_location.coord.get_coord()
        cur_x = coord[0]
        cur_y = coord[1]

        dest_x = destination_coordination[0]
        dest_y = destination_coordination[1]

        distance = ((cur_x - dest_x) ** 2 + (cur_y - dest_y) ** 2) ** 0.5
        return distance

    def _estimateTimeOfArival(self, destination_coordination):
        distance_with_method_of_choice = self.getDistanceFromCoordinate(
            destination_coordination
        )
        time = distance_with_method_of_choice / self._avg_speed

        return time

    def getAgentShiftEndTime(self, cur_time):
        num_of_days = int(cur_time / (24 * 3600))

        shift_name = self.shift._shift_name
        (time_start, time_finish) = self._shifting.getShiftTimes(shift_name)

        if time_start < time_finish or cur_time % (24 * 3600) <= time_finish:
            return time_finish + 24 * 3600 * num_of_days
        else:
            return time_finish + 24 * 3600 * (num_of_days + 1)

    def getShiftLength(self):
        shift_name = self.shift._shift_name
        (time_start, time_finish) = self._shifting.getShiftTimes(shift_name)

        if time_start < time_finish:
            return time_finish - time_start
        else:
            return 24 * 3600 - time_start + time_finish

    def setJob(
        self,
        node_name,
        action,
        entity,
        effect_definition_name,
        method_name,
        time_arival,
        time_done,
        iOnGoing,
    ):
        if self.isWorking == True:
            raise ValueError('The curent agent is working')

        self.isWorking = True
        self.cur_job_location = node_name
        self.cur_job_action = action
        self.cur_job_entity = entity
        self.cur_job_effect_definition_name = effect_definition_name
        self.cur_job_method_name = method_name
        self.cur_job_ongoing = iOnGoing
        self._time_of_arival = time_arival
        self.job_end_time = time_done


class Agents:
    def __init__(self, registry, shifting, jobs, restoration_log_book):
        # data:    is the
        # type:    agent type
        # sybtype: agent sub type
        # active:  active is true if it is the shift
        # ready:   read is true if the agent's active and it has nthing to do
        self._agents = pd.DataFrame(
            columns=['data', 'type', 'group', 'active', 'ready', 'available']
        )  # table that includes all data about agents including attributes for fast refinement, the index is the name of the agent (AKA agent_ID)
        self.group_names = {}
        self._shifting = shifting
        self._jobs = jobs
        self.restoration_log_book = restoration_log_book
        self.registry = registry

    def addAgent(self, agent_name, agent_type, definition):
        """Adds agent to the agent list

        Parameters
        ----------
        agent_type : str
            Given type name
        definition : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # number_of_agents = int(definition['Number'])
        agent_speed = self.registry.settings['crew_travel_speed']
        temp_agent_data = AgentData(
            agent_name,
            agent_type,
            float(definition['cur_x']),
            float(definition['cur_y']),
            definition['shift_name'],
            definition['group'],
            float(definition['base_x']),
            float(definition['base_y']),
            self._shifting,
            agent_speed,
        )
        self._agents.loc[agent_name] = [
            temp_agent_data,
            agent_type,
            definition['group'],
            False,
            False,
            False,
        ]
        if agent_type not in self.group_names:
            self.group_names[agent_type] = definition['group_name']

    def setActiveAgents(self, active_agent_ID_list):
        """Set agents active by a list of agents' ID

        Parameters
        ----------
        active_agent_ID_list : list
            agents ID name

        Returns
        -------
        None.

        """
        for active_agent_ID in active_agent_ID_list:
            self._agents['active'].loc[active_agent_ID] = True

    def getAgentGroupTagList(self, typed_ready_agent):
        ret = [None]
        agent_type = typed_ready_agent['type'].iloc[0]
        if 'group' in typed_ready_agent:
            ret = typed_ready_agent['group'].unique().tolist()
            number_of_groups = len(ret)
            if number_of_groups == 0:
                raise RuntimeError(
                    'There zero group for agent type: ' + repr(agent_type)
                )
            if None in ret:
                raise RuntimeError('None in agent type: ' + repr(agent_type))
        return ret, self.group_names[agent_type]

    def getAllAgentTypes(self):
        return self._agents['type'].unique().tolist()

    def getAllAgent(self):
        """Get a copy of all agent dataframe.

        Returns
        -------
        A copy of all agent dataFrame

        """
        return self._agents.copy(deep=True)

    def setChangeShift(self, time, working_check=True):
        for name, agent in self._agents.iterrows():
            if self._agents.loc[name, 'data'].isOnShift(time):
                if (
                    self._agents.loc[name, 'active'] == False
                ):  # if agent is active already and is on shift, it means that the agent has been active before teh shift change event
                    if self._agents.loc[name, 'available'] == True:
                        self._agents.loc[name, 'active'] = True
                        self._agents.loc[name, 'ready'] = True

            else:
                if (
                    self._agents.loc[name, 'ready'] == True
                    and self._agents.loc[name, 'data'].isWorking == True
                ):
                    raise RuntimeError(name + ' is working')
                self._agents.loc[name, 'active'] = False
                self._agents.loc[name, 'ready'] = False

    def initializeActiveAgents(self, time):
        for name, agent in self._agents.iterrows():
            if self._agents.loc[name, 'data'].isOnShift(time):
                self._agents.loc[name, 'active'] = True
            else:
                self._agents.loc[name, 'active'] = False

    def initializeReadyAgents(self):
        for name, agent in self._agents.iterrows():
            if self._agents.loc[name, 'active'] == True:
                self._agents.loc[name, 'ready'] = True
            else:
                self._agents.loc[name, 'ready'] = False

    def getReadyAgents(self):
        temp = self._agents[
            (self._agents['ready'] == True) & (self._agents['available'] == True)
        ]
        check_temp = temp['active'].all()
        if check_temp == False:
            print(temp[temp['active'] == False])
            raise ValueError('At least one agent is ready although is not on shift')

        return temp

    def getAvailabilityRatio(self, agent_type, time):
        if agent_type == 'WQOperator' or agent_type == 'WQWorker':
            av_data = pd.Series(data=[0, 0.5, 1], index=[0, 4, 24])
        elif agent_type == 'CONT':
            av_data = pd.Series(data=[0, 0, 1], index=[0, 48, 49])
        else:
            av_data = pd.Series(data=[1, 0.5, 1], index=[0, 4, 48])
        temp = av_data
        time = time / 3600
        if time in temp:
            return temp[time]
        if time > temp.index.max():
            return temp[temp.index.max()]
        else:
            temp[time] = np.nan
            temp.sort_index(inplace=True)
            temp.interpolate(method='index', inplace=True)
            return temp[time]

    def getDefaultAvailabilityRatio(agent_type, self):
        if agent_type == 'WQOperator' or agent_type == 'WQWorker':
            return 0
        else:
            return 1

    def assignsJobToAgent(
        self,
        agent_name,
        node_name,
        entity,
        action,
        time,
        wn,
        reminded_time,
        number_of_damages,
        orginal_element,
    ):
        if self._agents.loc[agent_name, 'active'] != True:
            raise ValueError('Agent ' + agent_name + ' is not active')
        if self._agents.loc[agent_name, 'ready'] != True:
            raise ValueError('Agent ' + agent_name + ' is not ready')

        if self._agents.loc[agent_name, 'data'].isOnShift(time) != True:
            raise ValueError('Agent ' + agent_name + ' is not on shift')

        if self._agents.loc[agent_name, 'data'].isWorking == True:
            raise ValueError('Agent ' + agent_name + ' is working')

        # logger.debug('Assiging job to '+agent_name)
        real_node_name = node_name
        if self._jobs._rm.entity[entity] == 'DISTNODE':
            damage_data = self._jobs._rm._registry.getDamageData(
                'DISTNODE', iCopy=False
            )
            if 'virtual_of' in damage_data.columns:
                real_node_name = get_node_name(node_name, damage_data)

        coord = wn.get_node(real_node_name).coordinates
        agent_type = self._agents.loc[agent_name, 'type']

        _ETA = self._agents.loc[agent_name, 'data']._estimateTimeOfArival(coord)
        effect_definition_name = self._jobs.getEffectDefinitionName(
            agent_type, action, entity
        )
        method_name = self._jobs.chooseMethodForCurrentJob(
            node_name, effect_definition_name, entity
        )

        if method_name == None:
            raise ValueError(
                'No method is applicale for ' + repr(effect_definition_name)
            )

        if reminded_time == None:
            _ETJ = self._jobs.getAJobEstimate(
                orginal_element,
                agent_type,
                entity,
                action,
                method_name,
                number_of_damages,
            )
        else:
            _ETJ = int(reminded_time)
            if reminded_time < 0:
                raise ValueError('Something wrong here: ' + repr(reminded_time))

        if effect_definition_name != 'CHECK':
            method_line = self._jobs._effect_data[effect_definition_name][
                method_name
            ]
        else:
            method_line = [{'EFFECT': 'CHECK'}]

        effects_only = [i['EFFECT'] for i in method_line]

        collective = None
        if 'SKIP' in effects_only:
            return (False, 'SKIP', None, collective)
        elif 'FASTCHECK' in effects_only:
            return (False, 'FASTCHECK', None, collective)
        elif 'RECONNECT' in effects_only:
            collective = 'BYPASS'
        elif 'ADD_RESERVOIR' in effects_only:
            collective = 'ADD_RESERVOIR'
        elif 'REMOVE_LEAK' in effects_only:
            collective = 'REMOVE_LEAK'
        elif 'ISOLATE_DN' in effects_only:
            collective = 'ISOLATE_DN'

        if _ETA < 0 or _ETJ <= 0:
            print(
                str(_ETA)
                + '  '
                + str(effect_definition_name)
                + '  '
                + str(orginal_element)
            )
            print(str(method_name) + '  ' + str(_ETJ))
            raise ValueError('Subzero ETA or sub-equal-zero ETJ')

        end_time = time + _ETA + _ETJ
        agent_shift_change_time = self._agents.loc[
            agent_name, 'data'
        ].getAgentShiftEndTime(time)
        shift_length = self._agents.loc[agent_name, 'data'].getShiftLength()

        minimum_job_time = self._jobs._rm._registry.settings['minimum_job_time']
        if end_time <= agent_shift_change_time:
            iget = 'INSIDE_SHIFT'
            iOnGoing = False
        elif (
            end_time > agent_shift_change_time
            and (shift_length - 2 * 3600) < _ETJ
            and (time + _ETA + 2 * 3600) < agent_shift_change_time
        ):
            iget = 'OUTSIDE_SHIFT'
            iOnGoing = True
        else:
            # logger.warning(agent_name+',  '+node_name+', '+repr(end_time))
            iget = 'ShortOfTime'

        if iget == 'ShortOfTime':
            return (False, iget, None, collective)
        self._agents.loc[agent_name, 'data'].current_location.coord.set_coord(
            coord[0], coord[1]
        )
        self._agents.loc[agent_name, 'data'].setJob(
            node_name,
            action,
            entity,
            effect_definition_name,
            method_name,
            time + _ETA,
            time + _ETA + _ETJ,
            iOnGoing,
        )
        self._agents.loc[agent_name, 'ready'] = False
        self.restoration_log_book.addAgentActionToLogBook(
            agent_name,
            node_name,
            entity,
            action,
            time,
            end_time,
            _ETA,
            effect_definition_name,
            method_name,
            iFinished=not iOnGoing,
        )

        return (True, iget, _ETJ, collective)

    def getJobEndTime(self, agent_name, icheck=True):
        end_time = self._agents.loc[agent_name, 'data'].job_end_time
        if icheck == True and end_time == None:
            raise ValueError('No Time is assigned to agent')
        if (
            icheck == True
            and self._agents.loc[agent_name, 'data'].isWorking == False
        ):
            raise ValueError('The agent is not working')
        return end_time

    def getJobArivalTime(self, agent_name, icheck=True):
        arival_time = self._agents.loc[agent_name, 'data']._time_of_arival
        if icheck == True and arival_time == None:
            raise ValueError('No Time is assigned to agent')
        if (
            icheck == True
            and self._agents.loc[agent_name, 'data'].isWorking == False
        ):
            raise ValueError('The agent is not working')
        return arival_time

    def releaseAgent(self, agent_name):
        if self._agents.loc[agent_name, 'ready'] == True:
            raise ValueError(agent_name + ' is already ready')
        if self._agents.loc[agent_name, 'active'] != True:
            raise ValueError(agent_name + ' is not active')
        if self._agents.loc[agent_name, 'data'].isWorking == False:
            raise ValueError(agent_name + ' is not working')

        self._agents.loc[agent_name, 'ready'] = True

        self._agents.loc[agent_name, 'data'].isWorking = False
        self._agents.loc[agent_name, 'data'].cur_job_location = None
        self._agents.loc[agent_name, 'data'].cur_job_action = None
        self._agents.loc[agent_name, 'data'].cur_job_entity = None
        self._agents.loc[agent_name, 'data']._time_of_arival = None
        self._agents.loc[agent_name, 'data'].cur_job_effect_definition_name = None
        self._agents.loc[agent_name, 'data'].cur_job_method_name = None
        self._agents.loc[agent_name, 'data'].job_end_time = None
        self._agents.loc[agent_name, 'data'].cur_job_ongoing = None


class AgentShift:
    def __init__(self, agent_name, name):  # , shifting_obj):
        self._agent_name = agent_name
        self._shift_name = name
        # shifting_obj.addAgentShift(self._agent_name, self._shift_name)


class Shifting:
    def __init__(self):
        self._all_agent_shift_data = {}
        self._shift_data = pd.DataFrame(columns=['begining', 'end'])

    def addShift(self, name, begining, ending):
        """Adds a shift to shift registry

        Parameters
        ----------
        name : str
            Shift's name.
        begining : int
            shift's begining time.
        ending : int
            shifts ending time.

        Raises
        ------
        ValueError
            if shift name is already in the registry,
            if the name is not string,
            if the begining time is not int,
            if the ending time is not int,
            if begining time is bigger than 24*3600,
            if ending time is bigger than 24*3600.

        Returns
        -------
        None.

        """
        if name in self._shift_data:
            raise ValueError('Shift name already registered')
        if type(begining) != int and type(begining) != float:
            raise ValueError('Begining time must be integer: ' + str(type(begining)))
        if type(ending) != int and type(ending) != float:
            raise ValueError('Ending time must be integer: ' + str(type(ending)))
        if begining > 24 * 3600:
            raise ValueError('begining time is bigger than 24*3600' + str(begining))
        if ending > 24 * 3600:
            raise ValueError('Ending time is bigger than 24*3600' + str(ending))
        begining = int(begining)
        ending = int(ending)

        self._shift_data.loc[name] = [begining, ending]

    def getShiftTimes(self, name):
        return (
            self._shift_data['begining'].loc[name],
            self._shift_data['end'].loc[name],
        )

    def getNextShiftTime(self, time):
        daily_time = time % (24 * 3600)
        num_of_days = int(time / (24 * 3600))

        next_shift_candidate = pd.Series()

        for shift_name, shift_data in self._shift_data.iterrows():
            beg_time = shift_data[0]
            end_time = shift_data[1]

            if beg_time > end_time and daily_time < end_time:
                beg_time -= 24 * 3600
            elif beg_time > end_time and daily_time >= end_time:
                # beg_time += 24*3600*num_of_days
                end_time += 24 * 3600

            if daily_time < end_time and daily_time >= beg_time:
                next_shift_candidate.loc[shift_name] = (
                    end_time + 24 * 3600 * num_of_days
                )
        change_shift_time = next_shift_candidate.min()
        # if beg_time > end_time:
        # next_shift_time = time +(change_shift_time - daily_time)
        # else:

        return change_shift_time

    def assignShiftToAgent(self, agent_ID, shift_name):
        """Assigns shoft to agent

        Parameters
        ----------
        agent_ID : str
            Agent's ID.
        shift_name : str
            Shoft's name

        Raises
        ------
        ValueError
            if agent_ID is already in agent's shoft data(probably already assigned,
            if shift name does not exist in shift registry.

        Returns
        -------
        None.

        """
        if agent_ID in self._all_agent_shift_data:
            raise ValueError('The agent ID currently in Agent ALl Shifts')
        if shift_name not in self._shift_data:
            raise ValueError('shift data is not in registered as shifts')

        self._all_agent_shift_data[agent_ID] = shift_name


class DispatchRule:
    def __init__(self, settings, method='deterministic', exclude=None):
        self.settings = settings
        self._rules = {}
        self._cumulative = {}

        if 'PIPE' not in exclude:
            self._rules['PIPE'] = self.settings['pipe_damage_discovery_model'][
                'time_discovery_ratio'
            ]
            # data2=pd.Series([0.90, 0.01, 0.01, 0.04, 0.04, 0, 0], index = [3600*n for n in [0, 12, 24, 36, 48, 60, 72]])

        if 'DISTNODE' not in exclude:
            self._rules['DISTNODE'] = self.settings['node_damage_discovery_model'][
                'time_discovery_ratio'
            ]
            # data=pd.Series([0, 0.67, 0.07, 0.07, 0.07, 0.07, 0.05], index = [3600*n for n in [0, 12, 24, 36, 48, 60, 72]])

        self._rules['GNODE'] = self.settings['Gnode_damage_discovery_model'][
            'time_discovery_ratio'
        ]
        self._rules['TANK'] = self.settings['tank_damage_discovery_model'][
            'time_discovery_ratio'
        ]
        self._rules['PUMP'] = self.settings['pump_damage_discovery_model'][
            'time_discovery_ratio'
        ]
        self._rules['RESERVOIR'] = self.settings['reservoir_damage_discovery_model'][
            'time_discovery_ratio'
        ]

        if method == 'deterministic':
            pass
        else:
            raise ValueError('Unknown dispatch Rule: ' + method)

        # for key in exclude:
        # self._rules.pop(key)

        for key, d in self._rules.items():
            self._cumulative[key] = self._rules[key].cumsum()

    def getDiscoveredPrecentage(self, time):
        res = {}
        for key in self._cumulative:
            temp = self._cumulative[key].copy()
            if time in temp.index:
                res[key] = temp[time]
            elif time < temp.index.min():
                res[key] = temp[temp.index.min()]
            elif time > temp.index.max():
                res[key] = temp[temp.index.max()]
            else:
                temp[time] = np.nan
                temp.sort_index(inplace=True)
                temp.interpolate(method='index', inplace=True)
                res[key] = temp[time]
        return res


class Dispatch:
    def __init__(self, restoration, settings, discovery_interval=0, method='old'):
        self.settings = settings
        self.method = method
        self.discovery_interval = discovery_interval
        self._rm = restoration
        self._discovered_entity = {}
        self._init_time = self._rm.restoration_start_time

        exclude = []

        if settings['pipe_damage_discovery_model']['method'] == 'leak_based':
            exclude.append('PIPE')
        elif settings['pipe_damage_discovery_model']['method'] == 'time_based':
            pass
        else:
            raise ValueError(
                'Unknown pipe damage discovery method in settings: '
                + repr(settings['pipe_damage_discovery_model']['method'])
            )

        if settings['node_damage_discovery_model']['method'] == 'leak_based':
            exclude.append('DISTNODE')
        elif settings['node_damage_discovery_model']['method'] == 'time_based':
            pass
        else:
            raise ValueError(
                'Unknown Node damage discovery method in settings: '
                + repr(settings['node_damage_discovery_model']['method'])
            )

        self._rules = DispatchRule(settings, exclude=exclude)
        self._last_discovered_number = {}
        for el in self._rm.ELEMENTS:
            if el in exclude:
                continue
            self._last_discovered_number[el] = 0

        self._rm._registry.addAttrToPipeDamageTable('discovered', False)
        self._rm._registry.addAttrToDistNodeDamageTable('discovered', False)

    def updateDiscovery(self, time):
        if time < self._rm.restoration_start_time:
            print('Time is less than init time')

        else:
            # if self.method == 'old':
            # time_since_dispatch_activity = time - self._rm.restoration_start_time
            # discovered_ratios         = self._rules.getDiscoveredPrecentage(time_since_dispatch_activity)
            # discovered_damage_numbers = self._getDamageNumbers(discovered_ratios)
            # self._updateDamagesNumbers(discovered_damage_numbers)

            if (
                self.settings['pipe_damage_discovery_model']['method']
                == 'leak_based'
            ):
                pipe_leak_criteria = self.settings['pipe_damage_discovery_model'][
                    'leak_amount'
                ]
                pipe_leak_time_span = self.settings['pipe_damage_discovery_model'][
                    'leak_time'
                ]

                pipe_damage_table = self._rm._registry._pipe_damage_table
                not_discovered_pipe_damage_table = pipe_damage_table[
                    pipe_damage_table['discovered'] == False
                ]
                to_be_checked_node_list = list(
                    not_discovered_pipe_damage_table.index
                )
                breaks_not_discovered_pipe_damage_table = pipe_damage_table[
                    (pipe_damage_table['discovered'] == False)
                    & (pipe_damage_table['damage_type'] == 'break')
                ]
                not_discovered_break_node_B = (
                    self._rm._registry._pipe_break_history.loc[
                        breaks_not_discovered_pipe_damage_table.index, 'Node_B'
                    ]
                )
                not_dicovered_node_B_list = not_discovered_break_node_B.to_list()
                to_be_checked_node_list.extend(not_dicovered_node_B_list)
                # break_pair = zip(breaks_not_discovered_pipe_damage_table, not_discovered_break_node_B)
                # not_discovered_pipe_damage_name_list = list(not_discovered_pipe_damage_table.index)
                # breaks_not_discovered_pipe_damage_table
                # all_nodes_name_list = set(self._rm._registry.result.columns)
                available_nodes = set(
                    self._rm._registry.result.node['demand'].columns
                )
                to_be_checked_node_list = set(to_be_checked_node_list)
                shared_nodes_name_list = (
                    to_be_checked_node_list.union(available_nodes)
                    - (available_nodes - to_be_checked_node_list)
                    - (to_be_checked_node_list - available_nodes)
                )
                if len(shared_nodes_name_list) > 0:
                    leaking_nodes_result = self._rm._registry.result.node['demand'][
                        list(shared_nodes_name_list)
                    ]

                    leaking_nodes_result = leaking_nodes_result.loc[
                        (leaking_nodes_result.index > (time - pipe_leak_time_span))
                    ]
                    discovered_bool = leaking_nodes_result >= pipe_leak_criteria
                    discovered_bool_temp = discovered_bool.any()
                    discovered_bool_temp = discovered_bool_temp[
                        discovered_bool_temp == True
                    ]
                    to_be_discoverd = discovered_bool_temp.index.to_list()

                    # time1    = leaking_nodes_result.index[1:]
                    # time2    = leaking_nodes_result.index[0:-1]
                    # time_dif = (pd.Series(time1) - pd.Series(time2))
                    # leaking_nodes_result = leaking_nodes_result.drop(leaking_nodes_result.index[-1])

                    # leaking_nodes_result.index = time_dif.to_numpy()
                    # leaking_nodes_result = leaking_nodes_result.apply(lambda x: x.values * x.index)
                    # summed_water_loss = leaking_nodes_result.sum()
                    # to_be_discoverd = summed_water_loss[summed_water_loss > 3600*2*0.2]
                    discovery_list = set()
                    # to_be_discoverd = list(to_be_discoverd.index)
                    for discovery_candidate in to_be_discoverd:
                        if discovery_candidate in not_dicovered_node_B_list:
                            candidate_break_A = not_discovered_break_node_B[
                                not_discovered_break_node_B == discovery_candidate
                            ].index[0]
                            discovery_list.add(candidate_break_A)
                        else:
                            discovery_list.add(discovery_candidate)
                    # discovery_list = list(discovery_list)
                    pipe_damage_table.loc[discovery_list, 'discovered'] = True

            if (
                self.settings['node_damage_discovery_model']['method']
                == 'leak_based'
            ):
                node_leak_criteria = self.settings['node_damage_discovery_model'][
                    'leak_amount'
                ]
                node_leak_time_span = self.settings['node_damage_discovery_model'][
                    'leak_time'
                ]

                nodal_damage_table = self._rm._registry._node_damage_table
                not_discovered_nodal_damage_table = nodal_damage_table[
                    nodal_damage_table['discovered'] == False
                ]
                if 'virtual_of' in not_discovered_nodal_damage_table.columns:
                    to_be_checked_node_list = list(
                        not_discovered_nodal_damage_table['virtual_of']
                    )
                else:
                    to_be_checked_node_list = list(
                        not_discovered_nodal_damage_table.index
                    )
                available_leak_nodes = set(
                    self._rm._registry.result.node['leak'].columns
                )
                to_be_checked_node_list = set(to_be_checked_node_list)
                shared_nodes_name_list = (
                    to_be_checked_node_list.union(available_leak_nodes)
                    - (available_leak_nodes - to_be_checked_node_list)
                    - (to_be_checked_node_list - available_leak_nodes)
                )
                if len(shared_nodes_name_list) > 0:
                    shared_nodes_name_list = list(shared_nodes_name_list)
                    leaking_nodes_result = self._rm._registry.result.node['leak'][
                        shared_nodes_name_list
                    ]
                    leaking_nodes_result = leaking_nodes_result.sort_index()

                    if 'virtual_of' in not_discovered_nodal_damage_table.columns:
                        leaking_number_of_damages = (
                            not_discovered_nodal_damage_table.groupby('virtual_of')[
                                'Number_of_damages'
                            ].sum()
                        )
                    else:
                        leaking_number_of_damages = (
                            not_discovered_nodal_damage_table.loc[
                                shared_nodes_name_list, 'Number_of_damages'
                            ]
                        )

                    leaking_nodes_result = leaking_nodes_result.loc[
                        (leaking_nodes_result.index > (time - node_leak_time_span))
                    ]
                    normalized_summed_water_loss = (
                        leaking_nodes_result / leaking_number_of_damages
                    )
                    discovered_bool = (
                        normalized_summed_water_loss >= node_leak_criteria
                    )
                    discovered_bool_temp = discovered_bool.any()
                    discovered_bool_temp = discovered_bool_temp[
                        discovered_bool_temp == True
                    ]
                    discovered_list = discovered_bool_temp.index.to_list()
                    if 'virtual_of' in not_discovered_nodal_damage_table.columns:
                        discovered_list = (
                            nodal_damage_table[
                                nodal_damage_table['virtual_of'].isin(
                                    discovered_list
                                )
                            ]
                        ).index

                    nodal_damage_table.loc[discovered_list, 'discovered'] = True
                # else:

            time_since_dispatch_activity = time - self._rm.restoration_start_time
            discovered_ratios = self._rules.getDiscoveredPrecentage(
                time_since_dispatch_activity
            )
            discovered_damage_numbers = self._getDamageNumbers(discovered_ratios)
            self._updateDamagesNumbers(discovered_damage_numbers)

            # else:
            # raise ValueError('Unknown method: '+repr(self.method))

    def _getDamageNumbers(self, discovered_ratios):
        num_damaged_entity = {}

        for el in discovered_ratios:
            if discovered_ratios[el] - 1 > 0:
                if discovered_ratios[el] - 1.000001 > 0:
                    raise ValueError(
                        'ratio is bigger than 1: '
                        + str(discovered_ratios[el])
                        + ' in element = '
                        + el
                    )
                else:
                    discovered_ratios[el] = 1
            temp = len(self._rm._registry.getDamageData(el))
            num_damaged_entity[el] = int(np.round(temp * discovered_ratios[el]))
        return num_damaged_entity

    def _updateDamagesNumbers(self, discovered_numbers):
        for el in discovered_numbers:
            if self._last_discovered_number[el] > discovered_numbers[el]:
                raise ValueError(
                    'Discovered number is less than what it used to be in element '
                    + el
                )
            elif self._last_discovered_number[el] < discovered_numbers[el]:
                refined_damaged_table = self._rm._registry.getDamageData(el)
                if len(refined_damaged_table) < discovered_numbers[el]:
                    raise ValueError(
                        'discovered number is bigger than all damages in element'
                        + el
                    )

                discovered_damage_table = refined_damaged_table[
                    refined_damaged_table['discovered'] == True
                ]
                if discovered_numbers[el] <= len(discovered_damage_table):
                    continue
                undiscovered_damage_table = refined_damaged_table[
                    refined_damaged_table['discovered'] == False
                ]

                # =============================================================================
                #                 used_number = []
                #                 i = 0
                #                 while i < (discovered_numbers[el] - self._last_discovered_number[el]):
                #                     picked_number = random.randint(0,len(undiscovered_damage_table)-1)
                #                     if picked_number not in used_number:
                #                         used_number.append(picked_number)
                #                         i += 1
                #                     else:
                #                         pass
                # =============================================================================
                if len(undiscovered_damage_table) > 0:
                    used_number = random.sample(
                        range(len(undiscovered_damage_table)),
                        discovered_numbers[el] - len(discovered_damage_table),
                    )
                else:
                    used_number = []
                for i in used_number:
                    temp_index = undiscovered_damage_table.index[i]
                    self._rm._registry.updateElementDamageTable(
                        el, 'discovered', temp_index, True, icheck=True
                    )

                if el == 'PIPE':
                    refined_damaged_table = self._rm._registry.getDamageData(el)
                    discovered_damage_table = refined_damaged_table[
                        refined_damaged_table['discovered'] == True
                    ]
                self._last_discovered_number[el] = discovered_numbers[el]


class Priority:
    def __init__(self, restoration):
        self._data = {}
        self._rm = restoration

    def addData(self, agent_type, priority, order):
        if agent_type not in self._data:
            self._data[agent_type] = pd.Series(index=[priority], data=[order])
        else:
            temp = self._data[agent_type]
            if priority in temp.index:
                raise ValueError(
                    'Prority redefiend. type: '
                    + agent_type
                    + ' & priority: '
                    + str(priority)
                )
            self._data[agent_type].loc[priority] = order

    def getPriority(self, agent_type, priority):
        if agent_type not in self._data:
            raise ValueError(
                'The agent type('
                + repr(agent_type)
                + ') is not defined in the prioity:'
                + repr(priority)
            )

        temp = self._data[agent_type]

        if priority not in temp.index:
            raise ValueError(
                'prioirty not in priority data. Agent_type: '
                + agent_type
                + ' & PriorityL '
                + priority
            )

        return temp.loc[priority]

    def getHydSigDamageGroups(self):
        damage_group_list = set()
        for crew_type in self._data:
            whole_priority_list = self._data[crew_type]
            primary_priority_list = whole_priority_list.loc[1]
            secondary_priority_list = whole_priority_list.loc[2]
            i = 0
            for cur_second_priority in secondary_priority_list:
                if cur_second_priority.upper() == 'HYDSIG':
                    cur_damage_group = primary_priority_list[i][1]
                    damage_group_list.add(cur_damage_group)
                i += 1
        return damage_group_list

    def sortDamageTable(
        self,
        wn,
        entity_data,
        entity,
        agent_type,
        target_priority_index,
        order_index,
        target_priority=None,
    ):
        all_priority_data = self._data[agent_type]
        target_priority_list = all_priority_data.loc[target_priority_index]

        if len(target_priority_list) == 0:
            return entity_data

        name_sugest = 'Priority_' + str(target_priority_index) + '_dist'

        if target_priority == None:
            target_priority = target_priority_list[order_index]

        if target_priority == None:
            return entity_data

        elif target_priority in self._rm.proximity_points:
            Proximity_list = self._rm.proximity_points[target_priority]
            node_name_list = list(entity_data.index)
            for node_name in node_name_list:
                # Sina: you can enhance the run time speed with having x, y coordinates in the damage table and not producing and droping them each time
                node_name_vir = get_node_name(node_name, entity_data)
                coord = wn.get_node(node_name_vir).coordinates
                entity_data.loc[node_name, 'X_COORD'] = coord[0]
                entity_data.loc[node_name, 'Y_COORD'] = coord[1]
            counter = 1
            columns_to_drop = []
            for x, y in Proximity_list:
                name_sug_c = name_sugest + '_' + str(counter)
                columns_to_drop.append(name_sug_c)
                entity_data[name_sug_c] = (
                    (entity_data['X_COORD'] - x) ** 2
                    + (entity_data['Y_COORD'] - y) ** 2
                ) ** 0.5
                counter += 1
            dist_only_entity_table = entity_data[columns_to_drop]
            min_dist_entity_table = dist_only_entity_table.min(axis=1)
            entity_data.loc[:, name_sugest] = min_dist_entity_table
            entity_data.sort_values(by=name_sugest, ascending=True, inplace=True)
            columns_to_drop.append(name_sugest)
            columns_to_drop.append('X_COORD')
            columns_to_drop.append('Y_COORD')
            entity_data.drop(columns=columns_to_drop, inplace=True)

        # Sina: It does nothing. When there are less damage location within
        # the priority definition for the crew type, thsi works fine, but
        # when there are more damage location within the priority definiton,
        # it does not gurantee that only teh cloest damage locations to the
        # crew-type agents are matched to jobs
        elif target_priority.upper() == 'CLOSEST':
            pass
        elif target_priority.upper() == 'HYDSIGLASTFLOW':
            element_type = self._rm.entity[entity]
            if element_type != 'PIPE':
                entity_data = self.sortDamageTable(
                    entity_data,
                    entity,
                    agent_type,
                    target_priority_index,
                    order_index,
                    target_priority='CLOSEST',
                )
            else:
                all_time_index = self._rm._registry.result.link['flowrate'].index[
                    : self._rm.restoration_start_time + 1
                ]
                pipe_name_list = entity_data.loc[:, 'Orginal_element']
                last_valid_time = [
                    cur_time
                    for cur_time in all_time_index
                    if cur_time not in self._rm._registry.result.maximum_trial_time
                ]
                last_valid_time.sort()
                if len(last_valid_time) == 0:
                    last_valid_time = self._rm.restoration_start_time
                else:
                    last_valid_time = last_valid_time[-1]

                name_sugest = 'Priority_' + str(target_priority_index) + '_dist'
                flow_rate = (
                    self._rm._registry.result.link['flowrate']
                    .loc[last_valid_time, pipe_name_list]
                    .abs()
                )
                entity_data.loc[:, name_sugest] = flow_rate.to_list()
                entity_data.sort_values(name_sugest, ascending=False, inplace=True)
                entity_data.drop(columns=name_sugest, inplace=True)

        elif (
            target_priority in self._rm.proximity_points
            and target_priority != 'WaterSource2'
        ):
            all_node_table = self._rm._registry.all_node_table
            Proximity_list = self._rm.proximity_points[target_priority]
            node_name_list = list(entity_data.index)
            for node_name in node_name_list:
                # Sina: you can enhance the run time speed with having x, y coordinates in the damage table and not producing and droping them each time
                node_name_vir = get_node_name(node_name, entity_data)
                coord = wn.get_node(node_name_vir).coordinates
                entity_data.loc[node_name, 'X_COORD'] = coord[0]
                entity_data.loc[node_name, 'Y_COORD'] = coord[1]
            counter = 1
            columns_to_drop = []

            g = nx.MultiDiGraph()

            for name, node in wn.nodes():
                g.add_node(name)
                nx.set_node_attributes(
                    g, name='pos', values={name: node.coordinates}
                )
                nx.set_node_attributes(g, name='type', values={name: node.node_type})

            for name, link in wn.links():
                start_node = link.start_node_name
                end_node = link.end_node_name
                g.add_edge(start_node, end_node, key=name)
                nx.set_edge_attributes(
                    g,
                    name='type',
                    values={(start_node, end_node, name): link.link_type},
                )

                try:
                    length = link.length
                    d = link.diameter
                    roughness = link.roughness
                    cost = (
                        length / np.power(d, 4.8655) / np.power(roughness, 1.852)
                        + 1 / d
                    )
                except:
                    cost = 0.00001

                weight = cost

                nx.set_edge_attributes(
                    g, name='weight', values={(start_node, end_node, name): weight}
                )

            g = g.to_undirected()

            for x, y in Proximity_list:
                point_length_vector = np.square(
                    all_node_table['X_COORD'] - x
                ) + np.square(all_node_table['Y_COORD'] - y)
                point_length_vector = np.sqrt(point_length_vector)
                closest_node_name = point_length_vector.idxmin()

                # print("closest_node_name= "+str(closest_node_name))

                orginal_pipe_name_list = entity_data['Orginal_element']
                damaged_pipe_node_list = [
                    self._rm._registry.undamaged_link_node_list[link_node_names]
                    for link_node_names in orginal_pipe_name_list
                ]
                try:
                    shortest_path_length = [
                        min(
                            nx.shortest_path_length(
                                g, closest_node_name, pipe_nodes_name[0], 'weight'
                            ),
                            nx.shortest_path_length(
                                g, closest_node_name, pipe_nodes_name[1], 'weight'
                            ),
                        )
                        for pipe_nodes_name in damaged_pipe_node_list
                    ]
                except nx.NetworkXNoPath:
                    shortest_path_length = []
                    for pipe_nodes_name in damaged_pipe_node_list:
                        start_node_name = pipe_nodes_name[0]
                        end_node_name = pipe_nodes_name[1]

                        try:
                            closest_path_from_start = nx.shortest_path_length(
                                g, closest_node_name, start_node_name, 'weight'
                            )
                        except nx.NetworkXNoPath:
                            closest_path_from_start = 10000000000000.0

                        try:
                            closest_path_from_end = nx.shortest_path_length(
                                g, closest_node_name, end_node_name, 'weight'
                            )
                        except nx.NetworkXNoPath:
                            closest_path_from_end = 10000000000000.0

                        cur_shortest_path_length = min(
                            closest_path_from_start, closest_path_from_end
                        )
                        shortest_path_length.append(cur_shortest_path_length)
                # print(shortest_path_length)

                name_sug_c = name_sugest + '_' + str(counter)
                columns_to_drop.append(name_sug_c)
                entity_data[name_sug_c] = shortest_path_length
                counter += 1
            dist_only_entity_table = entity_data[columns_to_drop]
            min_dist_entity_table = dist_only_entity_table.min(axis=1)
            entity_data.loc[:, name_sugest] = min_dist_entity_table
            entity_data.sort_values(by=name_sugest, ascending=True, inplace=True)
            columns_to_drop.append(name_sugest)
            columns_to_drop.append('X_COORD')
            columns_to_drop.append('Y_COORD')
            entity_data.drop(columns=columns_to_drop, inplace=True)
            # print(entity_data)
            # print("+++++++++++++++++++++++++++++++++++++++")

        # Sina: It does nothing. When there are less damage location within
        # the priority definition for the crew type, thsi works fine, but
        # when there are more damage location within the priority definiton,
        # it does not gurantee that only teh cloest damage locations to the
        # crew-type agents are matched to jobs

        elif target_priority.upper() == 'HYDSIG':
            element_type = self._rm.entity[entity]
            if element_type != 'PIPE':
                entity_data = self.sortDamageTable(
                    entity_data,
                    entity,
                    agent_type,
                    target_priority_index,
                    order_index,
                    target_priority='CLOSEST',
                )
            else:
                name_sugest = 'Priority_' + str(target_priority_index) + '_dist'
                hyd_sig = self._rm._registry.hydraulic_significance[
                    entity_data['Orginal_element']
                ]

                entity_data.loc[:, name_sugest] = hyd_sig.to_list()
                entity_data.sort_values(name_sugest, ascending=False, inplace=True)
                entity_data.drop(columns=name_sugest, inplace=True)

        # If element type is not leakable, it does nothing. IF nodes are not
        # Checked (i.e. check is not at the sequnce before the current action)
        # the leak data is real time leak for the damage location.
        elif target_priority.upper() == 'MOSTLEAKATCHECK':
            # real_node_name_list = []
            node_name_list = list(entity_data.index)
            name_sugest = (
                'Priority_' + str(target_priority_index) + '_leak_sina'
            )  # added sina so the possibility of a conflic of name is minimized
            # for node_name in node_name_list:
            # node_name_vir = get_node_name(node_name, entity_data)
            # real_node_name_list.append(node_name_vir)
            element_type = self._rm.entity[entity]
            leak_data = self._rm._registry.getMostLeakAtCheck(
                node_name_list, element_type
            )
            if leak_data is not None:
                entity_data.loc[node_name_list, name_sugest] = leak_data
                entity_data.sort_values(by=name_sugest, ascending=True, inplace=True)
                entity_data.drop(columns=[name_sugest], inplace=True)
            else:
                entity_data = self.sortDamageTable(
                    entity_data,
                    entity,
                    agent_type,
                    target_priority_index,
                    order_index,
                    target_priority='CLOSEST',
                )
        else:
            raise ValueError(
                'Unrcognized Secondary Primary: ' + repr(target_priority)
            )

        return entity_data

    def isAgentTypeInPriorityData(self, agent_type):
        return agent_type in self._data


class Jobs:
    def __init__(self, restoration):
        self._rm = restoration
        self._job_list = pd.DataFrame(
            columns=['agent_type', 'entity', 'action', 'time_argument']
        )
        self._effect_defualts = {}  # pd.DataFrame(columns=['effect_definition_name', 'method_name','argument','value'])
        self._effect_data = {}
        self._time_overwrite = {}
        self._final_method = {}
        self._once = {}

    def addEffect(self, effect_name, method_name, def_data):
        if effect_name not in self._effect_data:
            self._effect_data[effect_name] = None

        if self._effect_data[effect_name] != None:
            if method_name in self._effect_data[effect_name]:
                raise ValueError(
                    'Dupplicate method_name is given. Effect name: '
                    + str(effect_name)
                    + ', '
                    + str(method_name)
                )

        if self._effect_data[effect_name] == None:
            temp = {}
            temp[method_name] = def_data
            self._effect_data[effect_name] = temp
        else:
            self._effect_data[effect_name][method_name] = def_data

    def setJob(self, jobs_definition):
        self._job_list = pd.DataFrame.from_records(jobs_definition)

    def _filter(self, agent_type, entity, action):
        temp = self._job_list
        temp = temp[
            (
                temp[['agent_type', 'entity', 'action']]
                == [agent_type, entity, action]
            ).all(1)
        ]
        temp_length = len(temp)
        if temp_length > 1:
            raise ValueError('We have more than one job description')
        elif temp_length == 0:
            raise ValueError(
                'We have Zero one job description for agent type= '
                + repr(agent_type)
                + ', entity= '
                + repr(entity)
                + ', action= '
                + repr(action)
            )
        return temp

    def getAJobEstimate(
        self, orginal_element, agent_type, entity, action, method_name, number
    ):
        temp = self._filter(agent_type, entity, action)
        time_arg = temp['time_argument'].iloc[0]
        operation_name = temp['effect'].iloc[0]
        overwrite_key = (operation_name, method_name, orginal_element)
        if overwrite_key in self._time_overwrite:
            overwrite_data = self._time_overwrite[overwrite_key]
            if 'FIXED_TIME_OVERWRITE' in overwrite_data:
                time_arg = overwrite_data['FIXED_TIME_OVERWRITE']
            else:
                raise ValueError('Unknown Time Data')
        time = int(time_arg)
        # try:
        # time_arg = int(time_arg):
        # time = time_arg
        # except:
        # raise ValueError('Unknow time argument: '+str(type(time_arg)))

        once_flag = False
        if operation_name in self._once:
            if method_name in self._once[operation_name]:
                once_flag = True

        if once_flag == False:
            time = int(time * number)

        # IMPORTANT/sina
        if (method_name == 2 or method_name == 1) and action == 'reroute':
            pass

        return time

    def getMeanJobTime(self, agent_type, entity, action):
        temp = self._filter(agent_type, entity, action)
        time_arg = temp['time_argument'].iloc[0]
        if type(time_arg) == int:
            time = time_arg
        else:
            raise ValueError('Unknow time argument: ' + str(type(time_arg)))
        return time

    def getAllEffectByJobData(
        self, agent_type, action, entity, iWithout_data=True, iOnlyData=False
    ):
        temp = self._filter(agent_type, entity, action)
        all_effect_name = temp['effect'].iloc[0]

        if iOnlyData == True:
            return

    def addEffectDefaultValue(self, input_dict):
        _key = (
            input_dict['effect_definition_name'],
            input_dict['method_name'],
            input_dict['argument'],
        )

        if _key in self._effect_defualts:
            raise ValueError(
                'Duplicate effects definition: {0}, {1}, {2}'.format(
                    repr(input_dict['effect_definition_name']),
                    repr(input_dict['method_name']),
                    repr(input_dict['argument']),
                )
            )

        self._effect_defualts[_key] = input_dict[
            'value'
        ]  # self._effect_defualts.append(temp_s, ignore_index=True)

    def getEffectsList(self, effect_definition_name, method_name):
        if effect_definition_name == None:
            return []

        if effect_definition_name == 'CHECK':
            return [{'EFFECT': 'CHECK'}]
        all_methods = self._effect_data[effect_definition_name]
        effects_list = all_methods[method_name]
        return effects_list

    def getEffectDefinition(self, effect_definition_name, iWithout_data=True):
        all_methods = self._effect_data[effect_definition_name]

        if iWithout_data == True and 'DATA' in all_methods:
            all_methods = copy.deepcopy(all_methods)
            all_methods.pop('DATA')

        return all_methods

    def getEffectDefinitionName(self, agent_type, action, entity):
        temp = self._filter(agent_type, entity, action)
        effects_definition_name = temp['effect'].iloc[0]
        return effects_definition_name

    def chooseMethodForCurrentJob(self, node_name, effects_definition_name, entity):
        returned_method = None
        if effects_definition_name == None:
            return None
        elif (
            effects_definition_name == 'CHECK'
            or effects_definition_name == 'FASTCHECK'
            or effects_definition_name == 'SKIP'
        ):
            return effects_definition_name
        else:
            effects_definition = self.getEffectDefinition(
                effects_definition_name
            )  # self._effect_data[effects_definition_name]
            for method_name, effect_list in effects_definition.items():
                prob_applicability = self.iEffectApplicableByProbability(
                    effects_definition_name, method_name, node_name, entity
                )
                condition_applicability = self.iEffectApplicableByOtherConditions(
                    effects_definition_name, method_name, node_name, entity
                )
                if prob_applicability and condition_applicability:
                    returned_method = method_name
                    break

        if returned_method == None:
            try:
                returned_method = self._final_method[effects_definition_name]
            except:
                pass
        return returned_method

    def _getProbability(self, method, iCondition, element_type):
        if iCondition == True:
            if 'METHOD_PROBABILITY' in method:
                probability = method['METHOD_PROBABILITY']
            else:
                probability = 1
        # else:
        # if 'METHOD_PROBABILITY' in method:

    def _iConditionHolds(self, val1, con, val2):
        if con == 'BG':
            if val1 > val2:
                return True
            else:
                return False
        elif con == 'BG-EQ':
            if val1 >= val2:
                return True
            else:
                return False
        elif con == 'LT':
            if val1 < val2:
                return True
            else:
                return False
        elif con == 'LT-IF':
            if val1 <= val2:
                return True
            else:
                return False
        elif con == 'EQ':
            if val1 == val2:
                return True
            else:
                return False
        else:
            raise ValueError('Unrecognized condition: ' + repr(con))

    def getDefualtValue(self, effects_definition_name, method_name, argument):
        _default = self._effect_defualts
        value = _default.get((effects_definition_name, method_name, argument), None)

        return value

    def iEffectApplicableByOtherConditions(
        self, effects_definition_name, method_name, damaged_node_name, entity
    ):
        element_type = self._rm.entity[entity]
        effects_definition = self.getEffectDefinition(effects_definition_name)
        if element_type == 'DISTNODE':
            for single_effect in effects_definition[method_name]:
                if 'PIDR' in single_effect:
                    condition = single_effect['PIDR']
                    _con = condition[0]
                    _con_val = condition[1]
                    _PIDR_type = self.getDefualtValue(
                        effects_definition_name, method_name, 'PIDR_TYPE'
                    )
                    if _PIDR_type == None or _PIDR_type == 'ASSIGNED_DEMAND':
                        old_demand = self._rm._registry._node_damage_table.loc[
                            damaged_node_name, 'Demand1'
                        ]
                        new_demand = self._rm._registry._node_damage_table.loc[
                            damaged_node_name, 'Demand2'
                        ]
                    else:
                        raise ValueError('unrecognized Setting: ' + _PIDR_type)

                    _PIDR = new_demand / old_demand

                    iHold = self._iConditionHolds(_PIDR, _con, _con_val)

                    return iHold

        return True

    def iEffectApplicableByProbability(
        self, effects_definition_name, method_name, damaged_node_name, entity
    ):
        _prob = 0
        temp = self.getDefualtValue(
            effects_definition_name, method_name, 'METHOD_PROBABILITY'
        )
        if temp != None:
            _prob = temp
        try:
            self._check_probability(_prob)
        except Exception as e:
            print(
                'in Method bsaed Probability of method '
                + str(method_name)
                + ', and definition_name '
                + str(effects_definition_name)
                + ', :'
                + str(_prob)
            )
            raise ValueError(e)

        # =============================================================================
        #         if 'DEFAULT' in self._effect_data[effects_definition_name]:
        #             data = self._effect_data[effects_definition_name]['DEFAULT']
        #             if 'METHOD_PROBABILITY' in data:
        #                 if method_name in data['METHOD_PROBABILITY']:
        #                     _prob=data['METHOD_PROBABILITY'][method_name]
        #                     try:
        #                         _check_probability(_prob)
        #                     except Exception as e:
        #                         print('in Method bsaed Probability of method ' +method_name+ ', and definition_name '+effects_definition_name)
        #                         raise ValueError(e)
        # =============================================================================

        if 'DATA' in self._effect_data[effects_definition_name]:
            data = self._effect_data[effects_definition_name]['DATA']
            if 'METHOD_PROBABILITY' in data.columns:
                element_name = self._rm._registry.getOrginalElement(
                    damaged_node_name, self._rm.entity[entity]
                )

                # temp =data[(data[['ELEMENT_NAME','METHOD_NAME']]==[element_name, method_name]).all(1)]
                element_data = data[data['ELEMENT_NAME'] == element_name]
                if len(element_data) == 0:
                    pass
                else:
                    element_method_data = element_data[
                        element_data['METHOD_NAME'] == method_name
                    ]
                    if len(element_method_data) == 0:
                        _prob = 0
                    elif len(element_method_data) == 1:
                        _prob = element_method_data['METHOD_PROBABILITY'].iloc[0]
                    else:
                        raise ValueError(
                            'Number of probability found for element '
                            + element_name
                            + ', : '
                            + str(len(temp))
                        )
                    try:
                        self._check_probability(_prob)
                    except Exception as e:
                        print(
                            'in LIST of method '
                            + method_name
                            + ', and definition_name '
                            + effects_definition_name
                        )
                        raise ValueError(e)

        _rand = random.random()
        # if effects_definition_name == 'MJTRreroute':
        # print(str(method_name) + ' - ' + repr(_prob))
        logger.debug(_prob)
        if _rand < _prob:
            return True
        return False

    def _check_probability(self, _prob):
        mes = None
        _prob = float(_prob)
        if _prob < 0:
            raise ValueError('probability cannot be less than 0.')
        elif _prob > 1:
            res = False
            raise ValueError('probability cannot be more than 1.')


# =============================================================================
# class Effects():
#     def __init__(self, restoration_model):
#         #self._data_table = pd.DataFrame(columns=['effect', 'method_name', 'data_index'])
#
#
#
#
#
#         #self._data_table.loc[effect_name, 'method'] = method
#         #self._data_table.loc[effect_name, 'effect'] = effect
#         #self._data_table.loc[effect_name, 'connection'] = connection
#         #self._data_table.loc[effect_name, 'connection_value'] = connection_value
#         #self._data_table.loc[effect_name, 'CV'] = cv
# =============================================================================
