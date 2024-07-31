"""Created on Fri Dec 25 04:00:43 2020

@author: snaeimi
"""  # noqa: INP001, D400, D415

import copy
import logging
import random

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_node_name(node_name, table):  # noqa: ANN001, ANN201, D103
    if 'virtual_of' in table.columns:
        real_node_name = table.loc[node_name, 'virtual_of']
        if (
            real_node_name == None or real_node_name == np.nan  # noqa: E711, PLR1714
        ):  # SINA: probably NP.NAN does not work here. Correct it.
            real_node_name = node_name
        return real_node_name
    else:  # noqa: RET505
        return node_name


class Coordination:  # noqa: D101
    def __init__(self, X=None, Y=None, system=None):  # noqa: ANN001, ANN204, N803, D107
        self.x = X
        self.y = Y
        self.system = system

    def set_coord(self, X, Y, system=None):  # noqa: ANN001, ANN201, ARG002, N803, D102
        self.x = X
        self.y = Y

    def get_coord(self):  # noqa: ANN201, D102
        return (self.x, self.y)

    def set_system(self, system):  # noqa: ANN001, ANN201, D102
        self.system = system


class Location:  # noqa: D101
    def __init__(self, name, x, y):  # noqa: ANN001, ANN204, D107
        self.name = name
        self.coord = Coordination(x, y)


# =============================================================================
# class restoration_base():
#     def __init__(self):
#         self.coord = coordination()  # noqa: ERA001
#         self.ID = None  # noqa: ERA001
#         self.Object_typ = None  # noqa: ERA001
#
# =============================================================================


class AgentData:  # noqa: D101
    def __init__(  # noqa: ANN204, D107, PLR0913
        self,
        agent_name,  # noqa: ANN001
        agent_type,  # noqa: ANN001
        cur_x,  # noqa: ANN001
        cur_y,  # noqa: ANN001
        shift_name,  # noqa: ANN001
        base_name,  # noqa: ANN001
        base_x,  # noqa: ANN001
        base_y,  # noqa: ANN001
        shift_obj,  # noqa: ANN001
        agent_speed,  # noqa: ANN001
    ):
        if type(agent_type) != str:  # noqa: E721
            raise ValueError('agent type must be string')  # noqa: EM101, TRY003
        # if type(definition) != pd.Series:
        # raise ValueError('definiton must be a Pandas series')  # noqa: ERA001

        if type(cur_x) != float:  # noqa: E721
            raise ValueError('cur_x must be float')  # noqa: EM101, TRY003
        if type(cur_y) != float:  # noqa: E721
            raise ValueError('cur_y must be float')  # noqa: EM101, TRY003
        if type(base_x) != float:  # noqa: E721
            raise ValueError('base_x must be float')  # noqa: EM101, TRY003
        if type(base_y) != float:  # noqa: E721
            raise ValueError('base_y must be float')  # noqa: EM101, TRY003

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

    def isOnShift(self, time):  # noqa: ANN001, ANN201, N802
        """Checks if a time is on an agent's shift

        Parameters
        ----------
        time : int
            time.

        Returns
        -------
        bool
            Is true if the time is on the agent's shift.

        """  # noqa: D400, D401, D415
        shift_name = self.shift._shift_name  # noqa: SLF001
        (time_start, time_finish) = self._shifting.getShiftTimes(shift_name)

        if type(time) != int and type(time) != float:  # noqa: E721
            raise ValueError('time must be integer ' + type(time))

        time = int(time)
        time = time % (24 * 3600)

        if time_start > time_finish:
            new_time_finish = time_finish + 24 * 3600
            time_finish = new_time_finish
            if time < time_start:
                time = time + 24 * 3600

        if time >= time_start and time < time_finish:  # noqa: SIM103
            return True
        else:  # noqa: RET505
            return False

    def getDistanceFromCoordinate(self, destination_coordination):  # noqa: ANN001, ANN201, N802, D102
        coord = self.current_location.coord.get_coord()
        cur_x = coord[0]
        cur_y = coord[1]

        dest_x = destination_coordination[0]
        dest_y = destination_coordination[1]

        distance = ((cur_x - dest_x) ** 2 + (cur_y - dest_y) ** 2) ** 0.5
        return distance  # noqa: RET504

    def _estimateTimeOfArival(self, destination_coordination):  # noqa: ANN001, ANN202, N802
        distance_with_method_of_choice = self.getDistanceFromCoordinate(
            destination_coordination
        )
        time = distance_with_method_of_choice / self._avg_speed

        return time  # noqa: RET504

    def getAgentShiftEndTime(self, cur_time):  # noqa: ANN001, ANN201, N802, D102
        num_of_days = int(cur_time / (24 * 3600))

        shift_name = self.shift._shift_name  # noqa: SLF001
        (time_start, time_finish) = self._shifting.getShiftTimes(shift_name)

        if time_start < time_finish or cur_time % (24 * 3600) <= time_finish:
            return time_finish + 24 * 3600 * num_of_days
        else:  # noqa: RET505
            return time_finish + 24 * 3600 * (num_of_days + 1)

    def getShiftLength(self):  # noqa: ANN201, N802, D102
        shift_name = self.shift._shift_name  # noqa: SLF001
        (time_start, time_finish) = self._shifting.getShiftTimes(shift_name)

        if time_start < time_finish:
            return time_finish - time_start
        else:  # noqa: RET505
            return 24 * 3600 - time_start + time_finish

    def setJob(  # noqa: ANN201, N802, D102, PLR0913
        self,
        node_name,  # noqa: ANN001
        action,  # noqa: ANN001
        entity,  # noqa: ANN001
        effect_definition_name,  # noqa: ANN001
        method_name,  # noqa: ANN001
        time_arival,  # noqa: ANN001
        time_done,  # noqa: ANN001
        iOnGoing,  # noqa: ANN001, N803
    ):
        if self.isWorking == True:  # noqa: E712
            raise ValueError('The curent agent is working')  # noqa: EM101, TRY003

        self.isWorking = True
        self.cur_job_location = node_name
        self.cur_job_action = action
        self.cur_job_entity = entity
        self.cur_job_effect_definition_name = effect_definition_name
        self.cur_job_method_name = method_name
        self.cur_job_ongoing = iOnGoing
        self._time_of_arival = time_arival
        self.job_end_time = time_done


class Agents:  # noqa: D101
    def __init__(self, registry, shifting, jobs, restoration_log_book):  # noqa: ANN001, ANN204, D107
        # data:    is the
        # type:    agent type
        # sybtype: agent sub type
        # active:  active is true if it is the shift
        # ready:   read is true if the agent's active and it has nthing to do
        self._agents = pd.DataFrame(
            columns=['data', 'type', 'group', 'active', 'ready', 'available']
        )  # table that includes all data about agents including attributes for fast refinement, the index is the name of the agent (AKA agent_ID)  # noqa: E501
        self.group_names = {}
        self._shifting = shifting
        self._jobs = jobs
        self.restoration_log_book = restoration_log_book
        self.registry = registry

    def addAgent(self, agent_name, agent_type, definition):  # noqa: ANN001, ANN201, N802, D417
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

        """  # noqa: D400, D401, D415
        # number_of_agents = int(definition['Number'])  # noqa: ERA001
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

    def setActiveAgents(self, active_agent_ID_list):  # noqa: ANN001, ANN201, N802, N803
        """Set agents active by a list of agents' ID

        Parameters
        ----------
        active_agent_ID_list : list
            agents ID name

        Returns
        -------
        None.

        """  # noqa: D400, D415
        for active_agent_ID in active_agent_ID_list:  # noqa: N806
            self._agents['active'].loc[active_agent_ID] = True

    def getAgentGroupTagList(self, typed_ready_agent):  # noqa: ANN001, ANN201, N802, D102
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

    def getAllAgentTypes(self):  # noqa: ANN201, N802, D102
        return self._agents['type'].unique().tolist()

    def getAllAgent(self):  # noqa: ANN201, N802
        """Get a copy of all agent dataframe.

        Returns
        -------
        A copy of all agent dataFrame

        """
        return self._agents.copy(deep=True)

    def setChangeShift(self, time, working_check=True):  # noqa: ANN001, ANN201, FBT002, ARG002, N802, D102
        for name, agent in self._agents.iterrows():  # noqa: B007
            if self._agents.loc[name, 'data'].isOnShift(time):
                if (  # noqa: SIM102
                    self._agents.loc[name, 'active'] == False  # noqa: E712
                ):  # if agent is active already and is on shift, it means that the agent has been active before teh shift change event  # noqa: E501
                    if self._agents.loc[name, 'available'] == True:  # noqa: E712
                        self._agents.loc[name, 'active'] = True
                        self._agents.loc[name, 'ready'] = True

            else:
                if (
                    self._agents.loc[name, 'ready'] == True  # noqa: E712
                    and self._agents.loc[name, 'data'].isWorking == True  # noqa: E712
                ):
                    raise RuntimeError(name + ' is working')
                self._agents.loc[name, 'active'] = False
                self._agents.loc[name, 'ready'] = False

    def initializeActiveAgents(self, time):  # noqa: ANN001, ANN201, N802, D102
        for name, agent in self._agents.iterrows():  # noqa: B007
            if self._agents.loc[name, 'data'].isOnShift(time):
                self._agents.loc[name, 'active'] = True
            else:
                self._agents.loc[name, 'active'] = False

    def initializeReadyAgents(self):  # noqa: ANN201, N802, D102
        for name, agent in self._agents.iterrows():  # noqa: B007
            if self._agents.loc[name, 'active'] == True:  # noqa: E712
                self._agents.loc[name, 'ready'] = True
            else:
                self._agents.loc[name, 'ready'] = False

    def getReadyAgents(self):  # noqa: ANN201, N802, D102
        temp = self._agents[
            (self._agents['ready'] == True) & (self._agents['available'] == True)  # noqa: E712
        ]
        check_temp = temp['active'].all()
        if check_temp == False:  # noqa: E712
            print(temp[temp['active'] == False])  # noqa: T201, E712
            raise ValueError('At least one agent is ready although is not on shift')  # noqa: EM101, TRY003

        return temp

    def getAvailabilityRatio(self, agent_type, time):  # noqa: ANN001, ANN201, N802, D102
        if agent_type == 'WQOperator' or agent_type == 'WQWorker':  # noqa: PLR1714
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
        else:  # noqa: RET505
            temp[time] = np.nan
            temp.sort_index(inplace=True)  # noqa: PD002
            temp.interpolate(method='index', inplace=True)  # noqa: PD002
            return temp[time]

    def getDefaultAvailabilityRatio(agent_type, self):  # noqa: ANN001, ANN201, ARG002, N802, N805, D102
        if agent_type == 'WQOperator' or agent_type == 'WQWorker':  # noqa: PLR1714
            return 0
        else:  # noqa: RET505
            return 1

    def assignsJobToAgent(  # noqa: ANN201, C901, N802, D102, PLR0912, PLR0913, PLR0915
        self,
        agent_name,  # noqa: ANN001
        node_name,  # noqa: ANN001
        entity,  # noqa: ANN001
        action,  # noqa: ANN001
        time,  # noqa: ANN001
        wn,  # noqa: ANN001
        reminded_time,  # noqa: ANN001
        number_of_damages,  # noqa: ANN001
        orginal_element,  # noqa: ANN001
    ):
        if self._agents.loc[agent_name, 'active'] != True:  # noqa: E712
            raise ValueError('Agent ' + agent_name + ' is not active')
        if self._agents.loc[agent_name, 'ready'] != True:  # noqa: E712
            raise ValueError('Agent ' + agent_name + ' is not ready')

        if self._agents.loc[agent_name, 'data'].isOnShift(time) != True:  # noqa: E712
            raise ValueError('Agent ' + agent_name + ' is not on shift')

        if self._agents.loc[agent_name, 'data'].isWorking == True:  # noqa: E712
            raise ValueError('Agent ' + agent_name + ' is working')

        # logger.debug('Assiging job to '+agent_name)  # noqa: ERA001
        real_node_name = node_name
        if self._jobs._rm.entity[entity] == 'DISTNODE':  # noqa: SLF001
            damage_data = self._jobs._rm._registry.getDamageData(  # noqa: SLF001
                'DISTNODE', iCopy=False
            )
            if 'virtual_of' in damage_data.columns:
                real_node_name = get_node_name(node_name, damage_data)

        coord = wn.get_node(real_node_name).coordinates
        agent_type = self._agents.loc[agent_name, 'type']

        _ETA = self._agents.loc[agent_name, 'data']._estimateTimeOfArival(coord)  # noqa: SLF001, N806
        effect_definition_name = self._jobs.getEffectDefinitionName(
            agent_type, action, entity
        )
        method_name = self._jobs.chooseMethodForCurrentJob(
            node_name, effect_definition_name, entity
        )

        if method_name == None:  # noqa: E711
            raise ValueError(
                'No method is applicale for ' + repr(effect_definition_name)
            )

        if reminded_time == None:  # noqa: E711
            _ETJ = self._jobs.getAJobEstimate(  # noqa: N806
                orginal_element,
                agent_type,
                entity,
                action,
                method_name,
                number_of_damages,
            )
        else:
            _ETJ = int(reminded_time)  # noqa: N806
            if reminded_time < 0:
                raise ValueError('Something wrong here: ' + repr(reminded_time))

        if effect_definition_name != 'CHECK':
            method_line = self._jobs._effect_data[effect_definition_name][  # noqa: SLF001
                method_name
            ]
        else:
            method_line = [{'EFFECT': 'CHECK'}]

        effects_only = [i['EFFECT'] for i in method_line]

        collective = None
        if 'SKIP' in effects_only:
            return (False, 'SKIP', None, collective)
        elif 'FASTCHECK' in effects_only:  # noqa: RET505
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
            print(  # noqa: T201
                str(_ETA)
                + '  '
                + str(effect_definition_name)
                + '  '
                + str(orginal_element)
            )
            print(str(method_name) + '  ' + str(_ETJ))  # noqa: T201
            raise ValueError('Subzero ETA or sub-equal-zero ETJ')  # noqa: EM101, TRY003

        end_time = time + _ETA + _ETJ
        agent_shift_change_time = self._agents.loc[
            agent_name, 'data'
        ].getAgentShiftEndTime(time)
        shift_length = self._agents.loc[agent_name, 'data'].getShiftLength()

        minimum_job_time = self._jobs._rm._registry.settings['minimum_job_time']  # noqa: SLF001, F841
        if end_time <= agent_shift_change_time:
            iget = 'INSIDE_SHIFT'
            iOnGoing = False  # noqa: N806
        elif (
            end_time > agent_shift_change_time
            and (shift_length - 2 * 3600) < _ETJ
            and (time + _ETA + 2 * 3600) < agent_shift_change_time
        ):
            iget = 'OUTSIDE_SHIFT'
            iOnGoing = True  # noqa: N806
        else:
            # logger.warning(agent_name+',  '+node_name+', '+repr(end_time))  # noqa: ERA001
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

    def getJobEndTime(self, agent_name, icheck=True):  # noqa: ANN001, ANN201, FBT002, N802, D102
        end_time = self._agents.loc[agent_name, 'data'].job_end_time
        if icheck == True and end_time == None:  # noqa: E711, E712
            raise ValueError('No Time is assigned to agent')  # noqa: EM101, TRY003
        if (
            icheck == True  # noqa: E712
            and self._agents.loc[agent_name, 'data'].isWorking == False  # noqa: E712
        ):
            raise ValueError('The agent is not working')  # noqa: EM101, TRY003
        return end_time

    def getJobArivalTime(self, agent_name, icheck=True):  # noqa: ANN001, ANN201, FBT002, N802, D102
        arival_time = self._agents.loc[agent_name, 'data']._time_of_arival  # noqa: SLF001
        if icheck == True and arival_time == None:  # noqa: E711, E712
            raise ValueError('No Time is assigned to agent')  # noqa: EM101, TRY003
        if (
            icheck == True  # noqa: E712
            and self._agents.loc[agent_name, 'data'].isWorking == False  # noqa: E712
        ):
            raise ValueError('The agent is not working')  # noqa: EM101, TRY003
        return arival_time

    def releaseAgent(self, agent_name):  # noqa: ANN001, ANN201, N802, D102
        if self._agents.loc[agent_name, 'ready'] == True:  # noqa: E712
            raise ValueError(agent_name + ' is already ready')
        if self._agents.loc[agent_name, 'active'] != True:  # noqa: E712
            raise ValueError(agent_name + ' is not active')
        if self._agents.loc[agent_name, 'data'].isWorking == False:  # noqa: E712
            raise ValueError(agent_name + ' is not working')

        self._agents.loc[agent_name, 'ready'] = True

        self._agents.loc[agent_name, 'data'].isWorking = False
        self._agents.loc[agent_name, 'data'].cur_job_location = None
        self._agents.loc[agent_name, 'data'].cur_job_action = None
        self._agents.loc[agent_name, 'data'].cur_job_entity = None
        self._agents.loc[agent_name, 'data']._time_of_arival = None  # noqa: SLF001
        self._agents.loc[agent_name, 'data'].cur_job_effect_definition_name = None
        self._agents.loc[agent_name, 'data'].cur_job_method_name = None
        self._agents.loc[agent_name, 'data'].job_end_time = None
        self._agents.loc[agent_name, 'data'].cur_job_ongoing = None


class AgentShift:  # noqa: D101
    def __init__(self, agent_name, name):  # , shifting_obj):  # noqa: ANN001, ANN204, D107
        self._agent_name = agent_name
        self._shift_name = name
        # shifting_obj.addAgentShift(self._agent_name, self._shift_name)  # noqa: ERA001


class Shifting:  # noqa: D101
    def __init__(self):  # noqa: ANN204, D107
        self._all_agent_shift_data = {}
        self._shift_data = pd.DataFrame(columns=['begining', 'end'])

    def addShift(self, name, begining, ending):  # noqa: ANN001, ANN201, N802
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

        """  # noqa: D400, D401, D415
        if name in self._shift_data:
            raise ValueError('Shift name already registered')  # noqa: EM101, TRY003
        if type(begining) != int and type(begining) != float:  # noqa: E721
            raise ValueError('Begining time must be integer: ' + str(type(begining)))
        if type(ending) != int and type(ending) != float:  # noqa: E721
            raise ValueError('Ending time must be integer: ' + str(type(ending)))
        if begining > 24 * 3600:
            raise ValueError('begining time is bigger than 24*3600' + str(begining))
        if ending > 24 * 3600:
            raise ValueError('Ending time is bigger than 24*3600' + str(ending))
        begining = int(begining)
        ending = int(ending)

        self._shift_data.loc[name] = [begining, ending]

    def getShiftTimes(self, name):  # noqa: ANN001, ANN201, N802, D102
        return (
            self._shift_data['begining'].loc[name],
            self._shift_data['end'].loc[name],
        )

    def getNextShiftTime(self, time):  # noqa: ANN001, ANN201, N802, D102
        daily_time = time % (24 * 3600)
        num_of_days = int(time / (24 * 3600))

        next_shift_candidate = pd.Series()

        for shift_name, shift_data in self._shift_data.iterrows():
            beg_time = shift_data[0]
            end_time = shift_data[1]

            if beg_time > end_time and daily_time < end_time:
                beg_time -= 24 * 3600
            elif beg_time > end_time and daily_time >= end_time:
                # beg_time += 24*3600*num_of_days  # noqa: ERA001
                end_time += 24 * 3600

            if daily_time < end_time and daily_time >= beg_time:
                next_shift_candidate.loc[shift_name] = (
                    end_time + 24 * 3600 * num_of_days
                )
        change_shift_time = next_shift_candidate.min()
        # if beg_time > end_time:
        # next_shift_time = time +(change_shift_time - daily_time)  # noqa: ERA001
        # else:  # noqa: ERA001

        return change_shift_time  # noqa: RET504

    def assignShiftToAgent(self, agent_ID, shift_name):  # noqa: ANN001, ANN201, N802, N803
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

        """  # noqa: D400, D401, D415
        if agent_ID in self._all_agent_shift_data:
            raise ValueError('The agent ID currently in Agent ALl Shifts')  # noqa: EM101, TRY003
        if shift_name not in self._shift_data:
            raise ValueError('shift data is not in registered as shifts')  # noqa: EM101, TRY003

        self._all_agent_shift_data[agent_ID] = shift_name


class DispatchRule:  # noqa: D101
    def __init__(self, settings, method='deterministic', exclude=None):  # noqa: ANN001, ANN204, D107
        self.settings = settings
        self._rules = {}
        self._cumulative = {}

        if 'PIPE' not in exclude:
            self._rules['PIPE'] = self.settings['pipe_damage_discovery_model'][
                'time_discovery_ratio'
            ]
            # data2=pd.Series([0.90, 0.01, 0.01, 0.04, 0.04, 0, 0], index = [3600*n for n in [0, 12, 24, 36, 48, 60, 72]])  # noqa: ERA001, E501

        if 'DISTNODE' not in exclude:
            self._rules['DISTNODE'] = self.settings['node_damage_discovery_model'][
                'time_discovery_ratio'
            ]
            # data=pd.Series([0, 0.67, 0.07, 0.07, 0.07, 0.07, 0.05], index = [3600*n for n in [0, 12, 24, 36, 48, 60, 72]])  # noqa: ERA001, E501

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
        # self._rules.pop(key)  # noqa: ERA001

        for key, d in self._rules.items():  # noqa: B007, PERF102
            self._cumulative[key] = self._rules[key].cumsum()

    def getDiscoveredPrecentage(self, time):  # noqa: ANN001, ANN201, N802, D102
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
                temp.sort_index(inplace=True)  # noqa: PD002
                temp.interpolate(method='index', inplace=True)  # noqa: PD002
                res[key] = temp[time]
        return res


class Dispatch:  # noqa: D101
    def __init__(self, restoration, settings, discovery_interval=0, method='old'):  # noqa: ANN001, ANN204, D107
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

        self._rm._registry.addAttrToPipeDamageTable('discovered', False)  # noqa: FBT003, SLF001
        self._rm._registry.addAttrToDistNodeDamageTable('discovered', False)  # noqa: FBT003, SLF001

    def updateDiscovery(self, time):  # noqa: ANN001, ANN201, C901, N802, D102, PLR0912, PLR0915
        if time < self._rm.restoration_start_time:
            print('Time is less than init time')  # noqa: T201

        else:
            # if self.method == 'old':
            # time_since_dispatch_activity = time - self._rm.restoration_start_time  # noqa: ERA001
            # discovered_ratios         = self._rules.getDiscoveredPrecentage(time_since_dispatch_activity)  # noqa: ERA001, E501
            # discovered_damage_numbers = self._getDamageNumbers(discovered_ratios)  # noqa: ERA001
            # self._updateDamagesNumbers(discovered_damage_numbers)  # noqa: ERA001

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

                pipe_damage_table = self._rm._registry._pipe_damage_table  # noqa: SLF001
                not_discovered_pipe_damage_table = pipe_damage_table[
                    pipe_damage_table['discovered'] == False  # noqa: E712
                ]
                to_be_checked_node_list = list(
                    not_discovered_pipe_damage_table.index
                )
                breaks_not_discovered_pipe_damage_table = pipe_damage_table[
                    (pipe_damage_table['discovered'] == False)  # noqa: E712
                    & (pipe_damage_table['damage_type'] == 'break')
                ]
                not_discovered_break_node_B = (  # noqa: N806
                    self._rm._registry._pipe_break_history.loc[  # noqa: SLF001
                        breaks_not_discovered_pipe_damage_table.index, 'Node_B'
                    ]
                )
                not_dicovered_node_B_list = not_discovered_break_node_B.to_list()  # noqa: N806
                to_be_checked_node_list.extend(not_dicovered_node_B_list)
                # break_pair = zip(breaks_not_discovered_pipe_damage_table, not_discovered_break_node_B)  # noqa: ERA001, E501
                # not_discovered_pipe_damage_name_list = list(not_discovered_pipe_damage_table.index)  # noqa: ERA001, E501
                # breaks_not_discovered_pipe_damage_table  # noqa: ERA001
                # all_nodes_name_list = set(self._rm._registry.result.columns)  # noqa: ERA001
                available_nodes = set(
                    self._rm._registry.result.node['demand'].columns  # noqa: SLF001
                )
                to_be_checked_node_list = set(to_be_checked_node_list)
                shared_nodes_name_list = (
                    to_be_checked_node_list.union(available_nodes)
                    - (available_nodes - to_be_checked_node_list)
                    - (to_be_checked_node_list - available_nodes)
                )
                if len(shared_nodes_name_list) > 0:
                    leaking_nodes_result = self._rm._registry.result.node['demand'][  # noqa: SLF001
                        list(shared_nodes_name_list)
                    ]

                    leaking_nodes_result = leaking_nodes_result.loc[
                        (leaking_nodes_result.index > (time - pipe_leak_time_span))
                    ]
                    discovered_bool = leaking_nodes_result >= pipe_leak_criteria
                    discovered_bool_temp = discovered_bool.any()
                    discovered_bool_temp = discovered_bool_temp[
                        discovered_bool_temp == True  # noqa: E712
                    ]
                    to_be_discoverd = discovered_bool_temp.index.to_list()

                    # time1    = leaking_nodes_result.index[1:]  # noqa: ERA001
                    # time2    = leaking_nodes_result.index[0:-1]  # noqa: ERA001
                    # time_dif = (pd.Series(time1) - pd.Series(time2))  # noqa: ERA001
                    # leaking_nodes_result = leaking_nodes_result.drop(leaking_nodes_result.index[-1])  # noqa: ERA001, E501

                    # leaking_nodes_result.index = time_dif.to_numpy()  # noqa: ERA001
                    # leaking_nodes_result = leaking_nodes_result.apply(lambda x: x.values * x.index)  # noqa: ERA001, E501
                    # summed_water_loss = leaking_nodes_result.sum()  # noqa: ERA001
                    # to_be_discoverd = summed_water_loss[summed_water_loss > 3600*2*0.2]  # noqa: ERA001, E501
                    discovery_list = set()
                    # to_be_discoverd = list(to_be_discoverd.index)  # noqa: ERA001
                    for discovery_candidate in to_be_discoverd:
                        if discovery_candidate in not_dicovered_node_B_list:
                            candidate_break_A = not_discovered_break_node_B[  # noqa: N806
                                not_discovered_break_node_B == discovery_candidate
                            ].index[0]
                            discovery_list.add(candidate_break_A)
                        else:
                            discovery_list.add(discovery_candidate)
                    # discovery_list = list(discovery_list)  # noqa: ERA001
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

                nodal_damage_table = self._rm._registry._node_damage_table  # noqa: SLF001
                not_discovered_nodal_damage_table = nodal_damage_table[
                    nodal_damage_table['discovered'] == False  # noqa: E712
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
                    self._rm._registry.result.node['leak'].columns  # noqa: SLF001
                )
                to_be_checked_node_list = set(to_be_checked_node_list)
                shared_nodes_name_list = (
                    to_be_checked_node_list.union(available_leak_nodes)
                    - (available_leak_nodes - to_be_checked_node_list)
                    - (to_be_checked_node_list - available_leak_nodes)
                )
                if len(shared_nodes_name_list) > 0:
                    shared_nodes_name_list = list(shared_nodes_name_list)
                    leaking_nodes_result = self._rm._registry.result.node['leak'][  # noqa: SLF001
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
                        discovered_bool_temp == True  # noqa: E712
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
                # else:  # noqa: ERA001

            time_since_dispatch_activity = time - self._rm.restoration_start_time
            discovered_ratios = self._rules.getDiscoveredPrecentage(
                time_since_dispatch_activity
            )
            discovered_damage_numbers = self._getDamageNumbers(discovered_ratios)
            self._updateDamagesNumbers(discovered_damage_numbers)

            # else:  # noqa: ERA001
            # raise ValueError('Unknown method: '+repr(self.method))  # noqa: ERA001

    def _getDamageNumbers(self, discovered_ratios):  # noqa: ANN001, ANN202, N802
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
                else:  # noqa: RET506
                    discovered_ratios[el] = 1
            temp = len(self._rm._registry.getDamageData(el))  # noqa: SLF001
            num_damaged_entity[el] = int(np.round(temp * discovered_ratios[el]))
        return num_damaged_entity

    def _updateDamagesNumbers(self, discovered_numbers):  # noqa: ANN001, ANN202, N802
        for el in discovered_numbers:
            if self._last_discovered_number[el] > discovered_numbers[el]:
                raise ValueError(
                    'Discovered number is less than what it used to be in element '
                    + el
                )
            elif self._last_discovered_number[el] < discovered_numbers[el]:  # noqa: RET506
                refined_damaged_table = self._rm._registry.getDamageData(el)  # noqa: SLF001
                if len(refined_damaged_table) < discovered_numbers[el]:
                    raise ValueError(
                        'discovered number is bigger than all damages in element'
                        + el
                    )

                discovered_damage_table = refined_damaged_table[
                    refined_damaged_table['discovered'] == True  # noqa: E712
                ]
                if discovered_numbers[el] <= len(discovered_damage_table):
                    continue
                undiscovered_damage_table = refined_damaged_table[
                    refined_damaged_table['discovered'] == False  # noqa: E712
                ]

                # =============================================================================  # noqa: E501
                #                 used_number = []  # noqa: ERA001
                #                 i = 0  # noqa: ERA001
                #                 while i < (discovered_numbers[el] - self._last_discovered_number[el]):  # noqa: E501
                #                     picked_number = random.randint(0,len(undiscovered_damage_table)-1)  # noqa: ERA001, E501
                #                     if picked_number not in used_number:
                #                         used_number.append(picked_number)  # noqa: ERA001
                #                         i += 1  # noqa: ERA001
                #                     else:  # noqa: ERA001
                #                         pass
                # =============================================================================  # noqa: E501
                if len(undiscovered_damage_table) > 0:
                    used_number = random.sample(
                        range(len(undiscovered_damage_table)),
                        discovered_numbers[el] - len(discovered_damage_table),
                    )
                else:
                    used_number = []
                for i in used_number:
                    temp_index = undiscovered_damage_table.index[i]
                    self._rm._registry.updateElementDamageTable(  # noqa: SLF001
                        el, 'discovered', temp_index, True, icheck=True  # noqa: FBT003
                    )

                if el == 'PIPE':
                    refined_damaged_table = self._rm._registry.getDamageData(el)  # noqa: SLF001
                    discovered_damage_table = refined_damaged_table[
                        refined_damaged_table['discovered'] == True  # noqa: E712
                    ]
                self._last_discovered_number[el] = discovered_numbers[el]


class Priority:  # noqa: D101
    def __init__(self, restoration):  # noqa: ANN001, ANN204, D107
        self._data = {}
        self._rm = restoration

    def addData(self, agent_type, priority, order):  # noqa: ANN001, ANN201, N802, D102
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

    def getPriority(self, agent_type, priority):  # noqa: ANN001, ANN201, N802, D102
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

    def getHydSigDamageGroups(self):  # noqa: ANN201, N802, D102
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
                i += 1  # noqa: SIM113
        return damage_group_list

    def sortDamageTable(  # noqa: ANN201, C901, N802, D102, PLR0912, PLR0913, PLR0915
        self,
        wn,  # noqa: ANN001
        entity_data,  # noqa: ANN001
        entity,  # noqa: ANN001
        agent_type,  # noqa: ANN001
        target_priority_index,  # noqa: ANN001
        order_index,  # noqa: ANN001
        target_priority=None,  # noqa: ANN001
    ):
        all_priority_data = self._data[agent_type]
        target_priority_list = all_priority_data.loc[target_priority_index]

        if len(target_priority_list) == 0:
            return entity_data

        name_sugest = 'Priority_' + str(target_priority_index) + '_dist'

        if target_priority == None:  # noqa: E711
            target_priority = target_priority_list[order_index]

        if target_priority == None:  # noqa: E711
            return entity_data

        elif target_priority in self._rm.proximity_points:  # noqa: RET505
            Proximity_list = self._rm.proximity_points[target_priority]  # noqa: N806
            node_name_list = list(entity_data.index)
            for node_name in node_name_list:
                # Sina: you can enhance the run time speed with having x, y coordinates in the damage table and not producing and droping them each time  # noqa: E501
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
            entity_data.sort_values(by=name_sugest, ascending=True, inplace=True)  # noqa: PD002
            columns_to_drop.append(name_sugest)
            columns_to_drop.append('X_COORD')
            columns_to_drop.append('Y_COORD')
            entity_data.drop(columns=columns_to_drop, inplace=True)  # noqa: PD002

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
                all_time_index = self._rm._registry.result.link['flowrate'].index[  # noqa: SLF001
                    : self._rm.restoration_start_time + 1
                ]
                pipe_name_list = entity_data.loc[:, 'Orginal_element']
                last_valid_time = [
                    cur_time
                    for cur_time in all_time_index
                    if cur_time not in self._rm._registry.result.maximum_trial_time  # noqa: SLF001
                ]
                last_valid_time.sort()
                if len(last_valid_time) == 0:
                    last_valid_time = self._rm.restoration_start_time
                else:
                    last_valid_time = last_valid_time[-1]

                name_sugest = 'Priority_' + str(target_priority_index) + '_dist'
                flow_rate = (
                    self._rm._registry.result.link['flowrate']  # noqa: SLF001
                    .loc[last_valid_time, pipe_name_list]
                    .abs()
                )
                entity_data.loc[:, name_sugest] = flow_rate.to_list()
                entity_data.sort_values(name_sugest, ascending=False, inplace=True)  # noqa: PD002
                entity_data.drop(columns=name_sugest, inplace=True)  # noqa: PD002

        elif (
            target_priority in self._rm.proximity_points
            and target_priority != 'WaterSource2'
        ):
            all_node_table = self._rm._registry.all_node_table  # noqa: SLF001
            Proximity_list = self._rm.proximity_points[target_priority]  # noqa: N806
            node_name_list = list(entity_data.index)
            for node_name in node_name_list:
                # Sina: you can enhance the run time speed with having x, y coordinates in the damage table and not producing and droping them each time  # noqa: E501
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
                except:  # noqa: E722
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

                # print("closest_node_name= "+str(closest_node_name))  # noqa: ERA001

                orginal_pipe_name_list = entity_data['Orginal_element']
                damaged_pipe_node_list = [
                    self._rm._registry.undamaged_link_node_list[link_node_names]  # noqa: SLF001
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
                # print(shortest_path_length)  # noqa: ERA001

                name_sug_c = name_sugest + '_' + str(counter)
                columns_to_drop.append(name_sug_c)
                entity_data[name_sug_c] = shortest_path_length
                counter += 1
            dist_only_entity_table = entity_data[columns_to_drop]
            min_dist_entity_table = dist_only_entity_table.min(axis=1)
            entity_data.loc[:, name_sugest] = min_dist_entity_table
            entity_data.sort_values(by=name_sugest, ascending=True, inplace=True)  # noqa: PD002
            columns_to_drop.append(name_sugest)
            columns_to_drop.append('X_COORD')
            columns_to_drop.append('Y_COORD')
            entity_data.drop(columns=columns_to_drop, inplace=True)  # noqa: PD002
            # print(entity_data)  # noqa: ERA001
            # print("+++++++++++++++++++++++++++++++++++++++")  # noqa: ERA001

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
                hyd_sig = self._rm._registry.hydraulic_significance[  # noqa: SLF001
                    entity_data['Orginal_element']
                ]

                entity_data.loc[:, name_sugest] = hyd_sig.to_list()
                entity_data.sort_values(name_sugest, ascending=False, inplace=True)  # noqa: PD002
                entity_data.drop(columns=name_sugest, inplace=True)  # noqa: PD002

        # If element type is not leakable, it does nothing. IF nodes are not
        # Checked (i.e. check is not at the sequnce before the current action)
        # the leak data is real time leak for the damage location.
        elif target_priority.upper() == 'MOSTLEAKATCHECK':
            # real_node_name_list = []  # noqa: ERA001
            node_name_list = list(entity_data.index)
            name_sugest = (
                'Priority_' + str(target_priority_index) + '_leak_sina'
            )  # added sina so the possibility of a conflic of name is minimized
            # for node_name in node_name_list:
            # node_name_vir = get_node_name(node_name, entity_data)  # noqa: ERA001
            # real_node_name_list.append(node_name_vir)  # noqa: ERA001
            element_type = self._rm.entity[entity]
            leak_data = self._rm._registry.getMostLeakAtCheck(  # noqa: SLF001
                node_name_list, element_type
            )
            if leak_data is not None:
                entity_data.loc[node_name_list, name_sugest] = leak_data
                entity_data.sort_values(by=name_sugest, ascending=True, inplace=True)  # noqa: PD002
                entity_data.drop(columns=[name_sugest], inplace=True)  # noqa: PD002
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

    def isAgentTypeInPriorityData(self, agent_type):  # noqa: ANN001, ANN201, N802, D102
        return agent_type in self._data


class Jobs:  # noqa: D101
    def __init__(self, restoration):  # noqa: ANN001, ANN204, D107
        self._rm = restoration
        self._job_list = pd.DataFrame(
            columns=['agent_type', 'entity', 'action', 'time_argument']
        )
        self._effect_defualts = {}  # pd.DataFrame(columns=['effect_definition_name', 'method_name','argument','value'])  # noqa: E501
        self._effect_data = {}
        self._time_overwrite = {}
        self._final_method = {}
        self._once = {}

    def addEffect(self, effect_name, method_name, def_data):  # noqa: ANN001, ANN201, N802, D102
        if effect_name not in self._effect_data:
            self._effect_data[effect_name] = None

        if self._effect_data[effect_name] != None:  # noqa: SIM102, E711
            if method_name in self._effect_data[effect_name]:
                raise ValueError(
                    'Dupplicate method_name is given. Effect name: '
                    + str(effect_name)
                    + ', '
                    + str(method_name)
                )

        if self._effect_data[effect_name] == None:  # noqa: E711
            temp = {}
            temp[method_name] = def_data
            self._effect_data[effect_name] = temp
        else:
            self._effect_data[effect_name][method_name] = def_data

    def setJob(self, jobs_definition):  # noqa: ANN001, ANN201, N802, D102
        self._job_list = pd.DataFrame.from_records(jobs_definition)

    def _filter(self, agent_type, entity, action):  # noqa: ANN001, ANN202
        temp = self._job_list
        temp = temp[
            (
                temp[['agent_type', 'entity', 'action']]
                == [agent_type, entity, action]
            ).all(1)
        ]
        temp_length = len(temp)
        if temp_length > 1:
            raise ValueError('We have more than one job description')  # noqa: EM101, TRY003
        elif temp_length == 0:  # noqa: RET506
            raise ValueError(
                'We have Zero one job description for agent type= '
                + repr(agent_type)
                + ', entity= '
                + repr(entity)
                + ', action= '
                + repr(action)
            )
        return temp

    def getAJobEstimate(  # noqa: ANN201, N802, D102, PLR0913
        self, orginal_element, agent_type, entity, action, method_name, number  # noqa: ANN001
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
                raise ValueError('Unknown Time Data')  # noqa: EM101, TRY003
        time = int(time_arg)
        # try:  # noqa: ERA001
        # time_arg = int(time_arg):
        # time = time_arg  # noqa: ERA001
        # except:  # noqa: ERA001
        # raise ValueError('Unknow time argument: '+str(type(time_arg)))  # noqa: ERA001

        once_flag = False
        if operation_name in self._once:  # noqa: SIM102
            if method_name in self._once[operation_name]:
                once_flag = True

        if once_flag == False:  # noqa: E712
            time = int(time * number)

        # IMPORTANT/sina
        if (method_name == 2 or method_name == 1) and action == 'reroute':  # noqa: PLR1714, PLR2004
            pass

        return time

    def getMeanJobTime(self, agent_type, entity, action):  # noqa: ANN001, ANN201, N802, D102
        temp = self._filter(agent_type, entity, action)
        time_arg = temp['time_argument'].iloc[0]
        if type(time_arg) == int:  # noqa: E721
            time = time_arg
        else:
            raise ValueError('Unknow time argument: ' + str(type(time_arg)))
        return time

    def getAllEffectByJobData(  # noqa: ANN201, N802, D102
        self, agent_type, action, entity, iWithout_data=True, iOnlyData=False  # noqa: ANN001, FBT002, ARG002, N803
    ):
        temp = self._filter(agent_type, entity, action)
        all_effect_name = temp['effect'].iloc[0]  # noqa: F841

        if iOnlyData == True:  # noqa: E712
            return

    def addEffectDefaultValue(self, input_dict):  # noqa: ANN001, ANN201, N802, D102
        _key = (
            input_dict['effect_definition_name'],
            input_dict['method_name'],
            input_dict['argument'],
        )

        if _key in self._effect_defualts:
            raise ValueError(
                'Duplicate effects definition: {0}, {1}, {2}'.format(  # noqa: EM103, UP030
                    repr(input_dict['effect_definition_name']),
                    repr(input_dict['method_name']),
                    repr(input_dict['argument']),
                )
            )

        self._effect_defualts[_key] = input_dict[
            'value'
        ]  # self._effect_defualts.append(temp_s, ignore_index=True)

    def getEffectsList(self, effect_definition_name, method_name):  # noqa: ANN001, ANN201, N802, D102
        if effect_definition_name == None:  # noqa: E711
            return []

        if effect_definition_name == 'CHECK':
            return [{'EFFECT': 'CHECK'}]
        all_methods = self._effect_data[effect_definition_name]
        effects_list = all_methods[method_name]
        return effects_list  # noqa: RET504

    def getEffectDefinition(self, effect_definition_name, iWithout_data=True):  # noqa: ANN001, ANN201, FBT002, N802, N803, D102
        all_methods = self._effect_data[effect_definition_name]

        if iWithout_data == True and 'DATA' in all_methods:  # noqa: E712
            all_methods = copy.deepcopy(all_methods)
            all_methods.pop('DATA')

        return all_methods

    def getEffectDefinitionName(self, agent_type, action, entity):  # noqa: ANN001, ANN201, N802, D102
        temp = self._filter(agent_type, entity, action)
        effects_definition_name = temp['effect'].iloc[0]
        return effects_definition_name  # noqa: RET504

    def chooseMethodForCurrentJob(self, node_name, effects_definition_name, entity):  # noqa: ANN001, ANN201, N802, D102
        returned_method = None
        if effects_definition_name == None:  # noqa: E711
            return None
        elif (  # noqa: RET505
            effects_definition_name == 'CHECK'  # noqa: PLR1714
            or effects_definition_name == 'FASTCHECK'
            or effects_definition_name == 'SKIP'
        ):
            return effects_definition_name
        else:
            effects_definition = self.getEffectDefinition(
                effects_definition_name
            )  # self._effect_data[effects_definition_name]
            for method_name, effect_list in effects_definition.items():  # noqa: B007, PERF102
                prob_applicability = self.iEffectApplicableByProbability(
                    effects_definition_name, method_name, node_name, entity
                )
                condition_applicability = self.iEffectApplicableByOtherConditions(
                    effects_definition_name, method_name, node_name, entity
                )
                if prob_applicability and condition_applicability:
                    returned_method = method_name
                    break

        if returned_method == None:  # noqa: E711
            try:  # noqa: SIM105
                returned_method = self._final_method[effects_definition_name]
            except:  # noqa: S110, E722
                pass
        return returned_method

    def _getProbability(self, method, iCondition, element_type):  # noqa: ANN001, ANN202, ARG002, N802, N803
        if iCondition == True:  # noqa: E712
            if 'METHOD_PROBABILITY' in method:  # noqa: SIM401
                probability = method['METHOD_PROBABILITY']
            else:
                probability = 1  # noqa: F841
        # else:  # noqa: ERA001
        # if 'METHOD_PROBABILITY' in method:

    def _iConditionHolds(self, val1, con, val2):  # noqa: ANN001, ANN202, C901, N802, PLR0911, PLR0912
        if con == 'BG':
            if val1 > val2:  # noqa: SIM103
                return True
            else:  # noqa: RET505
                return False
        elif con == 'BG-EQ':
            if val1 >= val2:  # noqa: SIM103
                return True
            else:  # noqa: RET505
                return False
        elif con == 'LT':
            if val1 < val2:  # noqa: SIM103
                return True
            else:  # noqa: RET505
                return False
        elif con == 'LT-IF':
            if val1 <= val2:  # noqa: SIM103
                return True
            else:  # noqa: RET505
                return False
        elif con == 'EQ':
            if val1 == val2:  # noqa: SIM103
                return True
            else:  # noqa: RET505
                return False
        else:
            raise ValueError('Unrecognized condition: ' + repr(con))

    def getDefualtValue(self, effects_definition_name, method_name, argument):  # noqa: ANN001, ANN201, N802, D102
        _default = self._effect_defualts
        value = _default.get((effects_definition_name, method_name, argument), None)

        return value  # noqa: RET504

    def iEffectApplicableByOtherConditions(  # noqa: ANN201, N802, D102
        self, effects_definition_name, method_name, damaged_node_name, entity  # noqa: ANN001
    ):
        element_type = self._rm.entity[entity]
        effects_definition = self.getEffectDefinition(effects_definition_name)
        if element_type == 'DISTNODE':
            for single_effect in effects_definition[method_name]:
                if 'PIDR' in single_effect:
                    condition = single_effect['PIDR']
                    _con = condition[0]
                    _con_val = condition[1]
                    _PIDR_type = self.getDefualtValue(  # noqa: N806
                        effects_definition_name, method_name, 'PIDR_TYPE'
                    )
                    if _PIDR_type == None or _PIDR_type == 'ASSIGNED_DEMAND':  # noqa: E711, PLR1714
                        old_demand = self._rm._registry._node_damage_table.loc[  # noqa: SLF001
                            damaged_node_name, 'Demand1'
                        ]
                        new_demand = self._rm._registry._node_damage_table.loc[  # noqa: SLF001
                            damaged_node_name, 'Demand2'
                        ]
                    else:
                        raise ValueError('unrecognized Setting: ' + _PIDR_type)

                    _PIDR = new_demand / old_demand  # noqa: N806

                    iHold = self._iConditionHolds(_PIDR, _con, _con_val)  # noqa: N806

                    return iHold  # noqa: RET504

        return True

    def iEffectApplicableByProbability(  # noqa: ANN201, N802, D102
        self, effects_definition_name, method_name, damaged_node_name, entity  # noqa: ANN001
    ):
        _prob = 0
        temp = self.getDefualtValue(
            effects_definition_name, method_name, 'METHOD_PROBABILITY'
        )
        if temp != None:  # noqa: E711
            _prob = temp
        try:
            self._check_probability(_prob)
        except Exception as e:  # noqa: BLE001
            print(  # noqa: T201
                'in Method bsaed Probability of method '
                + str(method_name)
                + ', and definition_name '
                + str(effects_definition_name)
                + ', :'
                + str(_prob)
            )
            raise ValueError(e)  # noqa: B904

        # =============================================================================  # noqa: E501
        #         if 'DEFAULT' in self._effect_data[effects_definition_name]:
        #             data = self._effect_data[effects_definition_name]['DEFAULT']  # noqa: ERA001
        #             if 'METHOD_PROBABILITY' in data:
        #                 if method_name in data['METHOD_PROBABILITY']:
        #                     _prob=data['METHOD_PROBABILITY'][method_name]  # noqa: ERA001
        #                     try:  # noqa: ERA001
        #                         _check_probability(_prob)  # noqa: ERA001
        #                     except Exception as e:  # noqa: ERA001
        #                         print('in Method bsaed Probability of method ' +method_name+ ', and definition_name '+effects_definition_name)  # noqa: ERA001, E501
        #                         raise ValueError(e)  # noqa: ERA001
        # =============================================================================  # noqa: E501

        if 'DATA' in self._effect_data[effects_definition_name]:
            data = self._effect_data[effects_definition_name]['DATA']
            if 'METHOD_PROBABILITY' in data.columns:
                element_name = self._rm._registry.getOrginalElement(  # noqa: SLF001
                    damaged_node_name, self._rm.entity[entity]
                )

                # temp =data[(data[['ELEMENT_NAME','METHOD_NAME']]==[element_name, method_name]).all(1)]  # noqa: ERA001, E501
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
                    except Exception as e:  # noqa: BLE001
                        print(  # noqa: T201
                            'in LIST of method '
                            + method_name
                            + ', and definition_name '
                            + effects_definition_name
                        )
                        raise ValueError(e)  # noqa: B904

        _rand = random.random()  # noqa: S311
        # if effects_definition_name == 'MJTRreroute':
        # print(str(method_name) + ' - ' + repr(_prob))  # noqa: ERA001
        logger.debug(_prob)
        if _rand < _prob:  # noqa: SIM103
            return True
        return False

    def _check_probability(self, _prob):  # noqa: ANN001, ANN202
        mes = None  # noqa: F841
        _prob = float(_prob)
        if _prob < 0:
            raise ValueError('probability cannot be less than 0.')  # noqa: EM101, TRY003
        elif _prob > 1:  # noqa: RET506
            res = False  # noqa: F841
            raise ValueError('probability cannot be more than 1.')  # noqa: EM101, TRY003


# =============================================================================
# class Effects():
#     def __init__(self, restoration_model):
#         #self._data_table = pd.DataFrame(columns=['effect', 'method_name', 'data_index'])  # noqa: ERA001, E501
#
#
#
#
#
#         #self._data_table.loc[effect_name, 'method'] = method  # noqa: ERA001
#         #self._data_table.loc[effect_name, 'effect'] = effect  # noqa: ERA001
#         #self._data_table.loc[effect_name, 'connection'] = connection  # noqa: ERA001
#         #self._data_table.loc[effect_name, 'connection_value'] = connection_value  # noqa: ERA001
#         #self._data_table.loc[effect_name, 'CV'] = cv  # noqa: ERA001
# =============================================================================
