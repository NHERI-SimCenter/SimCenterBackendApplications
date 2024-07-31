# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 05:09:25 2020

@author: snaeimi
"""

import logging

# import warnings
import restoration.io as rio
import restoration.base as base
import pandas as pd
import numpy as np
import copy
import random
from restoration.base import get_node_name
from repair import Repair
from collections import OrderedDict

logger = logging.getLogger(__name__)


class Restoration:
    def __init__(self, conifg_file_name, registry, damage):
        self.ELEMENTS = ['PIPE', 'DISTNODE', 'GNODE', 'TANK', 'PUMP', 'RESERVOIR']
        self._CONDITIONS = ['EQ', 'BG', 'LT', 'BG-EQ', 'LT-EQ', 'NOTEQ']
        self.reserved_priority_names = [
            'CLOSEST',
            'MOSTLEAKATCHECK',
            'HYDSIG',
            'HYDSIGLASTFLOW',
        ]
        self._hard_event_table = pd.DataFrame(columns=['Requester', 'New', 'Detail'])
        self._reminder_time_hard_event = {}
        self.shifting = base.Shifting()
        self.jobs = base.Jobs(self)
        self.agents = base.Agents(
            registry, self.shifting, self.jobs, registry.restoration_log_book
        )
        self.proximity_points = {}
        self.priority = base.Priority(self)
        self.repair = Repair(registry)
        self.eq_time = None
        self.restoration_start_time = None
        self.earthquake = None
        self.if_initiated = False
        self.sequence = {}
        self.entity = {}
        self.entity_rule = {}
        self.group = {}
        self.pump_restoration = pd.DataFrame()
        self._damage = damage
        # self.temp =[]

        for el in self.ELEMENTS:
            self.group[el] = OrderedDict()

        self._registry = registry
        self.dispatch = base.Dispatch(self, registry.settings, method='new')

        rio.RestorationIO(self, conifg_file_name)
        retoration_data = {}
        retoration_data['sequence'] = self.sequence
        retoration_data['entity'] = self.entity
        retoration_data['group'] = self.group
        registry.retoration_data = retoration_data

        self.ApplyOverrides()

    def ApplyOverrides(self):
        overrides = self._registry.settings.overrides

        if 'POINTS' in overrides:
            points_overrides = overrides['POINTS']
            for point_group_name in points_overrides:
                if point_group_name not in self.proximity_points:
                    logger.warning(
                        'CAUTION!'
                        + '\n'
                        + 'Override Point Group '
                        + repr(point_group_name)
                        + ' is not a defined point group in the restoration plan.'
                    )
                self.proximity_points[point_group_name] = points_overrides[
                    point_group_name
                ]

    def perform_action(self, wn, stop_time):
        logger.debug(stop_time)

        # checks if the restoration is started
        if self.eq_time == None or self.restoration_start_time == None:
            raise ValueError('restoration is not initiated')

        # checks if the stop time is a hard event
        if not self._isHardEvent(stop_time):
            raise RuntimeError('stop time is not a hard event')

        # gets the latest damage revealed and reported to the damage board registery
        self.dispatch.updateDiscovery(stop_time)

        if self._isHardEvent(stop_time, 'pump'):
            pump_list = (
                self.pump_restoration[
                    self.pump_restoration['Restore_time'] == stop_time
                ]
            )['Pump_ID'].tolist()

            # logger.warning(pump_list)
            self.repair.restorePumps(pump_list, wn)

        if self._isHardEvent(stop_time, 'tank'):
            tank_list = (
                self.tank_restoration[
                    self.tank_restoration['Restore_time'] == stop_time
                ]
            )['Tank_ID'].tolist()

            # logger.warning(tank_list)
            self.repair.restoreTanks(tank_list, wn)

        if self._isHardEvent(stop_time, 'agent'):
            ##logger.debug('INSIDE RELEASE')
            released_agents = self.getHardEventDetails(stop_time, 'agent')

            logger.warning('-----------------')

            for r_agent in released_agents:
                agent_type = self.agents._agents.loc[r_agent, 'type']
                action = self.agents._agents.loc[r_agent, 'data'].cur_job_action
                entity = self.agents._agents.loc[r_agent, 'data'].cur_job_entity
                effect_definition_name = self.agents._agents.loc[
                    r_agent, 'data'
                ].cur_job_effect_definition_name
                method_name = self.agents._agents.loc[
                    r_agent, 'data'
                ].cur_job_method_name
                damaged_node_name = self.agents._agents.loc[
                    r_agent, 'data'
                ].cur_job_location
                iOngoing = self.agents._agents.loc[r_agent, 'data'].cur_job_ongoing
                element_type = self.entity[entity]

                effects_list = self.jobs.getEffectsList(
                    effect_definition_name, method_name
                )

                if iOngoing == False:
                    # This must be before apply effect because if not, bypass pipe will not be removed/Sina
                    _damage_data = self._registry.getDamageData(
                        element_type, iCopy=False
                    )
                    if (
                        self.entity[entity] == 'PIPE'
                        or self.entity[entity] == 'DISTNODE'
                    ):
                        orginal_name = _damage_data.loc[
                            damaged_node_name, 'Orginal_element'
                        ]

                        collective_damage_data = _damage_data[
                            _damage_data['Orginal_element'] == orginal_name
                        ]
                        collective_damage_data = collective_damage_data[
                            collective_damage_data[action] == 'Collective'
                        ]
                        collective_damage_data_name_list = (
                            collective_damage_data.index.to_list()
                        )
                        if len(collective_damage_data_name_list) > 0:
                            next_action = self.getNextSequence(element_type, action)
                            if next_action != None:
                                self._registry.setDamageDataByRowAndColumn(
                                    element_type,
                                    collective_damage_data_name_list,
                                    next_action,
                                    False,
                                )
                                self._registry.setDamageDataByRowAndColumn(
                                    element_type,
                                    collective_damage_data_name_list,
                                    'discovered',
                                    True,
                                )

                    self._registry.updateElementDamageTable(
                        element_type, action, damaged_node_name, True
                    )
                    for single_effect in effects_list:
                        self.applyEffect(
                            damaged_node_name,
                            single_effect,
                            element_type,
                            wn,
                            action,
                            stop_time,
                        )

                    next_action = self.getNextSequence(element_type, action)

                    if next_action != None:
                        if (
                            type(_damage_data.loc[damaged_node_name, next_action])
                            == str
                        ):
                            pass
                        elif np.isnan(
                            _damage_data.loc[damaged_node_name, next_action]
                        ):
                            self._registry.updateElementDamageTable(
                                element_type,
                                next_action,
                                damaged_node_name,
                                False,
                                icheck=True,
                            )
                else:
                    self._registry.assignAgenttoLongJob(
                        damaged_node_name, action, entity, None
                    )  # potential bug... When there is a long job availible but not a suitable agent to take crae of teh jiob, teh job will be forgotten

                self.agents.releaseAgent(r_agent)

        # checks for shift change and if the stop time is a shift change, changes the shift and update agent data accordingly
        self.updateShifiting(stop_time)
        self.updateAvailability(stop_time)

        # gets list of ready agents, (on shift and idle)
        ready_agent = self.agents.getReadyAgents()

        ready_agent_types = ready_agent['type'].unique()

        # for each agent type, we get the priority data (entity and action), refine damage data from entity that are waiting for action (action = False)
        for agent_type in ready_agent_types:
            typed_ready_agent = ready_agent[ready_agent['type'] == agent_type]
            typed_ready_agent._is_copy = None

            if not len(typed_ready_agent) > 0:
                continue

            agent_prime_priority_list = self.priority.getPriority(agent_type, 1)
            agent_group_tag_list, agent_group_name = (
                self.agents.getAgentGroupTagList(typed_ready_agent)
            )

            non_tagged_typed_ready_agent = typed_ready_agent.copy()
            non_tagged_typed_ready_agent._is_copy = None
            for agent_group_tag in agent_group_tag_list:
                typed_ready_agent = non_tagged_typed_ready_agent[
                    non_tagged_typed_ready_agent['group'] == agent_group_tag
                ]
                typed_ready_agent._is_copy = None
                order_counter = -1
                for prime_priority in agent_prime_priority_list:
                    order_counter += 1
                    action = list(prime_priority)[0]
                    entity = list(prime_priority)[1]
                    damage_data = self._registry.getDamageData(self.entity[entity])
                    entity_data = self.refineEntityDamageTable(
                        damage_data,
                        agent_group_name,
                        agent_group_tag,
                        self.entity[entity],
                    )
                    if len(entity_data) == 0:
                        continue
                    entity_data = entity_data[(entity_data['discovered'] == True)]
                    entity_data = entity_data[(entity_data[entity] == True)]
                    entity_data = entity_data[(entity_data[action] == False)]

                    logger.warning(
                        'action='
                        + action
                        + ', entity='
                        + entity
                        + ', len(entity_data)='
                        + repr(len(entity_data))
                        + ',  OC= '
                        + repr(order_counter)
                    )
                    for previous_action in self.sequence[self.entity[entity]]:
                        if previous_action == action:
                            break
                        entity_data = entity_data[
                            (entity_data[previous_action] != False)
                        ]

                    vacant_job_list = self._registry.getVacantOnGoingJobs(
                        action, entity
                    )

                    if len(vacant_job_list) > 0 and len(typed_ready_agent) > 0:
                        self.assignVacantJob(
                            vacant_job_list,
                            typed_ready_agent,
                            entity_data,
                            agent_type,
                            action,
                            entity,
                            stop_time,
                            order_counter,
                            wn,
                        )

                    res = self.perform_action_helper(
                        typed_ready_agent,
                        entity_data,
                        agent_type,
                        action,
                        entity,
                        stop_time,
                        order_counter,
                        wn,
                    )

                    if res == 'break':
                        break
                    elif res == 'continue':
                        continue

        new_events = self.getNewEventsTime(reset=True)
        self._registry.restoration_log_book.updateAgentLogBook(
            self.agents._agents, stop_time
        )
        self._registry.restoration_log_book.updateAgentHistory(
            self.agents._agents, stop_time
        )

        return new_events

    def perform_action_helper(
        self,
        typed_ready_agent,
        entity_data,
        agent_type,
        action,
        entity,
        stop_time,
        order_counter,
        wn,
        flag=False,
    ):
        ignore_list = []
        if len(entity_data) == 0:
            if flag == True:
                raise RuntimeError(
                    'Ongoing and zero-length emtity data does must never appen together.'
                )
            return 'continue'
        entity_data = self.priority.sortDamageTable(
            wn, entity_data, entity, agent_type, 2, order_counter
        )  # sort according to the possible secondary priority

        for node_name, damage_data in entity_data.iterrows():
            if not len(typed_ready_agent) > 0:
                break

            # if damage_data[action]!=False or node_name in ignore_list:
            if (
                node_name in ignore_list
            ):  # if this condition is not here there will be a problem regarding same pipe damages/Sina
                continue

            number_of_damages = damage_data['Number_of_damages']

            # mean_time_of_job = self.jobs.getMeanJobTime(agent_type, entity, action)
            # if not typed_ready_agent['data'].iloc[0].isOnShift(stop_time + mean_time_of_job + 900):
            # logger.debug('BREAK due to TIME at '+str(stop_time))
            # break

            distnace_agent_entity = pd.Series(
                index=typed_ready_agent.index.tolist(),
                data=typed_ready_agent.index.tolist(),
            )
            node_name_vir = get_node_name(node_name, entity_data)
            coord = wn.get_node(node_name_vir).coordinates

            distnace_agent_entity.apply(
                lambda x: typed_ready_agent.loc[x, 'data'].getDistanceFromCoordinate(
                    coord
                )
            )

            ##---------------------------------
            ##for agent_name, d_agent in typed_ready_agent.iterrows():
            ##distnace_agent_entity.loc[agent_name] = d_agent['data'].getDistanceFromCoordinate(coord)
            ##---------------------------------

            distnace_agent_entity.sort_values(ascending=True, inplace=True)
            if self.entity[entity] == 'PIPE':
                orginal_element = entity_data.loc[node_name, 'Orginal_element']
            else:
                orginal_element = node_name
            # -----------------------------------------------------------
            while len(distnace_agent_entity) > 0:
                choosed_agent_name = distnace_agent_entity.index[0]

                if flag == False:
                    i_assigned, description, job_gross_time, collective = (
                        self.agents.assignsJobToAgent(
                            choosed_agent_name,
                            node_name,
                            entity,
                            action,
                            stop_time,
                            wn,
                            None,
                            number_of_damages,
                            orginal_element,
                        )
                    )
                else:
                    reminded_time = self._registry.getLongJobRemindedTime(
                        node_name, action, entity
                    )
                    i_assigned, description, job_gross_time, collective = (
                        self.agents.assignsJobToAgent(
                            choosed_agent_name,
                            node_name,
                            entity,
                            action,
                            stop_time,
                            wn,
                            reminded_time,
                            None,
                            None,
                        )
                    )
                    collective = None  # Collective already assigned/Sina
                if i_assigned == False and description == 'ShortOfTime':
                    distnace_agent_entity.pop(distnace_agent_entity.index[0])
                    break

                elif i_assigned == False and description == 'FASTCHECK':
                    self._registry.updateElementDamageTable(
                        self.entity[entity], action, node_name, 'NA', icheck=True
                    )
                    next_action = self.getNextSequence(self.entity[entity], action)
                    if next_action != None:
                        self._registry.updateElementDamageTable(
                            self.entity[entity],
                            next_action,
                            node_name,
                            False,
                            icheck=True,
                        )
                    break

                elif i_assigned == False and description == 'SKIP':
                    break

                elif i_assigned == True:
                    if collective != None:
                        orginal_element = entity_data.loc[
                            node_name, 'Orginal_element'
                        ]
                        entity_same_element_damage_index = (
                            entity_data[
                                entity_data['Orginal_element'] == orginal_element
                            ]
                        ).index.to_list()

                        same_element_damage_data = self._registry.getDamageData(
                            self.entity[entity], iCopy=False
                        )
                        same_element_damage_data = same_element_damage_data[
                            same_element_damage_data['Orginal_element']
                            == orginal_element
                        ]

                        same_element_damage_index = (
                            same_element_damage_data.index.to_list()
                        )

                        same_element_damage_index.remove(node_name)
                        entity_same_element_damage_index.remove(node_name)

                        ignore_list.extend(same_element_damage_index)

                        _damage_data = self._registry.getDamageData(
                            self.entity[entity], iCopy=False
                        )

                        if (
                            _damage_data.loc[same_element_damage_index, action]
                            == 'Collective'
                        ).any():
                            same_element_damage_index.append(node_name)
                            raise ValueError('Hell to the naw' + repr(node_name))

                        _damage_data.loc[same_element_damage_index, action] = (
                            collective  # For times later
                        )

                        entity_data.loc[entity_same_element_damage_index, action] = (
                            'Collective'
                        )

                        self._registry.setDamageDataByRowAndColumn(
                            self.entity[entity],
                            same_element_damage_index,
                            action,
                            'Collective',
                        )
                        # tt=self._registry.getDamageData(self.entity[entity], iCopy=False)

                    self._registry.updateElementDamageTable(
                        self.entity[entity],
                        action,
                        node_name,
                        'On_Going',
                        icheck=not flag,
                    )
                    typed_ready_agent.drop(choosed_agent_name, inplace=True)
                    job_end_time = self.agents.getJobEndTime(choosed_agent_name)

                    if job_end_time != None and description == 'INSIDE_SHIFT':
                        modfied_end_time = self._addHardEvent(
                            job_end_time, 'agent', choosed_agent_name, stop_time
                        )
                        self._registry.restoration_log_book.addEndTimegentActionToLogBook(
                            choosed_agent_name, stop_time, modfied_end_time
                        )

                        if (
                            self._registry.isThereSuchOngoingLongJob(
                                node_name, action, entity
                            )
                            == True
                        ):
                            arival_time = self.agents.getJobArivalTime(
                                choosed_agent_name
                            )
                            self._registry.deductLongJobTime(
                                node_name, action, entity, job_end_time - arival_time
                            )
                            self._registry.removeLongJob(node_name, action, entity)

                        break

                    elif description == 'OUTSIDE_SHIFT':
                        # logger.warning('cur_time= '+repr(stop_time)+',   end_time= '+repr(stop_time+job_gross_time))
                        if not self._registry.isThereSuchOngoingLongJob(
                            node_name, action, entity
                        ):
                            self._registry.addLongJob(
                                node_name,
                                action,
                                entity,
                                job_gross_time,
                                choosed_agent_name,
                            )
                        else:
                            self._registry.assignAgenttoLongJob(
                                node_name, action, entity, choosed_agent_name
                            )

                        end_shift_time = self.agents._agents.loc[
                            choosed_agent_name, 'data'
                        ].getAgentShiftEndTime(stop_time)

                        arival_time = self.agents.getJobArivalTime(
                            choosed_agent_name
                        )
                        self._registry.deductLongJobTime(
                            node_name, action, entity, end_shift_time - arival_time
                        )
                        modfied_end_time = self._addHardEvent(
                            end_shift_time, 'agent', choosed_agent_name, stop_time
                        )
                        self._registry.restoration_log_book.addEndTimegentActionToLogBook(
                            choosed_agent_name, stop_time, modfied_end_time
                        )

                        break
                    else:
                        if job_end_time == None:
                            raise ValueError('Job is not assigned to the agent')
                        else:
                            raise ValueError('Unknown description: ' + description)
                else:
                    raise RuntimeError('i_assigned not boolean')

            # -----------------------------------------------------------
        # self._registry.updatePipeDamageTableTimeSeries(stop_time)

    def assignVacantJob(
        self,
        vacant_job_list,
        typed_ready_agent,
        entity_data,
        agent_type,
        action,
        entity,
        stop_time,
        order_counter,
        wn,
    ):
        if not len(typed_ready_agent) > 0:
            raise RuntimeError(
                'This should not happen. We have a condition before in performe action'
            )
        if not len(vacant_job_list) > 0:
            return
        damage_data = self._registry.getDamageData(self.entity[entity])
        entity_data = pd.DataFrame(
            columns=damage_data.columns, index=vacant_job_list
        )

        entity_data = entity_data.apply(lambda x: damage_data.loc[x.name], axis=1)
        self.perform_action_helper(
            typed_ready_agent,
            entity_data,
            agent_type,
            action,
            entity,
            stop_time,
            order_counter,
            wn,
            flag=True,
        )

    def applyEffect(
        self,
        damage_node_name,
        single_effect_data,
        element_type,
        wn,
        action,
        stop_time,
    ):
        effect_type = single_effect_data['EFFECT']
        damage_data = self._registry.getDamageData(element_type, iCopy=False)
        node_damage_data = damage_data.loc[damage_node_name]
        damage_type = None

        if element_type == 'PIPE':
            damage_type = damage_data.loc[damage_node_name, 'damage_type']

        if effect_type == 'CHECK':
            if element_type == 'DISTNODE':
                result = self._registry.result.node
                # damage_table = self._registry.getDamageData('DISTNODE', iCopy=False)
                real_node_name = get_node_name(damage_node_name, damage_data)
                if real_node_name in result['leak'].columns:
                    leak_demand = result['leak'].loc[stop_time, real_node_name]
                else:
                    leak_demand = 0
                real_demand = result['demand'].loc[stop_time, real_node_name]
                total_demand = leak_demand + real_demand

                node = wn.get_node(real_node_name)
                pattern_list = node.demand_timeseries_list.pattern_list()
                default_pattern = wn.options.hydraulic.pattern
                node_pattern_name = None
                if pattern_list[0] != None:
                    node_pattern_name = pattern_list[0].name
                elif pattern_list[0] == None and default_pattern != None:
                    node_pattern_name = str(default_pattern)

                if node_pattern_name == None:
                    multiplier = 1
                else:
                    cur_pattern = wn.get_pattern(node_pattern_name)
                    multiplier = cur_pattern.at(stop_time)

                base_demand = node.base_demand
                required_demand = multiplier * base_demand
                if 'virtual_of' in damage_data.columns:
                    vir_nodal_damage_list = damage_data[
                        damage_data['virtual_of'] == real_node_name
                    ]
                    vir_nodal_damage_list = vir_nodal_damage_list.index
                    if (
                        damage_data.loc[vir_nodal_damage_list, 'Demand1']
                        .isna()
                        .any()
                    ):
                        self._registry.addNodalDemandChange(
                            vir_nodal_damage_list, required_demand, total_demand
                        )  # Sina: maybe make it optional
                else:
                    self._registry.addNodalDemandChange(
                        damage_node_name, required_demand, total_demand
                    )

            elif element_type == 'PIPE':
                leak_sum = 0

                pipe_damage_table = self._registry._pipe_damage_table
                pipe_break_history = self._registry._pipe_break_history
                damage_type = pipe_damage_table.loc[damage_node_name, 'damage_type']
                available_node_results = (
                    self._registry.result.node['demand'].loc[stop_time].dropna()
                )
                available_node_results = available_node_results.index
                if damage_type == 'break':
                    if damage_node_name in pipe_damage_table.index:
                        break_node_B = pipe_break_history.loc[
                            damage_node_name, 'Node_B'
                        ]
                        if break_node_B in available_node_results:
                            leak_beark_node_B = self._registry.result.node[
                                'demand'
                            ].loc[stop_time, break_node_B]
                        else:
                            leak_beark_node_B = 0
                        leak_sum += leak_beark_node_B
                    else:
                        break_node_A = (
                            pipe_break_history[
                                pipe_break_history['Node_B'] == damage_node_name
                            ]
                        ).iloc[0]['Node_A']
                        if break_node_A in available_node_results:
                            leak_beark_node_A = self._registry.result.node[
                                'demand'
                            ].loc[stop_time, break_node_A]
                        else:
                            leak_beark_node_A = 0
                        leak_sum += leak_beark_node_A

                if damage_node_name in available_node_results:
                    leak_damaged_node = self._registry.result.node['demand'].loc[
                        stop_time, damage_node_name
                    ]
                    leak_sum += leak_damaged_node

                self._registry._pipe_damage_table.loc[
                    damage_node_name, 'LeakAtCheck'
                ] = leak_sum

        elif effect_type == 'RECONNECT':
            self._registry.addRestorationDataOnPipe(
                damage_node_name, stop_time, 'RECONNECT'
            )
            middle_pipe_size = None
            cv = False
            _length = None
            _friction = None
            if 'PIPESIZE' in single_effect_data:
                middle_pipe_size = single_effect_data['PIPESIZE']
            elif 'PIPESIZEFACTOR' in single_effect_data:
                attached_pipe_name = node_damage_data.attached_element
                attached_pipe = wn.get_link(attached_pipe_name)
                attached_pipe_diameter = attached_pipe.diameter

                middle_pipe_size = attached_pipe_diameter * (
                    single_effect_data['PIPESIZEFACTOR'] ** 0.5
                )

            elif 'CV' in single_effect_data:  # this has a problem /Sina
                cv = single_effect_data['CV']
            elif 'PIPELENGTH' in single_effect_data:
                _length = single_effect_data['PIPELENGTH']
            elif 'PIPEFRICTION' in single_effect_data:
                _friction = single_effect_data['PIPEFRICTION']

            self.repair.bypassPipe(
                damage_node_name,
                middle_pipe_size,
                damage_type,
                wn,
                length=_length,
                friction=_friction,
            )

        elif effect_type == 'ADD_RESERVOIR':
            pump = None
            middle_pipe_size = None
            cv = False
            if 'PIPESIZE' in single_effect_data:
                middle_pipe_size = single_effect_data['PIPESIZE']
            elif 'CV' in single_effect_data:
                cv = single_effect_data['CV']
            elif 'PUMP' in single_effect_data:
                pump = {}
                pump['POWER'] = single_effect_data['PUMP']

                self.repair.addReservoir(
                    damage_node_name, damage_type, 'PUMP', pump, wn
                )
            elif 'ADDEDELEVATION' in single_effect_data:
                reservoir = {}
                reservoir['ADDEDELEVATION'] = single_effect_data['ADDEDELEVATION']

                self.repair.addReservoir(
                    damage_node_name, damage_type, 'ADDEDELEVATION', reservoir, wn
                )
            else:
                raise ValueError(
                    'Unknown parameter. Damaged Node: ' + repr(damage_node_name)
                )

        elif effect_type == 'REMOVE_LEAK':
            factor = None
            if 'LEAKFACTOR' in single_effect_data:
                factor = single_effect_data['LEAKFACTOR']
                self.repair.removeLeak(damage_node_name, damage_type, wn, factor)
            else:
                self.repair.removeLeak(damage_node_name, damage_type, wn)

        elif effect_type == 'ISOLATE_DN':
            if 'FACTOR' in single_effect_data:
                factor = single_effect_data['FACTOR']
            else:
                factor = 1
            real_node_name = damage_node_name
            damage_table = self._registry.getDamageData('DISTNODE', iCopy=True)
            if 'virtual_of' in damage_table.columns:
                real_node_name = get_node_name(damage_node_name, damage_table)

            if self._registry.settings['damage_node_model'] == 'Predefined_demand':
                self.repair.removeDemand(real_node_name, factor, wn)
            elif (
                self._registry.settings['damage_node_model']
                == 'equal_diameter_emitter'
                or self._registry.settings['damage_node_model']
                == 'equal_diameter_reservoir'
            ):
                self.repair.removeDemand(real_node_name, factor, wn)
                self.repair.removeExplicitNodalLeak(real_node_name, factor, wn)
            else:
                raise ValueError('Unknown nodal damage method')

        elif effect_type == 'REPAIR':
            if element_type == 'PIPE':
                self._registry.addRestorationDataOnPipe(
                    damage_node_name, stop_time, 'REPAIR'
                )
                self.repair.removePipeRepair(damage_node_name, wn, action)
                self.repair.repairPipe(damage_node_name, damage_type, wn)
            elif element_type == 'DISTNODE':
                if self._registry.settings['Virtual_node'] == True:
                    real_node_name = get_node_name(
                        damage_node_name, self._registry._node_damage_table
                    )
                    virtual_node_table = self._registry._node_damage_table[
                        self._registry._node_damage_table['Orginal_element']
                        == real_node_name
                    ]
                    temp = virtual_node_table[action] == True
                    if temp.all():
                        self.repairDistNode(real_node_name, wn)

                    else:
                        repaired_number = temp.sum()
                        total_number = virtual_node_table['Number_of_damages'].sum()
                        if self._registry.isVirtualNodeDamaged(damage_node_name):
                            self._registry.setVirtualNodeRepaired(damage_node_name)
                            if (
                                self._registry.settings['damage_node_model']
                                == 'Predefined_demand'
                            ):
                                self.repair.modifyDISTNodeDemandLinearMode(
                                    damage_node_name,
                                    real_node_name,
                                    wn,
                                    repaired_number,
                                    total_number,
                                )
                            elif (
                                self._registry.settings['damage_node_model']
                                == 'equal_diameter_emitter'
                            ):
                                self.repair.modifyDISTNodeExplicitLeakEmitter(
                                    damage_node_name,
                                    real_node_name,
                                    wn,
                                    repaired_number,
                                    total_number,
                                )
                            elif (
                                self._registry.settings['damage_node_model']
                                == 'equal_diameter_reservoir'
                            ):
                                self.repair.modifyDISTNodeExplicitLeakReservoir(
                                    damage_node_name,
                                    real_node_name,
                                    wn,
                                    repaired_number,
                                    total_number,
                                )
                else:
                    self.repairDistNode(real_node_name, wn)

        else:
            raise ValueError('Unknown effect_type: ' + repr(effect_type))

    def repairDistNode(self, damage_node_name, wn):
        self.repair.removeNodeTemporaryRepair(damage_node_name, wn)

    def updateShifiting(self, time):
        """
        Updates shifting with the new time given

        Parameters
        ----------
        time : int
            the current time.

        Returns
        -------
        None.

        """
        if type(time) != int and type(time) != float:
            raise ValueError('Time must be integer not ' + str(type(time)))
        time = int(time)
        if time < 0:
            raise ValueError('Time must be bigger than zero')
        next_shift_time = self.shifting.getNextShiftTime(time)
        # logger.debug('next shitt time = ' + str(next_shift_time))
        self._addHardEvent(int(next_shift_time), 'shift')

        if 'shift' in self._hard_event_table['Requester'].loc[time]:
            self.agents.setChangeShift(time, working_check=True)

    def updateAvailability(self, time):
        # SINA DELETET IT [URGENT]
        # =============================================================================
        #        import pickle
        #
        #         with open("av_data.pkl","rb") as f:
        #             av_data = pickle.load(f)
        #         try:
        #             av_data_time = av_data[time]
        #         except:
        #             av_last_time = 0
        #             time_list = list(av_data.keys())
        #             time_list.append(time)
        #             time_list.sort()
        #
        #             time_list    = pd.Series(data = time_list)
        #             time_index   = time_list.searchsorted(time)
        #             av_last_time = time_list.iloc[time_index-1]
        #
        #             av_data_time = av_data[av_last_time]
        #
        #         self.agents._agents.loc[av_data_time.index, 'available'] = av_data_time.to_list()
        #         #for agent_type in agent_type_list:
        #         return
        # =============================================================================
        agent_type_list = self.agents._agents['type'].unique()
        availible_agent_table = self.agents._agents[
            self.agents._agents['available'].eq(True)
        ]
        for agent_type in agent_type_list:
            if time == self.eq_time:
                av_r = self.agents.getDefaultAvailabilityRatio(agent_type)
            elif time > self.eq_time:
                av_r = self.agents.getAvailabilityRatio(
                    agent_type, time - self.eq_time
                )
                if av_r > 1:
                    av_r = 1

            available_typed_table = availible_agent_table[
                availible_agent_table['type'].eq(agent_type)
            ]
            availible_number = len(available_typed_table)
            all_number = len(
                self.agents._agents[self.agents._agents['type'].eq(agent_type)]
            )
            new_availible_number = np.round(av_r * all_number) - availible_number

            if new_availible_number < 0:
                new_index_list = random.sample(
                    available_typed_table.index.to_list(),
                    int(abs(new_availible_number)),
                )
                self.agents._agents.loc[new_index_list, 'available'] = False
            elif new_availible_number > 0:
                not_available_typed_table = self.agents._agents[
                    (self.agents._agents['type'] == agent_type)
                    & (self.agents._agents['available'] == False)
                ]
                new_index_list = random.sample(
                    not_available_typed_table.index.to_list(),
                    int(new_availible_number),
                )
                self.agents._agents.loc[new_index_list, 'available'] = True

    def initializeActiveAgents(self, time):
        for name, data in self.agents._agents.iterrows():
            agent = data['data']
            if agent.isOnShift(time):
                data['active'] = True
                # data['ready'] = True

    def initializeReadyAgents(self):
        active_agents_name_list = self.agents._agents[
            self.agents._agents['active'].eq(True)
        ].index
        self.agents._agents.loc[active_agents_name_list, 'ready'] = True
        # for name, data in self.agents._agents.iterrows():
        # f data['active'] == True:
        # data['ready'] = True

    # def initializeAvailableAgents(self):
    # ready_agents_name_list = self.agents._agents['ready'].eq(True).index
    # self.agents._agents.loc[ready_agents_name_list, 'available'] = True

    def initializeEntities(self, WaterNetwork):
        for entity, val in self.entity_rule.items():
            element_type = self.entity[entity]
            if not element_type in self.ELEMENTS:
                raise ValueError('Unknown Element type')

            if val[0][0] == 'ALL':
                self._registry.setDamageData(element_type, entity, True)
            else:
                res = []
                node_res = []

                for line in val:
                    attribute = line[0]
                    condition = line[1]
                    condition_value = line[2]

                    temp, temp_node = self._getRefinedElementList(
                        WaterNetwork,
                        attribute,
                        condition,
                        condition_value,
                        element_type,
                        WaterNetwork,
                    )

                    res.append(temp)
                    node_res.append(temp_node)

                union_list = self._unionOfAll(res)

                node_res = self._unionOfAll(node_res)

                self._registry.setDamageDataByList(
                    element_type, node_res, entity, True, iCheck=True
                )

                self._registry.setDamageDataByList(
                    element_type, union_list, entity, True
                )

    def removeRecordsWithoutEntities(self, element_type):
        entity_list = []
        for entity in self.entity:
            if self.entity[entity] == element_type:
                entity_list.append(entity)

        damage_table = self._registry.getDamageData(element_type, iCopy=False)
        if len(entity_list) > 0:
            entities_damaged_table = damage_table[entity_list]

            not_asigned_damaged_table = entities_damaged_table[
                entities_damaged_table.isna().any(1)
            ].index.tolist()
        else:
            not_asigned_damaged_table = damage_table.index.to_list()
        damage_table.drop(not_asigned_damaged_table, inplace=True)

    def initializeGroups(self):
        for el in self.ELEMENTS:
            group_name_list = []

            if el in self.group:
                element_groups_data = self.group[el]
                if len(element_groups_data) < 1:
                    temp = self._registry.getListAllElementOrginalName(el).unique()
                    element_groups_data['default'] = pd.Series(
                        index=temp, data='Default'
                    )

                for group_name in element_groups_data:
                    self._registry.addAttrToElementDamageTable(
                        el, group_name, np.nan
                    )
                    group_name_list.append(group_name)
                    group_data = element_groups_data[group_name]

                    group_location_name_list = (
                        self._registry.getDamagedLocationListByOriginalElementList_2(
                            el, group_data
                        )
                    )
                    group_cat_list = group_data.reindex(group_location_name_list)

                    self._registry.setDamageDataByRowAndColumn(
                        el,
                        group_location_name_list.index.tolist(),
                        group_name,
                        group_cat_list.tolist(),
                    )

            temp = self._registry.getDamageData(el)

            temp = temp[group_name_list]

            temp_list = []
            for col_name, col in temp.iteritems():
                not_na = col.notna()
                not_na = not_na[not_na == False]

                temp_list.append(not_na.index.tolist())

            temp_list = self._unionOfAll(temp_list)
            if len(temp_list) > 0:
                print(
                    'In element: '
                    + repr(el)
                    + ', the following damaged locations does not have a assigned group and will not be affected in the course of restoration:\n'
                    + repr(temp_list)
                )
                logger.warning(
                    'In element: '
                    + repr(el)
                    + ', the following damaged locations does not have a assigned group and will not be affected in the course of restoration:\n'
                    + repr(temp_list)
                )

    def initializeGroups_old(self):
        for el in self.ELEMENTS:
            group_name_list = []

            if el in self.group:
                element_groups_data = self.group[el]
                if len(element_groups_data) < 1:
                    temp = self._registry.getListAllElementOrginalName(el).unique()
                    element_groups_data['default'] = pd.Series(
                        index=temp, data='Default'
                    )

                for group_name in element_groups_data:
                    self._registry.addAttrToElementDamageTable(
                        el, group_name, np.nan
                    )
                    group_name_list.append(group_name)
                    group_data = element_groups_data[group_name]

                    group_location_name_list = (
                        self._registry.getDamagedLocationListByOriginalElementList(
                            el, group_data
                        )
                    )
                    for (
                        damage_location,
                        element_name,
                    ) in group_location_name_list.iteritems():
                        group_cat = group_data.loc[element_name]
                        self._registry.setDamageDataByRowAndColumn(
                            el, damage_location, group_name, group_cat
                        )

            temp = self._registry.getDamageData(el)

            temp = temp[group_name_list]

            temp_list = []
            for col_name, col in temp.iteritems():
                not_na = col.notna()
                not_na = not_na[not_na == False]

                temp_list.append(not_na.index.tolist())

            temp_list = self._unionOfAll(temp_list)
            if len(temp_list) > 0:
                print(
                    'In element: '
                    + repr(el)
                    + ', the following damaged locations does not have a assigned group and will not be affected in the course of restoration:\n'
                    + repr(temp_list)
                )
                logger.warning(
                    'In element: '
                    + repr(el)
                    + ', the following damaged locations does not have a assigned group and will not be affected in the course of restoration:\n'
                    + repr(temp_list)
                )

    def initializeNumberOfDamages(self):
        for element_type in self.ELEMENTS:
            if (
                'Number_of_damages'
                not in (
                    self._registry.getDamageData(element_type, iCopy=False)
                ).columns
            ):
                self._registry.addAttrToElementDamageTable(
                    element_type, 'Number_of_damages', 1
                )

    def _unionOfAll(self, in_list):
        num_of_lists = len(in_list)

        if len(in_list) == 0:
            return in_list

        if len(in_list) == 1:
            if type(in_list[0]) == list:
                return in_list[0]
            else:
                raise ValueError('Something is wrong here')

        first_list = in_list[0]
        second_list = in_list[1]
        union_list = []

        for item in first_list:
            if item in second_list:
                union_list.append(item)

        if num_of_lists == 2:
            return union_list
        else:
            in_list.pop(0)
            in_list[0] = union_list
            return self._unionOfAll(in_list)

    def _getRefinedElementList(
        self, WaterNetwork, attribute, condition, condition_value, element_type, wn
    ):
        res = []
        node_res = []

        if element_type == 'PIPE':
            res = self._getRefinedPipeList(
                WaterNetwork, attribute, condition, condition_value
            )

        elif element_type == 'PUMP':
            res = self._getRefinedPumpList(
                WaterNetwork, attribute, condition, condition_value
            )

        elif element_type in ['DISTNODE', 'GNODE', 'TANK', 'PUMP', 'RESERVOIR']:
            res, node_res = self._getRefinedNodeElementList(
                WaterNetwork, attribute, condition, condition_value, element_type, wn
            )
        else:
            raise ValueError('Unknown Element Type:' + str(element_type))

        return res, node_res

    def refineEntityDamageTable(
        self, damaged_table, group_name, agent_group_tag, element_type
    ):
        ret = []
        # logger.warning('Sina')
        if group_name == None:
            ret = damaged_table
            # logger.warning('1')

        elif group_name in damaged_table.columns:
            # logger.warning('2')
            # logger.warning(group_name)
            # logger.warning(agent_group_tag)
            # logger.warning(damaged_table[damaged_table[group_name]==agent_group_tag])
            # agent_type = damaged_table['type'].iloc[0]
            ret = damaged_table[damaged_table[group_name] == agent_group_tag]
            if len(ret) == 0:
                logger.warning(
                    'Empty damage table in element type='
                    + repr(element_type)
                    + 'group name='
                    + repr(group_name)
                    + ', group_tag='
                    + repr(agent_group_tag)
                )
        else:
            ret = pd.DataFrame(columns=damaged_table.columns)

        return ret

    def _refine_table(self, table, attribute, condition, condition_value):
        refined_table = None

        if type(condition_value) == str:
            if condition == 'EQ':
                refined_table = table[table[attribute] == condition_value]
            elif condition == 'NOTEQ':
                refined_table = table[table[attribute] != condition_value]
            else:
                raise ValueError('Undefined condition: ' + repr(condition))
        elif type(condition_value) == int or type(condition_value) == float:
            if condition == 'EQ':
                refined_table = table[table[attribute] == condition_value]
            elif condition == 'BG-EQ':
                refined_table = table[table[attribute] >= condition_value]
            elif condition == 'LT-EQ':
                refined_table = table[table[attribute] <= condition_value]
            elif condition == 'BG':
                refined_table = table[table[attribute] > condition_value]
            elif condition == 'LT':
                refined_table = table[table[attribute] < condition_value]
            elif condition == 'NOTEQ':
                refined_table = table[table[attribute] != condition_value]
            else:
                raise ValueError('Undefined condition: ' + repr(condition))
        else:
            raise ValueError('Undefined data type: ' + repr(type(condition_value)))

        return refined_table

    def _getRefinedNodeElementList(
        self, WaterNetwork, attribute, condition, condition_value, element_type, wn
    ):
        res = []
        node_res = []

        if attribute == 'FILE' or attribute == 'NOT_IN_FILE':
            node_damage_list = self._registry.getDamageData(element_type)

            for org_file_name in condition_value:
                # not_included = set(org_file_name) - set(wn.node_name_list)
                if org_file_name not in node_damage_list.index:
                    # Sina out it back. Suppressed for runing in cluster
                    continue

                if 'virtual_of' in node_damage_list.columns:
                    node_name_list = node_damage_list.index
                    temp_damage_table = node_damage_list.set_index(
                        'Virtual_node', drop=False
                    )
                    temp_damage_table['random_sina_index'] = node_name_list.tolist()
                    temp = temp_damage_table.loc[org_file_name]
                    temp.index = temp['random_sina_index']
                    temp = temp.drop('random_sina_index', axis=1)
                else:
                    if type(org_file_name) == str:
                        org_file_name = [org_file_name]
                    temp = node_damage_list.loc[org_file_name]

                ichosen = False

                if len(temp) >= 1:
                    res.extend(temp.index.tolist())
                    ichosen = True

                if ichosen == False:
                    if org_file_name in wn.node_name_list:
                        ichosen = True
                        node_res.append(org_file_name)
                if ichosen == False:
                    raise ValueError(
                        'Element with ID: '
                        + repr(org_file_name)
                        + ' is not either a element: '
                        + repr(element_type)
                        + ' or a node.'
                    )

                if attribute == 'NOT_IN_FILE':
                    index_list = node_damage_list.index.tolist()
                    for in_file in res:
                        index_list.remove(in_file)

                    res = index_list

                # res.extend(node_res)

        elif attribute in self._registry.getDamageData(element_type).columns:
            temp = self._registry.getDamageData(element_type)

            refined_table = self._refine_table(
                temp, attribute, condition, condition_value
            )
            refined_table = refined_table.index
            res = refined_table.to_list()
        else:
            raise ValueError('Unknown Entity Condition: ' + condition)

        return res, node_res

    def _getRefinedPumpList(
        self, WaterNetwork, attribute, condition, condition_value
    ):
        element_res = []

        if attribute == 'FILE' or condition == 'NOT_IN_FILE':
            pump_damage_list = self._registry.getDamageData('PUMP')

            for org_file_name in condition_value:
                temp = pump_damage_list[
                    pump_damage_list['element_name'] == org_file_name
                ]

                if len(temp) == 1:
                    element_res.append(temp.element_name[0])
                elif len(temp) > 1:
                    raise ValueError('Something wrong here')

            if attribute == 'NOT_IN_FILE':
                index_list = pump_damage_list.element_name.tolist()
                for in_file in element_res:
                    index_list.remove(in_file)

                element_res = index_list

        elif attribute in self._registry.getDamageData('PUMP').columns:
            temp = self._registry._pump_damage_table

            refined_table = self._refine_table(
                temp, attribute, condition, condition_value
            )

            refined_table = refined_table.index
            element_res = refined_table.to_list()
        else:
            raise ValueError('Unknown Entity Condition: ' + attribute)

        res = []
        pump_name_list = pump_damage_list['element_name'].tolist()

        for element_name in element_res:
            if element_name in pump_name_list:
                temp = pump_damage_list[
                    pump_damage_list['element_name'] == element_name
                ].index[0]

                res.append(temp)
        return res

    def _getRefinedPipeList(
        self, WaterNetwork, attribute, condition, condition_value
    ):
        res = []

        # if condition in self._CONDITIONS:
        if attribute.upper() in ['DIAMETER']:
            # for pipe_name in WaterNetwork.pipe_name_list:
            for damage_name, line in self._registry.getDamageData('PIPE').iterrows():
                if attribute.upper() == 'DIAMETER':
                    # orginal_element   = line['Orginal_element']
                    attached_elements = line['attached_element']

                    pipe = WaterNetwork.get_link(attached_elements)
                    pipe_value = pipe.diameter
                else:
                    raise ValueError('Undefined attribute ' + attribute)

                if condition == 'EQ':
                    if pipe_value == condition_value:
                        res.append(damage_name)
                elif condition == 'BG':
                    if pipe_value > condition_value:
                        res.append(damage_name)
                elif condition == 'LT':
                    if pipe_value < condition_value:
                        res.append(damage_name)
                elif condition == 'BG-EQ':
                    if pipe_value >= condition_value:
                        res.append(damage_name)
                elif condition == 'LT-EQ':
                    if pipe_value <= condition_value:
                        res.append(damage_name)

        elif attribute == 'FILE' or attribute == 'NOT_IN_FILE':
            pipe_damage_list = self._registry.getDamageData('PIPE')
            for org_file_name in condition_value:
                temp = pipe_damage_list[
                    pipe_damage_list['Orginal_element'] == org_file_name
                ]

                if len(temp) == 1:
                    res.append(temp.index[0])
                elif len(temp) > 1:
                    res.extend(temp.index.to_list())

            if attribute == 'NOT_IN_FILE':
                index_list = pipe_damage_list.index.tolist()
                for in_file in res:
                    i = 0

                    while i in range(0, len(index_list)):
                        if index_list[i] == in_file:
                            index_list.pop(i)
                        i += 1

                res = index_list

        elif attribute in self._registry.getDamageData('PIPE').columns:
            temp = self._registry.getDamageData('PIPE')

            refined_table = self._refine_table(
                temp, attribute, condition, condition_value
            )

            refined_table = refined_table.index
            res = refined_table.to_list()
        else:
            raise ValueError('Unknown Entity Condition: ' + condition)

        return res

    def _getReminderTime(self, name):
        return self._reminder_time_hard_event[name]

    def _saveReminderTime(self, time, name):
        if name not in self._reminder_time_hard_event:
            self._reminder_time_hard_event[name] = int(time)
        else:
            self._reminder_time_hard_event[name] += int(time)

    def _addHardEvent(self, next_time, requester, detail=None, current_time=None):
        """
        Adds a hard event

        Parameters
        ----------
        time : int
            Time of hard event.
        requester : str
            Requeter type

        Returns
        -------
        None.

        """
        time = int(next_time)
        next_time = int(next_time)
        if type(next_time) != int and type(next_time) != float:
            raise ValueError('time must be int, not ' + str(type(next_time)))
        if detail != None and current_time == None:
            raise ValueError('When detail is provided, current time cannot be None')

        minimum_time_devision = int(self._registry.settings['simulation_time_step'])
        if current_time != None:
            if next_time < current_time:
                raise ValueError('Time is smaller than current time')
            if detail == None:
                raise ValueError(
                    'When current time is provided, detail cannot be None'
                )
            if minimum_time_devision < 0:
                raise ValueError('Minimum time devision cannot be negative')

            name = requester + '-' + detail

            time = next_time - current_time

            _b = np.round(time / minimum_time_devision)

            if abs(_b) < 0.01:
                _b = 1

            new_time = _b * minimum_time_devision
            reminder = time - new_time
            self._saveReminderTime(reminder, name)
            next_time = current_time + new_time

        if next_time not in self._hard_event_table.index:
            self._hard_event_table.loc[next_time] = [
                [
                    requester,
                ],
                True,
                [
                    detail,
                ],
            ]
        elif (
            requester in self._hard_event_table.loc[next_time, 'Requester']
            and detail == None
        ):
            pass
        else:
            self._hard_event_table.loc[next_time, 'Requester'].append(requester)
            self._hard_event_table.loc[next_time, 'New'] = True
            self._hard_event_table.loc[next_time, 'Detail'].append(detail)

        return next_time

    def _isHardEvent(self, time, requester=None):
        if requester == None:
            return time in self._hard_event_table.index
        else:
            if time in self._hard_event_table.index:
                req = self._hard_event_table.loc[time, 'Requester']
                if requester in req:
                    return True
            return False

    def getHardEventDetails(self, time, by=None):
        if by == None:
            return self._hard_event_table.loc[time, 'Detail']
        elif by not in self._hard_event_table.loc[time, 'Requester']:
            return []
        else:
            res = []
            requester_list = self._hard_event_table.loc[time, 'Requester']
            detail_list = self._hard_event_table.loc[time, 'Detail']
            i = 0
            for requester in requester_list:
                if requester == by:
                    res.append(detail_list[i])
                i += 1
            return res

    def getNewEventsTime(self, reset=False):
        new_event_table = self._hard_event_table[
            self._hard_event_table['New'] == True
        ]
        new_event_table = new_event_table.sort_index()

        if reset == True:
            for ind, val in new_event_table.iterrows():
                self._hard_event_table.loc[ind, 'New'] = False

        return list(new_event_table.index)

    def unmarkNewEvents(self):
        self._hard_event_table['new'][self._hard_event_table['New'] == True] = False

    def getAllSequences(self, element_type):
        return self.sequence[element_type]

    def getNextSequence(self, element_type, cur_seq):
        seq_list = self.sequence[element_type]
        if cur_seq not in seq_list:
            raise ValueError('Seqence was not in sequnce list: ' + str(cur_seq))
        i = 0
        for seq in seq_list:
            if cur_seq == seq:
                break
            i += 1
        if not i + 1 < len(seq_list):
            return None
        else:
            return seq_list[i + 1]

    def initialize(self, wn, stop_time, delay=0, earthquake=None):
        self.if_initiated = True
        self.eq_time = stop_time
        if delay < 0:
            raise ValueError('delay value is less than 0: ' + str(delay))
        self.delay = delay

        if stop_time < 0:
            raise ValueError('Stop time is less than 0')

        # refined_pump = self.pump_restoration[self.pump_restoration['Restore_time']>=stop_time]
        if not self.pump_restoration.empty:
            self.pump_restoration['Restore_time'] = (
                self.pump_restoration['Restore_time'] + stop_time
            )

        if not self.tank_restoration.empty:
            self.tank_restoration['Restore_time'] = (
                self.tank_restoration['Restore_time'] + stop_time
            )

        for (
            ind,
            row,
        ) in self.pump_restoration.items():
            self._addHardEvent(row['Restore_time'], 'pump')

        if type(self.tank_restoration) != pd.core.series.Series:
            raise
        for (
            ind,
            row,
        ) in self.tank_restoration.items():
            self._addHardEvent(row['Restore_time'], 'tank')

        self.restoration_start_time = stop_time + delay

        self._addHardEvent(self.restoration_start_time, 'start')
        self.initializeActiveAgents(stop_time)
        self.initializeReadyAgents()

        for node_name in wn.node_name_list:
            self._registry.addGeneralNodeDamageToRegistry(node_name)

        for tank_name in wn.tank_name_list:
            self._registry.addTankDamageToRegistry(tank_name)

        for pump_name in wn.pump_name_list:
            self._registry.addPumpDamageToRegistry(pump_name, wn.get_link(pump_name))

        for reservoir_name in wn.reservoir_name_list:
            self._registry.addReservoirDamageToRegistry(reservoir_name)

        self.initializeEntities(wn)
        self.removeRecordsWithoutEntities('TANK')
        self.removeRecordsWithoutEntities('RESERVOIR')
        self.removeRecordsWithoutEntities('PUMP')
        self.removeRecordsWithoutEntities('GNODE')

        for el in self.ELEMENTS:
            self._registry.setDamageData(el, 'discovered', False)
        self.initializeGroups()
        self.initializeNumberOfDamages()

        for seq_key, seq_list in self.sequence.items():
            self._registry.setDamageData(seq_key, seq_list[0], False)

        if self.delay == 0:
            event_time_list = self.perform_action(wn, stop_time)
        else:
            event_time_list = self.getNewEventsTime(reset=True)

        if earthquake != None:
            self.earthquake = earthquake

        event_time_list = event_time_list[1:]
        return event_time_list

    def iRestorationStopTime(self):
        if self.if_initiated == False:
            return False
        logger.debug('Func: node functionality')
        pipe_damage_end = self.iAllPipeLastActionDone()
        node_damage_end = self.iAllNodeLastActionDone()
        pump_damage_end = self.iAllPumpLastActionDone()
        GNODE_damage_end = self.iAllGNodeLastActionDone()
        tank_damage_end = self.iAllTankLastActionDone()
        reservoir_damage_end = self.iAllReservoirLastActionDone()

        logger.debug('pipe: ' + repr(pipe_damage_end))
        logger.debug('node: ' + repr(node_damage_end))
        logger.debug('pump: ' + repr(pump_damage_end))
        logger.debug('GNODE: ' + repr(GNODE_damage_end))
        logger.debug('tank: ' + repr(tank_damage_end))
        logger.debug('reservoir: ' + repr(reservoir_damage_end))

        if (
            pipe_damage_end
            and node_damage_end
            and pump_damage_end
            and GNODE_damage_end
            and tank_damage_end
            and reservoir_damage_end
        ):
            return True
        else:
            return False

    def iAllPipeLastActionDone(self):
        print()
        if 'PIPE' in self.sequence:
            if len(self._registry._pipe_damage_table) == 0:
                return True

            pipe_action = self.sequence['PIPE'][-1]
            pipe_last_action_values = self._registry._pipe_damage_table[pipe_action]
            if_pipe_last_action_true = (
                pipe_last_action_values
                == True | (pipe_last_action_values == 'Collective')
            ).all()
            if if_pipe_last_action_true:
                return True
            else:
                return False
        else:
            return True

    def iAllNodeLastActionDone(self):
        if 'DISTNODE' in self.sequence:
            if len(self._registry._node_damage_table) == 0:
                return True

            node_action = self.sequence['DISTNODE'][-1]
            node_last_action_values = self._registry._node_damage_table[node_action]
            if_node_last_action_true = (
                node_last_action_values
                == True | (node_last_action_values == 'Collective')
            ).all()

            if if_node_last_action_true == True:
                return True
            else:
                return False
        else:
            return True

    def iAllPumpLastActionDone(self):
        if 'PUMP' in self.sequence:
            if len(self._registry._pump_damage_table) == 0:
                return True

            pump_action = self.sequence['PUMP'][-1]
            pump_last_action_values = self._registry._pump_damage_table[pump_action]

            if len(self._registry._pump_damage_table) == 0:
                return True

            if_pump_last_action_true = (pump_last_action_values == True).all()

            if if_pump_last_action_true == True:
                return True
            else:
                return False
        else:
            return True

    def iAllGNodeLastActionDone(self):
        if 'GNODE' in self.sequence:
            if len(self._registry._gnode_damage_table) == 0:
                return True

            gnode_action = self.sequence['GNODE'][-1]
            gnode_last_action_values = self._registry._gnode_damage_table[
                gnode_action
            ]
            if_gnode_last_action_true = (gnode_last_action_values == True).all()

            if if_gnode_last_action_true == True:
                return True
            else:
                return False
        else:
            return True

    def iAllTankLastActionDone(self):
        if 'TANK' in self.sequence:
            if len(self._registry._tank_damage_table) == 0:
                return True

            tank_action = self.sequence['TANK'][-1]
            tank_last_action_values = self._registry._tank_damage_table[tank_action]
            if_tank_last_action_true = (tank_last_action_values == True).all()

            if if_tank_last_action_true == True:
                return True
            else:
                return False
        else:
            return True

    def iAllReservoirLastActionDone(self):
        if 'RESERVOIR' in self.sequence:
            if len(self._registry._reservoir_damage_table) == 0:
                return True

            reservoir_action = self.sequence['RESERVOIR'][-1]
            reservoir_last_action_values = self._registry._reservoir_damage_table[
                reservoir_action
            ]
            if_reservoir_last_action_true = (
                reservoir_last_action_values == True
            ).all()

            if if_reservoir_last_action_true == True:
                return True
            else:
                return False
        else:
            return True

    def getHydSigPipeList(self):
        damage_group_list = self.priority.getHydSigDamageGroups()
        pipe_damage_group_list = [
            cur_damage_group
            for cur_damage_group in damage_group_list
            if self.entity[cur_damage_group] == 'PIPE'
        ]
        return pipe_damage_group_list
