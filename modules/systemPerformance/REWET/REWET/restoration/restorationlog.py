# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 21:54:19 2021

@author: snaeimi
"""

from collections import OrderedDict
import pandas as pd


class RestorationLog:
    def __init__(self, settings):
        self.settings = settings
        self._agent_state_log_book = pd.DataFrame(
            columns=[
                'Time',
                'Name',
                'Type',
                'Lable',
                'Action',
                'EFN',
                'MN',
                'Location',
                'X',
                'Y',
            ]
        )
        self._agent_action_log_book = pd.DataFrame(
            columns=[
                'Agent',
                'Node',
                'Entity',
                'Action',
                'Time',
                'End_time',
                'Travel_time',
                'effect_definition_name',
                'method_name',
                'iFinished',
            ]
        )
        self.crew_history = OrderedDict()

    def updateAgentHistory(self, agent_table, time):
        if self.settings['record_restoration_agent_logs'] == False:
            return

        self.crew_history[time] = agent_table.copy()

    def updateAgentLogBook(self, agent_table, time):
        if self.settings['record_restoration_agent_logs'] == False:
            return

        for agent_name, line in agent_table.iterrows():
            temp = None
            if line['active'] == True and line['ready'] == False:
                data = line['data']
                _x = data.current_location.coord.x
                _y = data.current_location.coord.y
                _name = data.name
                _type = data.agent_type
                _lable = data.cur_job_entity
                _action = data.cur_job_action
                _EFN = data.cur_job_effect_definition_name
                _MN = data.cur_job_method_name
                _loc = data.cur_job_location

                temp = pd.Series(
                    data=[
                        int(time),
                        _name,
                        _type,
                        _lable,
                        _action,
                        _EFN,
                        _MN,
                        _loc,
                        _x,
                        _y,
                    ],
                    index=[
                        'Time',
                        'Name',
                        'Type',
                        'Lable',
                        'Action',
                        'EFN',
                        'MN',
                        'Location',
                        'X',
                        'Y',
                    ],
                )

            # if temp != None:
            self._agent_state_log_book = self._agent_state_log_book.append(
                temp, ignore_index=True
            )

    def addAgentActionToLogBook(
        self,
        agent_name,
        node_name,
        entity,
        action,
        time,
        end_time,
        travel_time,
        effect_definition_name,
        method_name,
        iFinished=True,
    ):
        if self.settings['record_restoration_agent_logs'] == False:
            return

        temp = pd.Series(
            data=[
                agent_name,
                node_name,
                entity,
                action,
                time,
                end_time,
                travel_time,
                effect_definition_name,
                method_name,
                iFinished,
            ],
            index=[
                'Agent',
                'Node',
                'Entity',
                'Action',
                'Time',
                'End_time',
                'Travel_time',
                'effect_definition_name',
                'method_name',
                'iFinished',
            ],
        )
        self._agent_action_log_book = self._agent_action_log_book.append(
            temp, ignore_index=True
        )

    def addEndTimegentActionToLogBook(self, agent_name, time, modified_end_time):
        if self.settings['record_restoration_agent_logs'] == False:
            return

        temp = self._agent_action_log_book[['Agent', 'Time']] == [agent_name, time]
        temp = self._agent_action_log_book[temp.all(1)]

        if len(temp) > 1:
            raise ValueError(
                'There are too many agents record with the same time and name'
            )

        elif len(temp) == 0:
            raise ValueError(
                'There is not agent agent record with this time and name'
            )

        ind = temp.index

        self._agent_action_log_book.loc[ind, 'Modified_end_time'] = modified_end_time


# =============================================================================
#     def getAgentActioLogBookat(self, time, end_time=True):
#         res=None
#
#         if end_time==True:
#             res=self._agent_action_log_book[self._agent_action_log_book['Modified_end_time']==time]
#         else:
#             res=self._agent_action_log_book[self._agent_action_log_book['Time']==time]
#
#         return res
# =============================================================================
