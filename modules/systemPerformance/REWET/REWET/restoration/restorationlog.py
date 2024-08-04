"""Created on Sun Jan 31 21:54:19 2021

@author: snaeimi
"""  # noqa: CPY001, D400, INP001

from collections import OrderedDict

import pandas as pd


class RestorationLog:  # noqa: D101
    def __init__(self, settings):  # noqa: ANN001, ANN204
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

    def updateAgentHistory(self, agent_table, time):  # noqa: ANN001, ANN201, N802, D102
        if self.settings['record_restoration_agent_logs'] == False:  # noqa: E712
            return

        self.crew_history[time] = agent_table.copy()

    def updateAgentLogBook(self, agent_table, time):  # noqa: ANN001, ANN201, N802, D102
        if self.settings['record_restoration_agent_logs'] == False:  # noqa: E712
            return

        for agent_name, line in agent_table.iterrows():  # noqa: B007
            temp = None
            if line['active'] == True and line['ready'] == False:  # noqa: E712
                data = line['data']
                _x = data.current_location.coord.x
                _y = data.current_location.coord.y
                _name = data.name
                _type = data.agent_type
                _lable = data.cur_job_entity
                _action = data.cur_job_action
                _EFN = data.cur_job_effect_definition_name  # noqa: N806
                _MN = data.cur_job_method_name  # noqa: N806
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

    def addAgentActionToLogBook(  # noqa: ANN201, N802, D102
        self,
        agent_name,  # noqa: ANN001
        node_name,  # noqa: ANN001
        entity,  # noqa: ANN001
        action,  # noqa: ANN001
        time,  # noqa: ANN001
        end_time,  # noqa: ANN001
        travel_time,  # noqa: ANN001
        effect_definition_name,  # noqa: ANN001
        method_name,  # noqa: ANN001
        iFinished=True,  # noqa: ANN001, FBT002, N803
    ):
        if self.settings['record_restoration_agent_logs'] == False:  # noqa: E712
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

    def addEndTimegentActionToLogBook(self, agent_name, time, modified_end_time):  # noqa: ANN001, ANN201, N802, D102
        if self.settings['record_restoration_agent_logs'] == False:  # noqa: E712
            return

        temp = self._agent_action_log_book[['Agent', 'Time']] == [agent_name, time]
        temp = self._agent_action_log_book[temp.all(1)]

        if len(temp) > 1:
            raise ValueError(  # noqa: TRY003
                'There are too many agents record with the same time and name'  # noqa: EM101
            )

        elif len(temp) == 0:  # noqa: RET506
            raise ValueError(  # noqa: TRY003
                'There is not agent agent record with this time and name'  # noqa: EM101
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
