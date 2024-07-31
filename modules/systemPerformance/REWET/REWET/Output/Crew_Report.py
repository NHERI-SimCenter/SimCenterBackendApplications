"""Created on Thu Oct 27 15:45:10 2022

@author: snaeimi
"""  # noqa: INP001, D400, D415

import pandas as pd


class Crew_Report:  # noqa: N801, D101
    def __init__(self):  # noqa: ANN204, D107
        pass

    def getCrewForTime(self, scn_name, time):  # noqa: ANN001, ANN201, N802, D102
        self.loadScneariodata(scn_name)
        reg = self.registry[scn_name]

        crew_table = reg.restoration_log_book._agent_state_log_book  # noqa: SLF001
        crew_table = crew_table.set_index('Time')
        crew_table = crew_table.loc[time]
        return crew_table  # noqa: RET504

    def getCrewTableAt(self, scn_name, time, crew_type_name, crew_zone=None):  # noqa: ANN001, ANN201, N802, D102
        self.loadScneariodata(scn_name)
        reg = self.registry[scn_name]
        # crew_type  = self.getCrewForTime(scn_name, time)  # noqa: ERA001
        crew_table = reg.restoration_log_book.crew_history[time]
        typed_crew_table = crew_table[crew_table['type'] == crew_type_name]

        if crew_zone is not None:
            if type(crew_zone) == str:  # noqa: E721
                typed_crew_table = typed_crew_table[
                    typed_crew_table['group'] == crew_zone
                ]
            elif type(crew_zone) == list:  # noqa: E721
                i = 0
                for crew_zone_value in crew_zone:
                    if i == 0:
                        res = typed_crew_table['group'] == crew_zone_value
                    else:
                        res = (typed_crew_table['group'] == crew_zone_value) | res
                    i += 1  # noqa: SIM113
                typed_crew_table = typed_crew_table[res]
            else:
                raise ValueError('Unknown crew_zone type: ' + repr(type(crew_zone)))

        return typed_crew_table

    def getCrewAvailabilityThroughTime(  # noqa: ANN201, N802, D102
        self, scn_name, crew_type_name, crew_zone=None  # noqa: ANN001
    ):
        self.loadScneariodata(scn_name)
        reg = self.registry[scn_name]
        crew_table = reg.restoration_log_book.crew_history
        time_list = list(crew_table.keys())
        time_list.sort()

        crew_number = pd.Series()

        for time in time_list:
            crew_table_time = self.getCrewTableAt(
                scn_name, time, crew_type_name, crew_zone
            )
            total_number = len(crew_table_time)
            available_number_time = crew_table_time[
                (crew_table_time['available'] == True)  # noqa: E712
                | (crew_table_time['active'] == True)  # noqa: E712
            ]
            crew_number.loc[time] = len(available_number_time)

        return total_number, crew_number

    def getCrewOnShiftThroughTime(  # noqa: ANN201, N802, D102
        self, scn_name, crew_type_name, crew_zone=None, not_on_shift=False  # noqa: ANN001, FBT002
    ):
        self.loadScneariodata(scn_name)
        reg = self.registry[scn_name]
        crew_table = reg.restoration_log_book.crew_history
        time_list = list(crew_table.keys())
        time_list.sort()

        crew_number = pd.Series()

        for time in time_list:
            crew_table_time = self.getCrewTableAt(
                scn_name, time, crew_type_name, crew_zone
            )
            total_number = len(crew_table_time)

            if not_on_shift == False:  # noqa: E712
                available_number_time = crew_table_time[
                    crew_table_time['active'] == True  # noqa: E712
                ]
            elif not_on_shift == True:  # noqa: E712
                available_number_time = crew_table_time[
                    crew_table_time['active'] == False  # noqa: E712
                ]
            else:
                raise ValueError('Unnown not on shift' + repr(not_on_shift))
            crew_number.loc[time] = len(available_number_time)

        return total_number, crew_number

    def getCrewWorkingThroughTime(  # noqa: ANN201, N802, D102
        self, scn_name, crew_type_name, crew_zone=None, not_on_working=False  # noqa: ANN001, FBT002
    ):
        self.loadScneariodata(scn_name)
        reg = self.registry[scn_name]
        crew_table = reg.restoration_log_book.crew_history
        time_list = list(crew_table.keys())
        time_list.sort()

        crew_number = pd.Series()

        for time in time_list:
            crew_table_time = self.getCrewTableAt(
                scn_name, time, crew_type_name, crew_zone
            )
            total_number = len(crew_table_time)
            # available_number_time = crew_table_time[crew_table_time['available']==True]  # noqa: ERA001, E501
            available_number_time = crew_table_time[
                crew_table_time['active'] == True  # noqa: E712
            ]
            if not_on_working == False:  # noqa: E712
                available_number_time = available_number_time[
                    available_number_time['ready'] == False  # noqa: E712
                ]
            elif not_on_working == True:  # noqa: E712
                available_number_time = available_number_time[
                    available_number_time['ready'] == True  # noqa: E712
                ]
            else:
                raise ValueError('Unnown not on shift' + repr(not_on_working))
            crew_number.loc[time] = len(available_number_time)

        return total_number, crew_number

    def getCrewCompleteStatusReport(self, scn_name, crew_type_name, crew_zone=None):  # noqa: ANN001, ANN201, N802, D102
        self.loadScneariodata(scn_name)
        reg = self.registry[scn_name]
        crew_table = reg.restoration_log_book.crew_history
        time_list = list(crew_table.keys())
        time_list.sort()

        crew_report = pd.DataFrame(
            index=time_list,
            columns=[
                'Reported',
                'Not-reported',
                'Total_not-reported',
                'on-duty',
                'off-duty',
                'idle',
                'busy',
            ],
            data=0,
        )

        for time in time_list:
            crew_table_time = self.getCrewTableAt(
                scn_name, time, crew_type_name, crew_zone
            )

            for agent_index, agent_row in crew_table_time.iterrows():  # noqa: B007
                if agent_row['data'].isOnShift(time):
                    crew_report.loc[time, 'on-duty'] += 1
                else:
                    crew_report.loc[time, 'off-duty'] += 1

                # iAvailable = agent_row['available']  # noqa: ERA001
                if agent_row['available'] or agent_row['active']:
                    crew_report.loc[time, 'Reported'] += 1
                    if agent_row['active'] and agent_row['ready']:
                        crew_report.loc[time, 'idle'] += 1
                    elif agent_row['active'] and agent_row['ready'] == False:  # noqa: E712
                        crew_report.loc[time, 'busy'] += 1
                else:
                    crew_report.loc[time, 'Total_not-reported'] += 1
                    if agent_row['data'].isOnShift(time):
                        crew_report.loc[time, 'Not-reported'] += 1
                    if agent_row['active'] == True:  # noqa: E712
                        print('time=' + str(time))  # noqa: T201
                        print(agent_row)  # noqa: T201

        return crew_report
