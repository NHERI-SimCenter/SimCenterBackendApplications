# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:45:10 2022

@author: snaeimi
"""

import pandas as pd

class Crew_Report():
    def __init__(self):
        pass
    
    def getCrewForTime(self, scn_name, time):
        self.loadScneariodata(scn_name)
        reg = self.registry[scn_name]
        
        crew_table = reg.restoration_log_book._agent_state_log_book
        crew_table = crew_table.set_index('Time')
        crew_table = crew_table.loc[time]
        return crew_table
    
    def getCrewTableAt(self, scn_name, time, crew_type_name, crew_zone=None):
        self.loadScneariodata(scn_name)
        reg = self.registry[scn_name]
        #crew_type  = self.getCrewForTime(scn_name, time)
        crew_table = reg.restoration_log_book.crew_history[time]
        typed_crew_table = crew_table[crew_table['type']==crew_type_name]
        
        if type(crew_zone) != type(None):
            if type(crew_zone) == str:
                typed_crew_table = typed_crew_table[typed_crew_table['group']==crew_zone]
            elif type(crew_zone) == list:
                i = 0
                for crew_zone_value in crew_zone:
                    if i == 0:
                        res = typed_crew_table['group']==crew_zone_value
                    else:
                        res = (typed_crew_table['group']==crew_zone_value) | res
                    i += 1
                typed_crew_table = typed_crew_table[res]
            else:
                raise ValueError("Unknown crew_zone type: " + repr(type(crew_zone) ) )
            
        return typed_crew_table
    
    def getCrewAvailabilityThroughTime(self, scn_name, crew_type_name, crew_zone=None):
        self.loadScneariodata(scn_name)
        reg = self.registry[scn_name]
        crew_table = reg.restoration_log_book.crew_history
        time_list = list(crew_table.keys())
        time_list.sort()
        
        crew_number = pd.Series()
        
        for time in time_list:
            crew_table_time       = self.getCrewTableAt(scn_name, time, crew_type_name, crew_zone)
            total_number          = len(crew_table_time)
            available_number_time = crew_table_time[(crew_table_time['available']==True) | (crew_table_time['active']==True)]
            crew_number.loc[time] = len(available_number_time)
        
        return total_number, crew_number
    
    def getCrewOnShiftThroughTime(self, scn_name, crew_type_name, crew_zone=None, not_on_shift=False):
        self.loadScneariodata(scn_name)
        reg = self.registry[scn_name]
        crew_table = reg.restoration_log_book.crew_history
        time_list = list(crew_table.keys())
        time_list.sort()
        
        crew_number = pd.Series()
        
        for time in time_list:
            crew_table_time       = self.getCrewTableAt(scn_name, time, crew_type_name, crew_zone)
            total_number          = len(crew_table_time)

            if not_on_shift==False:
                available_number_time = crew_table_time[crew_table_time['active']==True]
            elif not_on_shift==True:
                available_number_time = crew_table_time[crew_table_time['active']==False]
            else:
                raise ValueError("Unnown not on shift" + repr(not_on_shift))
            crew_number.loc[time] = len(available_number_time)
        
        return total_number, crew_number
    
    def getCrewWorkingThroughTime(self, scn_name, crew_type_name, crew_zone=None, not_on_working=False):
        self.loadScneariodata(scn_name)
        reg = self.registry[scn_name]
        crew_table = reg.restoration_log_book.crew_history
        time_list = list(crew_table.keys())
        time_list.sort()
        
        crew_number = pd.Series()
        
        for time in time_list:
            crew_table_time       = self.getCrewTableAt(scn_name, time, crew_type_name, crew_zone)
            total_number          = len(crew_table_time)
            #available_number_time = crew_table_time[crew_table_time['available']==True]
            available_number_time = crew_table_time[crew_table_time['active']==True]
            if not_on_working==False:
                available_number_time = available_number_time[available_number_time['ready']==False]
            elif not_on_working==True:
                available_number_time = available_number_time[available_number_time['ready']==True]
            else:
                raise ValueError("Unnown not on shift" + repr(not_on_working))
            crew_number.loc[time] = len(available_number_time)
        
        return total_number, crew_number
    
    
    def getCrewCompleteStatusReport(self, scn_name, crew_type_name, crew_zone=None):
        self.loadScneariodata(scn_name)
        reg = self.registry[scn_name]
        crew_table = reg.restoration_log_book.crew_history
        time_list = list(crew_table.keys())
        time_list.sort()
    
        crew_report = pd.DataFrame(index=time_list, columns=["Reported", "Not-reported", "Total_not-reported", "on-duty", "off-duty", "idle", "busy"], data=0)
        
        for time in time_list:
            crew_table_time       = self.getCrewTableAt(scn_name, time, crew_type_name, crew_zone)

            for agent_index , agent_row in crew_table_time.iterrows():
                
                if agent_row['data'].isOnShift(time):
                    crew_report.loc[time, 'on-duty'] += 1
                else:
                    crew_report.loc[time, 'off-duty'] += 1
                
                #iAvailable = agent_row['available']
                if agent_row['available'] or agent_row['active']:
                    crew_report.loc[time, 'Reported'] += 1
                    if agent_row["active"] and agent_row["ready"]:
                        crew_report.loc[time, 'idle'] += 1
                    elif agent_row["active"] and agent_row["ready"]==False:
                        crew_report.loc[time, 'busy'] += 1
                else:
                    crew_report.loc[time, 'Total_not-reported'] += 1
                    if agent_row['data'].isOnShift(time):
                        crew_report.loc[time, 'Not-reported'] += 1
                    if agent_row['active'] == True:
                        print("time=" + str(time))
                        print(agent_row)

        
        return crew_report