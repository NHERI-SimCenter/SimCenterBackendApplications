# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 02:00:40 2020

@author: snaeimi
"""

import pandas as pd
import numpy
import logging

logger = logging.getLogger(__name__)

EVENT_TYPE=['dmg','rpr','rst'] #event types are defined here
class Timeline():
    
# =============================================================================
# This classs has many functions that can make a lot of exceptions.
# We need to modify their codes, so their usage be safe and bug-free.
# =============================================================================

    def __init__(self, simulation_end_time, restoration, registry):
        if  simulation_end_time<0:
            raise ValueError('simulation end time must be zero or bigger than zero')
        self._current_time = 0
        self._event_time_register = pd.DataFrame(dtype = 'bool') #craete event at time 0 with No event marked as True
        #print(type(self._event_time_register))
        self._event_time_register.loc[0, EVENT_TYPE] = [False for i in range(len(EVENT_TYPE))] #create event at time 0 with No event marked as True
        self._event_time_register.loc[simulation_end_time, EVENT_TYPE] = [False for i in range(len(EVENT_TYPE))] #create event at time simulation end time with No event marked as True
        self.restoration          = restoration
        self._simulation_end_time = simulation_end_time
        self._ending_Event_ignore_time = 0 # in seconds - events in less than this value is ignored
        self._iFirst_time_zero = True
        self._current_time_indexOfIndex = 0
        self.registry = registry
        
    def iContinue(self):
        if self._current_time==0 and self._iFirst_time_zero == True: #So that the other condition happens
            self._iFirst_time_zero = False
  
        else:
            self._current_time               = self.getNextTime()
            self._current_time_indexOfIndex  += 1
            if abs(self._simulation_end_time - self._current_time) <= abs(self._ending_Event_ignore_time):
                print("End_Time_Reached")
                return False
            
            simulation_minimum_time = self.restoration._registry.settings["minimum_simulation_time"]
            minimum_simulation_time_satisfied = self._current_time >= simulation_minimum_time
            consider_last_sequence_termination = self.registry.settings.process['last_sequence_termination']
            consider_node_demand_temination    = self.registry.settings.process['node_demand_temination']
            
            if minimum_simulation_time_satisfied == True:
                if  consider_last_sequence_termination == True:
                    if self.restoration.iRestorationStopTime():
                        print("Last_sequence_termination")
                        return False
                
                if  consider_node_demand_temination == True:
                    if self.iFunctionalityRequirementReached():
                        print("FunctionalityRequirementReached")
                        return False
            
        return True
            
    def getNextTime(self):
        if not self._event_time_register.index.is_monotonic_increasing: # for just in case if the index of event time register is not sorted
            self._event_time_register.sort_index()
                
        if self._event_time_register.index[self._current_time_indexOfIndex]\
            !=self._current_time:
            raise RuntimeError('A possible violation of time in timeline event variables and/or event time registry')
        next_time = self._event_time_register.index[self._current_time_indexOfIndex+1]
        return next_time
    
    def getCurrentStopTime(self):
        return int(self._current_time)
    
    def iCurrentTimeRepairEvent(self):
        return self._event_time_register['rpr'].loc[self._current_time]
    
    def iCurenttimeRestorationEvent(self):
        print("current_time is= "+str(self._current_time) )
        print(self._event_time_register['rst'].loc[self._current_time])
        return self._event_time_register['rst'].loc[self._current_time]
    
    def iCurrentTimeDamageEvent(self):
        return self._event_time_register['dmg'].loc[self._current_time]
    
    def addEventTime(self, event_distinct_time, event_type='dmg'):
        """
        This function is a low-level function to add event type in an already-
        existing event_time in event_time_register. FOR NOW TEH DISTICT TIMES
        CAN BE A LIST OR A LIST. MAYBE IN THE FUTURE WE CAN DECIDE WETHER IT
        SHOULD BE LEFT THE WAY IT IS OR IT SHOULD BE MODIFIED IN A SINGLE
        VARIABLE OR LIST VARIABLE.
        Parameters
        ----------
        event_distinct_time : numpy.float64 or int or float or list
            This variable is either a list or a seriest of data to represent
            time of an specified event
            
        event_type : str, optional
            Evenet type. FOR CURRENT VERSSION AN EVENT COULD BE EIOTHER
            dmg(damage) or rpr(repair). The default is 'dmg'.

        Raises
        ------
        ValueError
            If the input variable for distinct time or the type of event is not
            recognizabe, a Valueerrorr exception is raised
            Also if the damage typeis  not recognized

        Returns
        -------
        None.

        """
        if type(event_distinct_time)!=pd.core.series.Series:
            if type(event_distinct_time) == numpy.float64 or type(event_distinct_time) == int or type(event_distinct_time) == float or type(event_distinct_time) == list:
                event_distinct_time = pd.Series(data=event_distinct_time, dtype="int64")
            else:
                print(type(event_distinct_time))
                raise ValueError('event_distinct_time must be pandas.Series type')
        
        if event_type not in EVENT_TYPE:
            raise ValueError('unrecognized value for event_type')
        
        #check for duplicate in time index. if there is duplicate, we will only change the true and false value in the DataFrame
        temp_to_pop = []
        logger.debug("event distinct time "+ repr(event_distinct_time))
        
        for i, i_time in event_distinct_time.items():
            if i_time in self._event_time_register.index:
                self._event_time_register.loc[i_time, event_type]=True
                self.checkAndAmendTime()
                temp_to_pop.append(i_time)
        logger.debug('temp_to_pop'+repr(temp_to_pop))
        
        for i_time in temp_to_pop:
            ind = event_distinct_time[event_distinct_time==i_time].index[0]
            event_distinct_time.pop(ind)
        
        if len(event_distinct_time) != 0:
            for i, i_time in event_distinct_time.items():
                self._event_time_register.loc[i_time, EVENT_TYPE] = [False for i in range(len(EVENT_TYPE))]
                self._event_time_register.loc[i_time, event_type] = True
            self._event_time_register = self._event_time_register.sort_index()
            self.checkAndAmendTime()
        
    def iEventTypeAt(self, begin_time, event_type):
        """
        Checks if an event type is in event registry at the time of begin_time
        ----------
        begin_time : int
            begining time
        event_type : str
            damage type

        Returns
        -------
        bool
            rResult if such data exist or not

        """
        
        if not begin_time in self._event_time_register.index:
            return False
        if self._event_time_register[event_type].loc[begin_time]:
            return True
        else:
            return False
    
    def checkAndAmendTime(self):
        """
        Checks if the time of event is higher than the sim time.Also checks
        if the the ending event has any thing event(nothings must be true). 
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        
        first_length = len(self._event_time_register.index)
        self._event_time_register = self._event_time_register[self._event_time_register.index <= self._simulation_end_time]
        if first_length > len(self._event_time_register):
            print("here was " + repr(first_length - len(self._event_time_register)) + "amended")
            
        # Sina: I removed teh following code at the tiem for immegration to
        #Pandas 1.5.2. Not only it is not efficient piece of code, but also
        #this nto required. The end time event is already made when teh event
        #table is created.
        #if self._event_time_register[self._event_time_register.index==self._simulation_end_time].empty==True:
            #self._event_time_register=self._event_time_register.append(pd.DataFrame(data = False , index = [self._simulation_end_time], columns = EVENT_TYPE))

    def iFunctionalityRequirementReached(self):
        
        logger.debug("Func: node functionality")
        ratio_criteria = self.registry.settings.process['node_demand_termination_ratio']
        time_window    = self.registry.settings.process.settings['node_demand_termination_time' ]
        stop_time      = self.getCurrentStopTime()
        if self.registry.if_first_event_occured == False:
            return False

        elif self.registry.if_first_event_occured == True:
            if self.registry.result == None:
                return False
            
            #for checking if we have still any leak in the system, since we
            #cannot measure the effect of exessive leak in the LDN
            if stop_time in self.registry.result.node['leak'].index:
                return False
            
            last_pre_event_time  = self.registry.pre_event_demand_met.index.max()
            pre_event_demand     = self.registry.pre_event_demand_met[self.registry.pre_event_demand_met.index <= (last_pre_event_time - time_window)]
            demand_met           = self.registry.result.node['demand']
            begining_time_window = stop_time - time_window 
            demand_met           = demand_met.loc[demand_met.index > begining_time_window]
    
            """
            calcualting requried demand for each dmeand node
            """
            
            #demand_ratio      = self.registry.settings['demand_ratio']
            time_index        = demand_met.index
            req_node_demand   = {}
            default_pattern   = self.registry.wn.options.hydraulic.pattern
            #node_pattern_list = pd.Series(index=self.registry.demand_node_name_list, dtype=str)
            
            #req_node_demand = req_node_demand.transpose()
            
            demand_nodes_list = [self.registry.wn.get_node(node_name) for node_name in self.registry.demand_node_name_list]
            
            if type(default_pattern) != type(None):
                node_pattern_list = [(node.name, node.demand_timeseries_list.pattern_list()[0]) if node.demand_timeseries_list.pattern_list()[0] != None else (node.name, default_pattern) for node in demand_nodes_list]
            else:
                node_pattern_list = [(node.name, node.demand_timeseries_list.pattern_list()[0]) for node in demand_nodes_list if node.demand_timeseries_list.pattern_list()[0] != None]
                
            base_demand_list = [ node.base_demand for node in demand_nodes_list]
            one_time_base_demand = dict(zip(self.registry.demand_node_name_list, base_demand_list))
            req_node_demand = pd.DataFrame.from_dict([one_time_base_demand] * len(time_index))
            req_node_demand.index = time_index

                    
            req_node_demand = pd.DataFrame.from_dict(req_node_demand)

            #node_pattern_list = node_pattern_list.dropna()
            if len(node_pattern_list) > 0:
                node_pattern_list = pd.Series(index=[node_pattern_iter[0] for node_pattern_iter in node_pattern_list], data=[node_pattern_iter[1] for node_pattern_iter in node_pattern_list])
                patterns_list     = node_pattern_list.unique()
                multiplier = pd.DataFrame(index=time_index, columns = list(patterns_list) )
                
                for pattern_name in patterns_list:
                    cur_pattern = self.registry.wn.get_pattern(pattern_name)
                    time_index = time_index.unique()
                    cur_patern_time = [cur_pattern.at(time) for time in iter(time_index)]
                    multiplier.loc[:, pattern_name] = cur_patern_time
    
                for node_name, pattern_name in node_pattern_list.items():
                    cur_node_req_demand = multiplier[pattern_name] * self.registry.wn.get_node(node_name).demand_timeseries_list[0].base_value 
                    cur_node_req_demand.name = node_name
                    cur_node_req_demand=pd.DataFrame(cur_node_req_demand).transpose()
                    req_node_demand = pd.concat([req_node_demand, cur_node_req_demand])

            #print(req_node_demand)
            #raise
            #req_node_demand = req_node_demand.transpose()
            req_node_demand  = req_node_demand.filter(self.registry.demand_node_name_list)
            req_node_demand  = req_node_demand.filter(pre_event_demand.columns)
            demand_met       = demand_met.filter(self.registry.demand_node_name_list)
            demand_met       = demand_met.filter(pre_event_demand.columns)
            demand_met       = demand_met.dropna(axis=1)
            
            pre_event_demand = demand_met.filter(self.registry.demand_node_name_list)
            
            if len(demand_met.columns) < len(pre_event_demand.columns):
                return False
            
            ratio = demand_met.mean() / pre_event_demand.mean()
            mean_of_ratio_satisfied = (ratio >= ratio_criteria).sum() / len(ratio)
            logger.debug("ratio that is= " + repr(mean_of_ratio_satisfied))
            if (ratio >= ratio_criteria).all():
                return True
            else:
                return False
            