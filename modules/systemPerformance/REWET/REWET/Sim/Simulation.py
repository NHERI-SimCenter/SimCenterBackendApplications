import os
import math
import pandas as pd
import numpy as np
from EnhancedWNTR.sim.epanet import EpanetSimulator
from EnhancedWNTR.sim.results import SimulationResults
import wntr


class Hydraulic_Simulation():
    def __init__(self, wn, settings, current_stop_time, worker_rank, prev_isolated_junctions, prev_isolated_links):
        self.wn                    = wn
        self.nne_flow_criteria     = settings.process['nne_flow_limit']
        self.nne_pressure_criteria = settings.process['nne_pressure_limit']
        self.minimum_pressure      = 8
        self.required_pressure     = 25
        self.current_stop_time     = current_stop_time
        self.worker_rank           = worker_rank
        
        self.wn.options.hydraulic.demand_model = "PDA"
        
        temp_folder                = settings['temp_directory']
        if type(temp_folder) != str:
            raise ValueError("temp folder type is not str")
            
        if settings['save_time_step'] == True:
            if temp_folder == '':
                self.temp_directory = str(worker_rank) + "_" + repr(current_stop_time)
            else:
                self.temp_directory = os.path.join(temp_folder, str(worker_rank) + "_" + repr(current_stop_time))
                
        elif settings['save_time_step'] == False:
            if temp_folder == '':
                self.temp_directory = str(worker_rank)
            else:
                self.temp_directory = os.path.join(temp_folder, str(worker_rank))
        else:
            raise ValueError("Unknown value for settings 'save_time_step': " + repr())
        self._prev_isolated_junctions = prev_isolated_junctions
        self._prev_isolated_links     = prev_isolated_links
        
    def removeNonDemandNegativeNodeByPythonMinorLoss(self, maximum_iteration):
        current_stop_time = self.current_stop_time
        minimum_pressure  = self.minimum_pressure
        required_pressure = self.required_pressure
        temp_file_dest    = self.temp_directory
        orginal_c_dict    = {}
        for itrr in range(0, maximum_iteration):
            print(itrr)
            sim    = EpanetSimulator(self.wn)
            self.s = sim
            self._prev_isolated_junctions, self._prev_isolated_links = sim._get_isolated_junctions_and_links(self._prev_isolated_junctions, self._prev_isolated_links)

            sim.manipulateTimeOrder(current_stop_time, current_stop_time)
            rr, i_run_successful = sim.run_sim(file_prefix = temp_file_dest, start_time = current_stop_time, iModified=False)
            new_closed_pipes, ifinish = sim.now_temp_2(rr, self._prev_isolated_links, self._prev_isolated_junctions, self.nne_flow_criteria, self.nne_pressure_criteria)
                            
            if ifinish:
                break
            
            new_pipe_altered_name_list = new_closed_pipes.keys()
            #print(str(list(new_pipe_altered_name_list)[0]) + "   " + str(new_closed_pipes[list(new_pipe_altered_name_list)[0]]))
            new_c_altered = [pipe_name for pipe_name in new_pipe_altered_name_list if pipe_name not in orginal_c_dict]
            for pipe_name in new_c_altered:
                if pipe_name not in orginal_c_dict: #in order not tohange orginal C to something very new in the last oteration
                    orginal_c_dict[pipe_name] = new_closed_pipes[pipe_name]
        return orginal_c_dict
    
    def isolateReservoirs(self, isolated_nodes):
        for reservoir_name, reservoir in self.wn.reservoirs():
            if self.wn._node_reg.get_usage(reservoir_name) == None:
                reservoir._is_isolated = True
                isolated_nodes.add(reservoir_name)
        return isolated_nodes
        
    
    def isolateTanks(self, isolated_nodes):
        for tank_name, tank in self.wn.tanks():
            if self.wn._node_reg.get_usage(tank_name) == None:
                tank._is_isolated = True
                isolated_nodes.add(tank_name)
        return isolated_nodes
    
    def removeNonDemandNegativeNodeByPythonClose(self, maximum_iteration):
        current_stop_time = self.current_stop_time
        minimum_pressure  = self.minimum_pressure
        required_pressure = self.required_pressure
        temp_file_dest    = self.temp_directory
        self.closed_pipes = {}
        for itrr in range(0, maximum_iteration):
            print(itrr)
            sim    = EpanetSimulator(self.wn)
            self._prev_isolated_junctions, self._prev_isolated_links = sim._get_isolated_junctions_and_links(self._prev_isolated_junctions, self._prev_isolated_links)
            sim.manipulateTimeOrder(current_stop_time, current_stop_time)
            rr, i_run_successful = sim.run_sim(file_prefix = temp_file_dest, start_time = current_stop_time, iModified=False)
            new_closed_pipes, ifinish = sim.closePipeNNN(rr, self._prev_isolated_links, self._prev_isolated_junctions, self.nne_flow_criteria, self.nne_pressure_criteria)
                            
            if ifinish:
                break
            
            new_pipe_altered_name_list = new_closed_pipes.keys()
            new_c_altered = [pipe_name for pipe_name in new_pipe_altered_name_list if pipe_name not in self.closed_pipes]
            for pipe_name in new_c_altered:
                self.closed_pipes[pipe_name] = new_closed_pipes[pipe_name]
        #self.closed_pipes = orginal_c_dict
        #return orginal_c_dict
    
    def rollBackPipeMinorLoss(self, altered_pipes):
        for pipe_name in altered_pipes:
            self.wn.get_link(pipe_name).minor_loss = altered_pipes[pipe_name]
    
    def rollBackPipeClose(self):
        altered_pipes = self.closed_pipes
        for pipe_name in altered_pipes:
            pipe = self.wn.get_link(pipe_name)
            pipe.initial_status = altered_pipes[pipe_name]
        
    def performSimulation(self, next_event_time, iModified):
        current_stop_time = self.current_stop_time
        minimum_pressure  = self.minimum_pressure
        required_pressure = self.required_pressure
        temp_file_dest    = self.temp_directory
        sim = EpanetSimulator(self.wn)
        #self.s=sim
        self._prev_isolated_junctions, self._prev_isolated_links = sim._get_isolated_junctions_and_links(self._prev_isolated_junctions, self._prev_isolated_links)
        self._prev_isolated_junctions = self.isolateReservoirs(self._prev_isolated_junctions)
        self._prev_isolated_junctions = self.isolateTanks(self._prev_isolated_junctions)
        print('***********')
        print(len(self._prev_isolated_junctions))
        print(len(self._prev_isolated_links))
        print('-----------')
        sim.manipulateTimeOrder(current_stop_time, next_event_time) #, change_time_step=True, min_correction_time_step=self._min_correction_time)
        rr, i_run_successful = sim.run_sim(file_prefix = temp_file_dest, start_time = current_stop_time,iModified=iModified)
        return rr, i_run_successful
    
    def estimateRun(self, next_event_time, iModified):
        current_stop_time = self.current_stop_time
        minimum_pressure  = self.minimum_pressure
        required_pressure = self.required_pressure
        
        sim = EpanetSimulator(self.wn)
        duration = self.wn.options.time.duration
        report_time_step  = self.wn.options.time.report_timestep
        sim.manipulateTimeOrder(current_stop_time, current_stop_time)
        
        temp_file_dest    = self.temp_directory
        self._prev_isolated_junctions, self._prev_isolated_links = sim._get_isolated_junctions_and_links(self._prev_isolated_junctions, self._prev_isolated_links)
        self._prev_isolated_junctions = self.isolateReservoirs(self._prev_isolated_junctions)
        self._prev_isolated_junctions = self.isolateTanks(self._prev_isolated_junctions)
        rr, i_run_successful = sim.run_sim(file_prefix= temp_file_dest, start_time = current_stop_time, iModified=iModified)
        self.wn.options.time.duration = duration
        self.wn.options.time.report_timestep = report_time_step
        rr = self.approximateNewResult(rr, current_stop_time, next_event_time, 0)
        
        return rr, i_run_successful
    
    def estimateWithoutRun(self, result, next_event_time):
        current_stop_time = self.current_stop_time
        minimum_pressure  = self.minimum_pressure
        required_pressure = self.required_pressure
        
        time = result.node["demand"].index.to_list()
        unreliable_time_list = result.maximum_trial_time
        time.reverse()
        
        last_valid_time = int(-1)
        for checked_last_time in time:
            if checked_last_time not in unreliable_time_list:
                last_valid_time = checked_last_time
                break
        
        if last_valid_time == int(-1):
            raise ValueError("Last reliabale tiem is not found")
        
        time_step = min(self.wn.options.time.hydraulic_timestep, self.wn.options.time.report_timestep)
        time_step = int(time_step)
        end_time  = next_event_time
        end_time  = int(end_time)
        #result_node_head     = {}
        #result_node_demand   = {}
        #result_node_pressure = {}
        result_node_status   = {}
        result_node_setting  = {}
        result_node_flowrate = {}
        
        sim = EpanetSimulator(self.wn)
        
        self._prev_isolated_junctions, self._prev_isolated_links = sim._get_isolated_junctions_and_links(self._prev_isolated_junctions, self._prev_isolated_links)
        self._prev_isolated_junctions = self.isolateReservoirs(self._prev_isolated_junctions)
        self._prev_isolated_junctions = self.isolateTanks(self._prev_isolated_junctions)
        
        #available_node_list = [node_name for node_name in self.wn.node_name_list if self.wn.get_node(node_name)._is_isolated == False]
        #available_link_list = [link_name for link_name in self.wn.link_name_list if self.wn.get_link(link_name)._is_isolated == False]
        
        available_node_list = [node_name for node_name in self.wn.node_name_list]
        available_link_list = [link_name for link_name in self.wn.link_name_list]
        
        available_node_list = [node_name for node_name in available_node_list if node_name in result.node['demand'].columns]
        available_link_list = [link_name for link_name in available_link_list if link_name in result.link['flowrate'].columns]
        
        result_node_head     = pd.DataFrame(columns= available_node_list)
        result_node_demand   = pd.DataFrame(columns= available_node_list)
        result_node_pressure = pd.DataFrame(columns= available_node_list)
        result_node_leak     = pd.DataFrame(columns= available_node_list)
        
        result_link_status   = pd.DataFrame(columns= available_link_list)
        result_link_setting  = pd.DataFrame(columns= available_link_list)
        result_link_flowrate = pd.DataFrame(columns= available_link_list)
        
        first_step = True
        for time_step_iter in range(current_stop_time, end_time+1, time_step):
            #print(result.node['head'].loc[last_valid_time, available_node_list])
            result_node_head.loc[time_step_iter, available_node_list]     = result.node['head'].loc[last_valid_time, available_node_list]
            result_node_demand.loc[time_step_iter, available_node_list]   = result.node['demand'].loc[last_valid_time, available_node_list]
            result_node_pressure.loc[time_step_iter, available_node_list] = result.node['pressure'].loc[last_valid_time, available_node_list]
            result_node_leak.loc[time_step_iter, available_node_list]     = result.node['leak'].loc[last_valid_time, result.node['leak'].columns]
            
            result_link_status.loc[time_step_iter]   = result.link['status'].loc[last_valid_time, available_link_list]
            result_link_setting.loc[time_step_iter]  = result.link['setting'].loc[last_valid_time, available_link_list]
            result_link_flowrate.loc[time_step_iter] = result.link['flowrate'].loc[last_valid_time, available_link_list]
            #print("---------------")
            #print(result_node_head)
            #print("---------------")
            if first_step==True:
                first_step = False
            else:
                self.updateTankHeadsAndPressure(result_node_demand, result_node_head, result_node_pressure, time_step_iter, time_step)
            
        
        rr = SimulationResults()
        result_node_head     = result_node_head.sort_index()
        result_node_demand   = result_node_demand.sort_index()
        result_node_pressure = result_node_pressure.sort_index()
        result_node_status   = pd.DataFrame.from_dict(result_node_status).sort_index()
        result_node_setting  = pd.DataFrame.from_dict(result_node_setting).sort_index()
        result_node_flowrate = pd.DataFrame.from_dict(result_node_flowrate).sort_index()
        
        rr.node = {'head': result_node_head, 'demand': result_node_demand, 'pressure': result_node_pressure, "leak": result_node_leak}
        rr.link = {'status': result_link_status, 'setting': result_link_setting, 'flowrate': result_link_flowrate}
        return rr, True
    
    def updateTankHeadsAndPressure(self, demand, head, pressure, sim_time, time_step): # Addapted from the latest version of wntrfr. Courtessy of WNTR: https://github.com/USEPA/WNTR
        """
        Parameters
        ----------
        wn: wntrfr.network.WaterNetworkModel
        """
        dt = time_step
        #print(sim_time)
        demand_na   = demand.loc[sim_time].isna()
        head_na     = head.loc[sim_time].isna()
        pressure_na = pressure.loc[sim_time].isna()
        
        for tank_name, tank in self.wn.tanks():
            
            #checks if the node is isolated.
            if tank._is_isolated == True:
                continue
            
            #checks of this node has been isolated at the last valid time. if
            #so, ignores this node, even though it is not isolated at this time
            #print(sim_time)
            #print(demand_na.loc[tank_name])
            #print(demand.loc[sim_time, tank_name])
            if demand_na.loc[tank_name] or head_na.loc[tank_name] or pressure_na.loc[tank_name]:
                continue
            
            #With formers checks, this "if statement" must not be needed.
            #Just leave it here for now
            
            if tank_name in demand.columns:
                q_net = demand.loc[sim_time, tank_name]
            else:
                q_net = 0.0
                
            dV = q_net * dt
            
            previous_head  = head.loc[sim_time, tank_name]
            if tank.vol_curve is None:    
                delta_h   = 4.0 * dV / (math.pi * tank.diameter ** 2)
                new_level =  previous_head + delta_h - tank.elevation
            else:
                vcurve = np.array(tank.vol_curve.points)
                level_x = vcurve[:,0]
                volume_y = vcurve[:,1]
                
                previous_level = previous_head - tank.elevation
                                
                V0 = np.interp(previous_level,level_x,volume_y)
                V1 = V0 + dV
                new_level = np.interp(V1,volume_y,level_x)
                delta_h = new_level - previous_level
            
            #checks if the new levels and head are within the tanks limit.
            #It ignores the possibility of tank overflow and does not alter the
            #tank demand.
            if new_level < tank.min_level:
                new_level = tank.min_level
                new_head  = tank.elevation + tank.min_level
            elif new_level > tank.max_level:
                new_level = tank.max_level
                new_head  = tank.elevation + tank.max_level
                
            new_head = previous_head + delta_h
            head.loc[sim_time, tank_name    ] = new_head
            pressure.loc[sim_time, tank_name] = new_head - tank.elevation
           
    
    def approximateNewResult(self, rr, current_stop_time, end_time, little_time_step):
        time_step = min(self.wn.options.time.hydraulic_timestep, self.wn.options.time.report_timestep)
        current_stop_time = int(current_stop_time)
        end_time          = int(end_time)
        time_step         = int(time_step)
        not_isolated_tanks = [tank_name for tank_name, tank in self.wn.tanks()   if tank._is_isolated == False]
        #isolated_tanks     = [tank_name for tank_name in self.tanks_name_list if tank_name in self._prev_isolated_junctions]
        #isolated_nodes     = [node_name for node_name in self.node_name_list  if node_name in self._prev_isolated_junctions]
        tank_heads = rr.node['head'][not_isolated_tanks]
        #tank_demands=rr.node['demand'][self.wn.tank_name_list]
        if little_time_step==0:
            tank_elevation_list  = [self.wn.get_node(e).elevation for e in not_isolated_tanks]
            tank_min_level_list  = [self.wn.get_node(l).min_level for l in not_isolated_tanks]
            tank_max_level_list  = [self.wn.get_node(l).max_level for l in not_isolated_tanks]
            
            tanks_min_heads = [tank_elevation_list[i]+tank_min_level_list[i] for i in range(len(tank_elevation_list))]
            tanks_max_heads = [tank_elevation_list[i]+tank_max_level_list[i] for i in range(len(tank_elevation_list))]
            
            tank_heads_diff = rr.node['demand'][not_isolated_tanks]
            tank_heads_diff = tank_heads_diff.iloc[-1]
            
            tanks_min_heads = pd.Series(tanks_min_heads, not_isolated_tanks)
            tanks_max_heads = pd.Series(tanks_max_heads, not_isolated_tanks)
            
            print(current_stop_time)
            print(time_step)
            print(end_time)
            for time_step_iter in range(current_stop_time+time_step, end_time+1, time_step):
                rr.node['head'].loc[time_step_iter]     = rr.node['head'].loc[current_stop_time]
                rr.node['demand'].loc[time_step_iter]   = rr.node['demand'].loc[current_stop_time]
                rr.node['pressure'].loc[time_step_iter] = rr.node['pressure'].loc[current_stop_time]
                rr.link['status'].loc[time_step_iter]   = rr.link['status'].loc[current_stop_time]
                rr.link['setting'].loc[time_step_iter]  = rr.link['setting'].loc[current_stop_time]
                rr.link['flowrate'].loc[time_step_iter] = rr.link['flowrate'].loc[current_stop_time]
                
                new_tank_heads  = tank_heads.loc[current_stop_time]+(tank_heads_diff * (time_step_iter-current_stop_time) )
                
                under_min_heads = new_tank_heads[new_tank_heads<tanks_min_heads]
                over_max_heads  = new_tank_heads[new_tank_heads>tanks_max_heads]
                        
                new_tank_heads.loc[under_min_heads.index] = tanks_min_heads.loc[under_min_heads.index]
                new_tank_heads.loc[over_max_heads.index]  = tanks_min_heads.loc[over_max_heads.index]
        
                rr.node['head'].loc[time_step_iter, new_tank_heads.index] = new_tank_heads.to_list()
                
                #Future updates: updating  tank levels based on Newer version of WNTR for tansks with curves
        else:
            tank_elevation_list  = [self.wn.get_node(e).elevation for e in not_isolated_tanks]
            tank_min_level_list  = [self.wn.get_node(l).min_level for l in not_isolated_tanks]
            tank_max_level_list  = [self.wn.get_node(l).max_level for l in not_isolated_tanks]
            
            tanks_min_heads = [tank_elevation_list[i]+tank_min_level_list[i] for i in range(len(tank_elevation_list))]
            tanks_max_heads = [tank_elevation_list[i]+tank_max_level_list[i] for i in range(len(tank_elevation_list))]
            
            tank_heads_diff = tank_heads.loc[current_stop_time + little_time_step]-tank_heads.loc[current_stop_time]
            
            tanks_min_heads = pd.Series(tanks_min_heads, not_isolated_tanks)
            tanks_max_heads = pd.Series(tanks_max_heads, not_isolated_tanks)
            #print(repr(current_stop_time)+'  '+repr(time_step)+'  '+repr(end_time)+'  '+repr(time_step)+'  ')
            for time_step_iter in range(int(current_stop_time+time_step), int(end_time+1), int(time_step)):
                #print(time_step_iter)
                new_tank_heads  = tank_heads.loc[current_stop_time]+(tank_heads_diff * (time_step_iter-current_stop_time) / little_time_step)
                
                under_min_heads = new_tank_heads[new_tank_heads<tanks_min_heads]
                over_max_heads  = new_tank_heads[new_tank_heads>tanks_max_heads]
                        
                new_tank_heads.loc[under_min_heads.index] = tanks_min_heads.loc[under_min_heads.index]
                new_tank_heads.loc[over_max_heads.index]  = tanks_min_heads.loc[over_max_heads.index]
        
                rr.node['head'].loc[time_step_iter] = rr.node['head'].loc[current_stop_time]
                rr.node['head'].loc[time_step_iter, new_tank_heads.columns] = new_tank_heads
                
                rr.node['demand'].loc[time_step_iter] = rr.node['demand'].loc[current_stop_time]
                rr.node['pressure'].loc[time_step_iter] = rr.node['pressure'].loc[current_stop_time]
                rr.link['status'].loc[time_step_iter] = rr.link['status'].loc[current_stop_time]
                rr.link['setting'].loc[time_step_iter] = rr.link['setting'].loc[current_stop_time]
            
            rr.node['head']=rr.node['head'].drop(current_stop_time+little_time_step)
            rr.node['demand']=rr.node['demand'].drop(current_stop_time+little_time_step)
            rr.node['pressure']=rr.node['pressure'].drop(current_stop_time+little_time_step)
            rr.link['status']=rr.link['status'].drop(current_stop_time+little_time_step)
            rr.link['setting']=rr.link['setting'].drop(current_stop_time+little_time_step)
            
        return rr