"""Created on Wed Apr  8 20:19:10 2020

@author: snaeimi
"""

import logging
import os
import pickle
import sys

import Damage
import EnhancedWNTR.network.model
import pandas as pd
import wntrfr
from EnhancedWNTR.sim.results import SimulationResults
from Sim.Simulation import Hydraulic_Simulation
from timeline import Timeline
from wntrfr.network.model import LinkStatus

# from wntrplus import WNTRPlus
from wntrfr.utils.ordered_set import OrderedSet

logger = logging.getLogger(__name__)


class StochasticModel:
    def __init__(
        self,
        water_network,
        damage_model,
        registry,
        simulation_end_time,
        restoration,
        mode='PDD',
        i_restoration=True,
    ):
        if (
            type(water_network) != wntrfr.network.model.WaterNetworkModel
            and type(water_network) != EnhancedWNTR.network.model.WaterNetworkModel
        ):
            raise ValueError(
                'Water_network model is not legitimate water Network Model'
            )
        if type(damage_model) != Damage.Damage:
            raise ValueError('damage_model is not a ligitimate Damage Model')
        self.wn = water_network
        self.damage_model = damage_model
        self._simulation_time = simulation_end_time
        self.timeline = Timeline(simulation_end_time, restoration, registry)
        damage_distict_time = self.damage_model.get_damage_distinct_time()
        self.timeline.addEventTime(damage_distict_time)
        self.timeline.checkAndAmendTime()

        self.simulation_mode = None
        if mode == 'PDD' or mode == 'DD':
            self.simulation_mode = mode
        else:
            self.simulation_mode = 'PDD'
        self._linear_result = registry.result
        self.registry = registry
        # self.wp                      = WNTRPlus(restoration._registry)
        self.restoration = restoration
        self._min_correction_time = 900
        self.simulation_time = 0
        self.restoration_time = 0
        self.iRestoration = i_restoration
        self._prev_isolated_junctions = OrderedSet()
        self._prev_isolated_links = OrderedSet()
        self.first_leak_flag = True

    def runLinearScenario(self, damage, settings, worker_rank=None):
        """Runs a simple linear analysis of water damage scenario
        Parameters

        Water Network object (WN) shall not be altered in any object except restoration
        ----------
        damage : Damage Object

        Returns
        -------
        Result.

        """
        while self.timeline.iContinue():
            sys.stdout.flush()
            current_stop_time = self.timeline.getCurrentStopTime()
            print('--------------------------------------')
            print('At stop Time: ' + repr(current_stop_time / 3600))
            # =============================================================================
            # Restoration event Block
            if (
                self.timeline.iCurenttimeRestorationEvent()
                and self.iRestoration == True
            ):
                logger.debug('\t Restoration Event ')

                event_time_list = self.restoration.perform_action(
                    self.wn, current_stop_time
                )

                self.timeline.addEventTime(event_time_list, event_type='rst')

            # =============================================================================
            #           Damage (earthquake) event block
            if self.timeline.iCurrentTimeDamageEvent():
                self.ttemp = pd.DataFrame()
                self.registry.if_first_event_occured = True
                logger.debug('\t DAMAGE EVENT')
                # pipe_list = self.restoration.getPipeListForHydraulicSignificant()
                if len(self.restoration.getHydSigPipeList()) > 0:
                    last_demand_node_pressure = None
                    pipe_list = damage.getPipeDamageListAt(current_stop_time)
                    for pipe_name in pipe_list:
                        if last_demand_node_pressure is None:
                            time_index = self.registry.result.node['pressure'].index
                            time_index = list(
                                set(time_index)
                                - set(self.registry.result.maximum_trial_time)
                            )
                            time_index.sort()
                            if len(time_index) > 0:
                                time_index = time_index[-1]
                            else:
                                self.registry.hydraulic_significance.loc[
                                    pipe_name
                                ] = -1000
                                continue
                                time_index = current_stop_time
                            demand_node_list = self.registry.demand_node_name_list
                            demand_node_list = set(demand_node_list).intersection(
                                self.registry.result.node['pressure'].columns
                            )
                            last_demand_node_pressure = self.registry.result.node[
                                'pressure'
                            ].loc[time_index, list(demand_node_list)]
                            last_demand_node_pressure.loc[
                                last_demand_node_pressure[
                                    last_demand_node_pressure < 0
                                ].index
                            ] = 0
                        pipe = self.wn.get_link(pipe_name)
                        initial_pipe_status = pipe.initial_status
                        if initial_pipe_status == LinkStatus.Closed:
                            continue

                        pipe.initial_status = LinkStatus.Closed
                        hyd_sim = Hydraulic_Simulation(
                            self.wn,
                            settings,
                            current_stop_time,
                            worker_rank,
                            self._prev_isolated_junctions,
                            self._prev_isolated_links,
                        )
                        self.hyd_temp = hyd_sim
                        duration = self.wn.options.time.duration
                        report_time_step = self.wn.options.time.report_timestep
                        try:  # Run with modified EPANET V2.2
                            print('Performing method 1')
                            rr, i_run_successful = hyd_sim.performSimulation(
                                current_stop_time, True
                            )
                            if current_stop_time in rr.maximum_trial_time:
                                pass
                                # self.registry.hydraulic_significance.loc[pipe_name] = -20000
                                # pipe.initial_status = initial_pipe_status
                                # self._prev_isolated_junctions = hyd_sim._prev_isolated_junctions
                                # self._prev_isolated_links     = hyd_sim._prev_isolated_links
                                # continue
                            demand_node_list = self.registry.demand_node_name_list
                            demand_node_list = set(demand_node_list).intersection(
                                rr.node['pressure'].columns
                            )
                            new_node_pressure = rr.node['pressure'].loc[
                                current_stop_time, list(demand_node_list)
                            ]
                            new_node_pressure.loc[
                                new_node_pressure[new_node_pressure < 0].index
                            ] = 0

                            hydraulic_impact = (
                                last_demand_node_pressure - new_node_pressure
                            ).mean()
                            self.registry.hydraulic_significance.loc[pipe_name] = (
                                hydraulic_impact
                            )

                        except Exception as epa_err_1:
                            raise
                            if epa_err_1.args[0] == 'EPANET Error 110':
                                print('Method 1 failed. Performing method 2')
                                self.wn.options.time.duration = duration
                                self.wn.options.time.report_timestep = (
                                    report_time_step
                                )
                                self.registry.hydraulic_significance.loc[
                                    pipe_name
                                ] = -1
                        pipe.initial_status = initial_pipe_status
                        self._prev_isolated_junctions = (
                            hyd_sim._prev_isolated_junctions
                        )
                        self._prev_isolated_links = hyd_sim._prev_isolated_links
                        self.wn.options.time.duration = duration
                        self.wn.options.time.report_timestep = report_time_step
                damage.applyPipeDamages(self.wn, current_stop_time)
                damage.applyNodalDamage(self.wn, current_stop_time)
                damage.applyPumpDamages(self.wn, current_stop_time)
                damage.applyTankDamages(self.wn, current_stop_time)

                if self.iRestoration == True:
                    event_time_list = self.restoration.initialize(
                        self.wn, current_stop_time
                    )  # starts restoration
                    self.timeline.addEventTime(event_time_list, event_type='rst')

            # =============================================================================
            #           This is for updatng the pipe damage log
            if settings['record_damage_table_logs'] == True:
                self.restoration._registry.updatePipeDamageTableTimeSeries(
                    current_stop_time
                )
                self.restoration._registry.updateNodeDamageTableTimeSeries(
                    current_stop_time
                )
            # =============================================================================
            #           runing the model
            next_event_time = self.timeline.getNextTime()
            logger.debug('next event time is: ' + repr(next_event_time))

            self.wn.implicitLeakToExplicitReservoir(self.registry)

            print('***** Running hydraulic *****')

            if type(worker_rank) != str:
                worker_rank = str(worker_rank)

            hyd_sim = Hydraulic_Simulation(
                self.wn,
                settings,
                current_stop_time,
                worker_rank,
                self._prev_isolated_junctions,
                self._prev_isolated_links,
            )
            self.hyd_temp = hyd_sim
            duration = self.wn.options.time.duration
            report_time_step = self.wn.options.time.report_timestep
            try:  # Run with modified EPANET V2.2
                print('Performing method 1')
                rr, i_run_successful = hyd_sim.performSimulation(
                    next_event_time, True
                )
            except Exception as epa_err_1:
                if epa_err_1.args[0] == 'EPANET Error 110':
                    print('Method 1 failed. Performing method 2')
                    try:  # Remove Non-Demand Node by Python-Side iterative algorythm with closing
                        # self.wn.options.time.duration        = duration
                        # self.wn.options.time.report_timestep = report_time_step
                        # hyd_sim.removeNonDemandNegativeNodeByPythonClose(1000)
                        # rr, i_run_successful = hyd_sim.performSimulation(next_event_time, False)
                        # hyd_sim.rollBackPipeClose()
                        raise
                    except Exception as epa_err_2:
                        if True:  # epa_err_2.args[0] == 'EPANET Error 110':
                            try:  # Extend result from teh reult at the begining of teh time step with modified EPANET V2.2
                                # print("Method 2 failed. Performing method 3")
                                self.wn.options.time.duration = duration
                                self.wn.options.time.report_timestep = (
                                    report_time_step
                                )
                                # hyd_sim.rollBackPipeClose()
                                rr, i_run_successful = hyd_sim.estimateRun(
                                    next_event_time, True
                                )
                            except Exception as epa_err_3:
                                if epa_err_3.args[0] == 'EPANET Error 110':
                                    print('Method 3 failed. Performing method 4')
                                    try:  # Extend result from teh reult at the begining of teh time step with modified EPANET V2.2
                                        self.wn.options.time.duration = duration
                                        self.wn.options.time.report_timestep = (
                                            report_time_step
                                        )
                                        rr, i_run_successful = (
                                            hyd_sim.performSimulation(
                                                next_event_time, False
                                            )
                                        )
                                    except Exception as epa_err_4:
                                        if epa_err_4.args[0] == 'EPANET Error 110':
                                            try:
                                                self.wn.options.time.duration = (
                                                    duration
                                                )
                                                self.wn.options.time.report_timestep = report_time_step
                                                print(
                                                    'Method 4 failed. Performing method 5'
                                                )
                                                # Extend result from teh reult at the begining of teh time step with modified EPANET V2.2
                                                rr, i_run_successful = (
                                                    hyd_sim.estimateRun(
                                                        next_event_time, False
                                                    )
                                                )
                                            except Exception as epa_err_5:
                                                if (
                                                    epa_err_5.args[0]
                                                    == 'EPANET Error 110'
                                                ):
                                                    try:
                                                        print(
                                                            'Method 5 failed. Performing method 6'
                                                        )
                                                        self.wn.options.time.duration = duration
                                                        self.wn.options.time.report_timestep = report_time_step
                                                        rr, i_run_successful = (
                                                            hyd_sim.estimateWithoutRun(
                                                                self._linear_result,
                                                                next_event_time,
                                                            )
                                                        )
                                                    except Exception as epa_err_6:
                                                        print(
                                                            'ERROR in rank='
                                                            + repr(worker_rank)
                                                            + ' and time='
                                                            + repr(current_stop_time)
                                                        )
                                                        raise epa_err_6
                                                else:
                                                    raise epa_err_5
                                        else:
                                            raise epa_err_4
                                else:
                                    raise epa_err_3
                        else:
                            raise epa_err_2
                else:
                    raise epa_err_1
            self._prev_isolated_junctions = hyd_sim._prev_isolated_junctions
            self._prev_isolated_links = hyd_sim._prev_isolated_links
            print(
                '***** Finish Running at time '
                + repr(current_stop_time)
                + '  '
                + repr(i_run_successful)
                + ' *****'
            )

            if i_run_successful == False:
                continue
            self.wn.updateWaterNetworkModelWithResult(rr, self.restoration._registry)

            self.KeepLinearResult(
                rr,
                self._prev_isolated_junctions,
                node_attributes=['pressure', 'head', 'demand', 'leak'],
                link_attributes=['status', 'setting', 'flowrate'],
            )
            if self.registry.settings['limit_result_file_size'] > 0:
                self.dumpPartOfResult()
            # self.wp.unlinkBreackage(self.registry)
            self.wn.resetExplicitLeak()

        # =============================================================================
        # self.resoration._registry.updateTankTimeSeries(self.wn, current_stop_time)
        self.restoration._registry.updateRestorationIncomeWaterTimeSeries(
            self.wn, current_stop_time
        )

        return self._linear_result

    def KeepLinearResult(
        self,
        result,
        isolated_nodes,
        node_attributes=None,
        link_attributes=None,
        iCheck=False,
    ):  # , iNeedTimeCorrection=False, start_time=None):
        if self.registry.if_first_event_occured == False:
            self.registry.pre_event_demand_met = (
                self.registry.pre_event_demand_met.append(result.node['demand'])
            )

        # if node_attributes == None:
        # node_attributes = ['pressure','head','demand','quality']
        # if link_attributes == None:
        # link_attributes = ['linkquality', 'flowrate', 'headloss', 'velocity', 'status', 'setting', 'frictionfact', 'rxnrate']

        just_initialized_flag = False
        if self._linear_result == None:
            just_initialized_flag = True
            self._linear_result = result

            self.restoration._registry.result = self._linear_result
            node_result_type_elimination_list = set(result.node.keys()) - set(
                node_attributes
            )
            link_result_type_elimination_list = set(result.link.keys()) - set(
                link_attributes
            )

            for node_result_type in node_result_type_elimination_list:
                self._linear_result.node.pop(node_result_type)

            for link_result_type in link_result_type_elimination_list:
                self._linear_result.link.pop(link_result_type)

            self._linear_result.node['leak'] = pd.DataFrame(dtype=float)

        active_pipe_damages = self.restoration._registry.active_pipe_damages

        temp_active = active_pipe_damages.copy()
        for virtual_demand_node in active_pipe_damages:
            if (
                virtual_demand_node in isolated_nodes
                or active_pipe_damages[virtual_demand_node] in isolated_nodes
            ):
                temp_active.pop(virtual_demand_node)

        virtual_demand_nodes = list(temp_active.keys())
        real_demand_nodes = list(temp_active.values())

        if len(temp_active) > 0:
            # this must be here in the case that a node that is not isolated at
            # this step does not have a result. This can happen if the result is
            # simulated without run.. For example, in the latest vallid result
            # some nodes were isolated, but not in the current run.
            available_nodes_in_current_result = result.node[
                'demand'
            ].columns.to_list()
            not_available_virtual_node_names = set(virtual_demand_nodes) - set(
                available_nodes_in_current_result
            )
            if len(not_available_virtual_node_names):
                not_available_real_node_names = [
                    temp_active[virtual_node_name]
                    for virtual_node_name in not_available_virtual_node_names
                ]
                virtual_demand_nodes = (
                    set(virtual_demand_nodes) - not_available_virtual_node_names
                )
                real_demand_nodes = set(real_demand_nodes) - set(
                    not_available_real_node_names
                )
                virtual_demand_nodes = list(virtual_demand_nodes)
                real_demand_nodes = list(real_demand_nodes)

            result.node['demand'][real_demand_nodes] = result.node['demand'][
                virtual_demand_nodes
            ]
            result.node['demand'].drop(virtual_demand_nodes, axis=1, inplace=True)

        active_nodal_damages = self.restoration._registry.active_nodal_damages
        temp_active = active_nodal_damages.copy()

        for virtual_demand_node in active_nodal_damages:
            if (
                virtual_demand_node in isolated_nodes
                or temp_active[virtual_demand_node] in isolated_nodes
            ):
                temp_active.pop(virtual_demand_node)

        virtual_demand_nodes = list(temp_active.keys())
        real_demand_nodes = list(temp_active.values())

        if len(temp_active) > 0:
            # this must be here in the case that a node that is not isolated at
            # this step has not result. This can happen if the result is being
            # simulated without run.. For example, in the latest vallid result
            # some nodes were isolated, but not in the current run.
            available_nodes_in_current_result = result.node[
                'demand'
            ].columns.to_list()
            not_available_virtual_node_names = set(virtual_demand_nodes) - set(
                available_nodes_in_current_result
            )
            if len(not_available_virtual_node_names):
                not_available_real_node_names = [
                    temp_active[virtual_node_name]
                    for virtual_node_name in not_available_virtual_node_names
                ]
                virtual_demand_nodes = (
                    set(virtual_demand_nodes) - not_available_virtual_node_names
                )
                real_demand_nodes = set(real_demand_nodes) - set(
                    not_available_real_node_names
                )
                virtual_demand_nodes = list(virtual_demand_nodes)
                real_demand_nodes = list(real_demand_nodes)

            non_isolated_pairs = dict(zip(virtual_demand_nodes, real_demand_nodes))
            result.node['leak'] = result.node['demand'][virtual_demand_nodes].rename(
                non_isolated_pairs, axis=1
            )

        if just_initialized_flag == False:
            self._linear_result.maximum_trial_time.extend(result.maximum_trial_time)

            saved_max_time = self._linear_result.node[
                list(self._linear_result.node.keys())[0]
            ].index.max()
            to_be_saved_min_time = result.node[
                list(result.node.keys())[0]
            ].index.min()
            if (
                abs(to_be_saved_min_time - saved_max_time) != 0
            ):  # >= min(self.wn.options.time.hydraulic_timestep, self.wn.options.time.report_timestep):
                # logger.error(repr(to_be_saved_min_time)+ '  ' + repr(saved_max_time))
                raise ValueError(
                    'saved result and to be saved result are not the same. '
                    + repr(saved_max_time)
                    + '   '
                    + repr(to_be_saved_min_time)
                )
            for att in node_attributes:
                if len(active_nodal_damages) == 0 and att == 'leak':
                    continue
                _leak_flag = False

                leak_first_time_result = None
                if (
                    att == 'leak' and 'leak' in result.node
                ):  # the second condition is not needed. It's there only for assurance
                    former_nodes_list = set(self._linear_result.node['leak'].columns)
                    to_add_nodes_list = set(result.node[att].columns)
                    complete_result_node_list = to_add_nodes_list - former_nodes_list
                    if len(complete_result_node_list) > 0:
                        _leak_flag = True

                    leak_first_time_result = result.node['leak'][
                        complete_result_node_list
                    ].iloc[0]

                if att in result.node:
                    result.node[att].drop(result.node[att].index[0], inplace=True)
                    self._linear_result.node[att] = self._linear_result.node[
                        att
                    ].append(result.node[att])

                if _leak_flag:
                    self._linear_result.node['leak'].loc[
                        leak_first_time_result.name, leak_first_time_result.index
                    ] = leak_first_time_result
                    self._linear_result.node['leak'] = self._linear_result.node[
                        'leak'
                    ].sort_index()

            for att in link_attributes:
                result.link[att].drop(result.link[att].index[0], inplace=True)
                self._linear_result.link[att] = self._linear_result.link[att].append(
                    result.link[att]
                )

    def dumpPartOfResult(self):
        limit_size = self.registry.settings['limit_result_file_size']
        limit_size_byte = limit_size * 1024 * 1024

        total_size = 0

        for att in self._linear_result.node:
            att_size = sys.getsizeof(self._linear_result.node[att])
            total_size += att_size

        for att in self._linear_result.link:
            att_size = sys.getsizeof(self._linear_result.link[att])
            total_size += att_size

        print('total size= ' + repr(total_size / 1024 / 1024))

        if total_size >= limit_size_byte:
            dump_result = SimulationResults()
            dump_result.node = {}
            dump_result.link = {}
            for att in self._linear_result.node:
                # just to make sure. it obly add tens of micro seconds for each
                # att

                self._linear_result.node[att].sort_index(inplace=True)
                att_result = self._linear_result.node[att]
                if att_result.empty:
                    continue
                # first_time_index = att_result.index[0]
                last_valid_time = []
                att_time_index = att_result.index.to_list()
                last_valid_time = [
                    cur_time
                    for cur_time in att_time_index
                    if cur_time not in self._linear_result.maximum_trial_time
                ]
                last_valid_time.sort()

                if len(last_valid_time) > 0:
                    last_valid_time = last_valid_time[-2]
                else:
                    print(att_time_index)
                    last_valid_time = att_time_index[-2]

                dump_result.node[att] = att_result.loc[:last_valid_time]
                last_valid_time_index = att_result.index.searchsorted(
                    last_valid_time
                )
                self._linear_result.node[att].drop(
                    att_result.index[: last_valid_time_index + 1], inplace=True
                )

            for att in self._linear_result.link:
                # just to make sure. it obly add tens of micro seconds for each
                # att
                self._linear_result.link[att].sort_index(inplace=True)
                att_result = self._linear_result.link[att]
                if att_result.empty:
                    continue
                # first_time_index = att_result.index[0]
                last_valid_time = []
                att_time_index = att_result.index.to_list()
                last_valid_time = [
                    cur_time
                    for cur_time in att_time_index
                    if cur_time not in self._linear_result.maximum_trial_time
                ]
                last_valid_time.sort()

                if len(last_valid_time) > 0:
                    last_valid_time = last_valid_time[-2]
                else:
                    last_valid_time = att_time_index[-2]

                dump_result.link[att] = att_result.loc[:last_valid_time]
                last_valid_time_index = att_result.index.searchsorted(
                    last_valid_time
                )
                self._linear_result.link[att].drop(
                    att_result.index[: last_valid_time_index + 1], inplace=True
                )

            dump_file_index = len(self.registry.result_dump_file_list) + 1

            if dump_file_index >= 1:
                list_file_opening_mode = 'at'
            else:
                list_file_opening_mode = 'wt'

            result_dump_file_name = (
                self.registry.scenario_name + '.part' + str(dump_file_index)
            )
            result_dump_file_dst = os.path.join(
                self.registry.settings.process['result_directory'],
                result_dump_file_name,
            )

            with open(result_dump_file_dst, 'wb') as resul_file:
                pickle.dump(dump_result, resul_file)

            dump_list_file_name = self.registry.scenario_name + '.dumplist'
            list_file_dst = os.path.join(
                self.registry.settings.process['result_directory'],
                dump_list_file_name,
            )

            with open(list_file_dst, list_file_opening_mode) as part_list_file:
                part_list_file.writelines([result_dump_file_name])

            self.registry.result_dump_file_list.append(result_dump_file_name)
