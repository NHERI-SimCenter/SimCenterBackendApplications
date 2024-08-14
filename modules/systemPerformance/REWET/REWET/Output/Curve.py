"""Created on Tue Oct 25 14:30:01 2022

@author: snaeimi
"""  # noqa: CPY001, D400, INP001

import pandas as pd

from .Helper import hhelper


class Curve:  # noqa: D101
    def __init__():
        pass

    def getPipeStatusByAction(self, scn_name, action):  # noqa: N802, D102
        self.loadScneariodata(scn_name)
        reg = self.registry[scn_name]
        sequence = reg.retoration_data['sequence']['PIPE']
        if action not in sequence:
            raise ValueError('the action is not in the sequence: ' + str(action))
        pipe_damage_table_time_series = reg._pipe_damage_table_time_series  # noqa: SLF001
        time_action_done = {}
        for time in pipe_damage_table_time_series:
            current_pipe_damage_table = pipe_damage_table_time_series[time]
            current_action_damage = current_pipe_damage_table[action]
            number_of_all = len(current_action_damage)
            if number_of_all < 1:
                continue
            current_action_damage = current_action_damage[
                ~current_action_damage.isna()
            ]
            current_action_damage_true = current_action_damage[
                current_action_damage == True  # noqa: E712
            ]
            unique_done_orginal_element_list = (
                (
                    current_pipe_damage_table.loc[current_action_damage_true.index][
                        'Orginal_element'
                    ]
                )
                .unique()
                .tolist()
            )
            current_pipe_damage_table = current_pipe_damage_table.set_index(
                'Orginal_element'
            )
            current_action_damage = current_pipe_damage_table.loc[
                unique_done_orginal_element_list
            ]

            number_of_done = len(current_action_damage)
            time_action_done[time] = number_of_done / number_of_all

        return pd.Series(time_action_done)

    def getNodeStatusByAction(self, scn_name, action):  # noqa: N802, D102
        self.loadScneariodata(scn_name)
        reg = self.registry[scn_name]
        sequence = reg.retoration_data['sequence']['DISTNODE']
        if action not in sequence:
            raise ValueError('the action is not in the sequence: ' + str(action))
        node_damage_table_time_series = reg._node_damage_table_time_series  # noqa: SLF001
        time_action_done = {}
        for time in node_damage_table_time_series:
            current_node_damage_table = node_damage_table_time_series[time]
            current_action_damage = current_node_damage_table[action]
            number_of_all = len(current_action_damage)
            if number_of_all < 1:
                continue
            current_action_damage = current_action_damage[
                ~current_action_damage.isna()
            ]
            current_action_damage_true = current_action_damage[
                current_action_damage == True  # noqa: E712
            ]
            unique_done_orginal_element_list = (
                (
                    current_node_damage_table.loc[current_action_damage_true.index][
                        'Orginal_element'
                    ]
                )
                .unique()
                .tolist()
            )
            current_node_damage_table = current_node_damage_table.set_index(
                'Orginal_element'
            )
            current_action_damage = current_node_damage_table.loc[
                unique_done_orginal_element_list
            ]

            number_of_done = len(current_action_damage)
            time_action_done[time] = number_of_done / number_of_all

        return pd.Series(time_action_done)

    def getPumpStatus(self, scn_name):  # noqa: N802, D102
        self.loadScneariodata(scn_name)
        res = self.data[scn_name]
        reg = self.registry[scn_name]
        time_list = res.node['demand'].index
        pump_damage = reg.damage.damaged_pumps
        pump_damage_time = pump_damage.index

        time_action_done = {}
        for time in time_list:
            done_list = pump_damage_time[pump_damage_time >= time]
            time_action_done[time] = len(done_list) / len(pump_damage_time)

        return pd.Series(time_action_done)

    def getTankStatus(self, scn_name):  # noqa: N802, D102
        self.loadScneariodata(scn_name)
        reg = self.registry[scn_name]
        time_list = reg.time_list
        tank_damage = reg.damage.tamk_damage
        tank_damage_time = tank_damage.index

        time_action_done = {}
        for time in time_list:
            done_list = tank_damage_time[tank_damage_time >= time]
            time_action_done[time] = len(done_list) / len(tank_damage_time)

        return pd.Series(time_action_done)

    def getInputWaterFlowCurve(  # noqa: C901, N802, D102
        self,
        scn_name,
        tank_name_list=None,
        reservoir_name_list=None,
        mode='all',
    ):
        self.loadScneariodata(scn_name)
        res = self.data[scn_name]

        if tank_name_list == None:  # noqa: E711
            tank_name_list = self.wn.tank_name_list

        not_known_tank = set(tank_name_list) - set(self.wn.tank_name_list)
        if len(not_known_tank) > 0:
            raise ValueError(
                'The following tanks in the input are not known in the water network'
                + repr(tank_name_list)
            )

        if reservoir_name_list == None:  # noqa: E711
            reservoir_name_list = self.wn.reservoir_name_list

        not_known_reservoir = set(reservoir_name_list) - set(
            self.wn.reservoir_name_list
        )
        if len(not_known_reservoir) > 0:
            raise ValueError(
                'The following reservoirs in the input are not known in the water network'
                + repr(reservoir_name_list)
            )

        outbound_flow = pd.Series(0, index=res.node['demand'].index)
        inbound_flow = pd.Series(0, index=res.node['demand'].index)
        # inbound_flow  = 0
        # outbound_flow = 0

        waterFlow = None  # noqa: N806

        for tank_name in tank_name_list:
            if tank_name in res.node['demand'].columns:
                flow_in_time = res.node['demand'][tank_name]
            else:
                continue
            for time, flow in flow_in_time.iteritems():
                # print(flow)
                if flow > 0:
                    outbound_flow.loc[time] += -1 * flow
                elif flow < 0:
                    inbound_flow.loc[time] += -1 * flow

                if mode == 'all':
                    waterFlow = outbound_flow + inbound_flow  # noqa: N806
                elif mode == 'out':
                    waterFlow = outbound_flow  # noqa: N806
                elif mode == 'in':
                    waterFlow = inbound_flow  # noqa: N806
                else:
                    raise ValueError('Unnown mode: ' + repr(mode))

        for reservoir_name in reservoir_name_list:
            if reservoir_name in res.node['demand'].columns:
                flow_in_time = res.node['demand'][reservoir_name]
            else:
                continue
            for time, flow in flow_in_time.iteritems():
                # print(flow)
                if flow > 0:
                    outbound_flow.loc[time] += -1 * flow
                elif flow < 0:
                    inbound_flow.loc[time] += -1 * flow

                if mode == 'all':
                    waterFlow = outbound_flow + inbound_flow  # noqa: N806
                elif mode == 'out':
                    waterFlow = outbound_flow  # noqa: N806
                elif mode == 'in':
                    waterFlow = inbound_flow  # noqa: N806
                else:
                    raise ValueError('Unnown mode: ' + repr(mode))

        return waterFlow

    def getOveralDemandSatisfied(self, scn_name, pure=False):  # noqa: FBT002, N802, D102
        self.loadScneariodata(scn_name)
        if pure == False:  # noqa: E712
            demand_node_name_list = self.demand_node_name_list
        else:
            demand_node_name_list = []
            for node_name in self.wn.junction_name_list:
                if (
                    self.wn.get_node(node_name).demand_timeseries_list[0].base_value
                    > 0
                ):
                    demand_node_name_list.append(node_name)

        sat_node_demands = (
            self.data[scn_name].node['demand'].filter(demand_node_name_list)
        )
        # sat_node_demands = sat_node_demands.applymap(hhelper)
        s = sat_node_demands.sum(axis=1)

        return s  # noqa: RET504

    def getWaterLeakingFromNode(self, scn_name):  # noqa: N802, D102
        self.loadScneariodata(scn_name)
        res = self.data[scn_name]
        sum_amount = 0
        try:
            res = res.node['leak']
            sum_amount = res.sum(axis=1)
        except:  # noqa: E722
            sum_amount = 0
        return sum_amount

    def getWaterLeakingFromPipe(self, scn_name, mode='all'):  # noqa: N802, D102
        self.loadScneariodata(scn_name)
        reg = self.registry[scn_name]
        res = self.data[scn_name]

        damage_location_list = reg._pipe_damage_table  # noqa: SLF001

        if mode == 'leak' or mode == 'break':  # noqa: PLR1714
            damage_location_list = damage_location_list[
                damage_location_list['damage_type'] == mode
            ]
        elif mode == 'all':
            pass
        else:
            raise ValueError('The mode is not recognized: ' + repr(mode))

        break_damage_table = damage_location_list[
            damage_location_list['damage_type'] == 'break'
        ]
        pipe_B_list = self.registry[scn_name]._pipe_break_history.loc[  # noqa: SLF001, N806
            break_damage_table.index, 'Node_B'
        ]

        damage_location_list = damage_location_list.index
        wanted_nodes = pipe_B_list.to_list()
        wanted_nodes.extend(damage_location_list.to_list())

        available_nodes = set(res.node['demand'].columns)
        wanted_nodes = set(wanted_nodes)

        not_available_nodes = wanted_nodes - available_nodes
        available_nodes = wanted_nodes - not_available_nodes

        leak_from_pipe = res.node['demand'][available_nodes]

        leak = leak_from_pipe < -0.1  # noqa: PLR2004
        if leak.any().any():
            raise ValueError('There is negative leak')  # noqa: EM101, TRY003

        return leak_from_pipe.sum(axis=1)

    def getSystemServiceabilityIndexCurve(self, scn_name, iPopulation='No'):  # noqa: N802, N803, D102
        s4 = self.getRequiredDemandForAllNodesandtime(scn_name)
        sat_node_demands = (
            self.data[scn_name].node['demand'].filter(self.demand_node_name_list)
        )
        sat_node_demands = sat_node_demands.applymap(hhelper)

        if iPopulation == 'Yes':
            s4 = s4 * self._population_data  # noqa: PLR6104
            sat_node_demands = sat_node_demands * self._population_data  # noqa: PLR6104
        elif iPopulation == 'No':
            pass
        else:
            raise ValueError('unknown iPopulation value: ' + repr(iPopulation))

        s = sat_node_demands.sum(axis=1) / s4.sum(axis=1)

        for time_index, val in s.iteritems():
            if val < 0:
                val = 0  # noqa: PLW2901
            elif val > 1:
                val = 1  # noqa: PLW2901
            s.loc[time_index] = val

        return s

    def getBSCIndexPopulation_4(  # noqa: N802, D102
        self,
        scn_name,
        bsc='DL',
        iPopulation=False,  # noqa: FBT002, N803
        ratio=False,  # noqa: FBT002
        consider_leak=False,  # noqa: FBT002
        leak_ratio=1,
    ):
        if bsc == 'DL':
            return self.getDLIndexPopulation_4(
                scn_name,
                iPopulation=iPopulation,
                ratio=ratio,
                consider_leak=consider_leak,
                leak_ratio=leak_ratio,
            )
        elif bsc == 'QN':  # noqa: RET505
            return self.getQNIndexPopulation_4(
                scn_name,
                iPopulation=iPopulation,
                ratio=ratio,
                consider_leak=consider_leak,
                leak_ratio=leak_ratio,
            )
        else:
            raise ValueError(f'BSC input is not recognizable: {bsc}')  # noqa: EM102, TRY003

    def getDLIndexPopulation_4(  # noqa: C901, N802, D102
        self,
        scn_name,
        iPopulation='No',  # noqa: N803
        ratio=False,  # noqa: FBT002
        consider_leak=False,  # noqa: FBT002
        leak_ratio=1,
    ):
        if type(leak_ratio) != float:  # noqa: E721
            leak_ratio = float(leak_ratio)

        self.loadScneariodata(scn_name)
        res = self.data[scn_name]

        if type(self._population_data) == type(None) or iPopulation == False:  # noqa: E712, E721
            pop = pd.Series(index=self.demand_node_name_list, data=1)
        elif type(self._population_data) == type(None) and iPopulation == True:  # noqa: E712, E721
            raise ValueError('Population data is not available')  # noqa: EM101, TRY003
        else:
            pop = self._population_data

        total_pop = pop.sum()

        result = []
        refined_result = res.node['demand'][self.demand_node_name_list]
        demands = self.getRequiredDemandForAllNodesandtime(scn_name)
        demands = demands[self.demand_node_name_list]

        union_ = (
            set(res.node['leak'].columns).union(set(self.demand_node_name_list))
            - (set(res.node['leak'].columns) - set(self.demand_node_name_list))
            - (set(self.demand_node_name_list) - set(res.node['leak'].columns))
        )
        union_ = list(union_)
        leak_res = res.node['leak'][union_]

        leak_data = []

        if consider_leak:
            for name in leak_res:
                demand_name = demands[name]
                leak_res_name = leak_res[name].dropna()
                time_list = set(leak_res[name].dropna().index)
                time_list_drop = set(demands.index) - time_list
                demand_name = demand_name.drop(time_list_drop)
                leak_more_than_criteria = leak_res_name >= leak_ratio * demand_name
                if leak_more_than_criteria.any(0):
                    leak_data.append(leak_more_than_criteria)
        leak_data = pd.DataFrame(leak_data).transpose()

        s = refined_result > demands * 0.1
        for name in s:
            if name in leak_data.columns:
                leak_data_name = leak_data[name]
                for time in leak_data_name.index:
                    if leak_data_name.loc[time] == True:  # noqa: E712
                        s.loc[time, name] = False

        s = s * pop[s.columns]  # noqa: PLR6104

        if ratio == False:  # noqa: E712
            total_pop = 1
        else:
            total_pop = pop.sum()

        result = s.sum(axis=1) / total_pop

        return result  # noqa: RET504

    def getQNIndexPopulation_4(  # noqa: C901, N802, D102
        self,
        scn_name,
        iPopulation=False,  # noqa: FBT002, N803
        ratio=False,  # noqa: FBT002
        consider_leak=False,  # noqa: FBT002
        leak_ratio=0.75,
    ):
        if type(leak_ratio) != float:  # noqa: E721
            leak_ratio = float(leak_ratio)

        self.loadScneariodata(scn_name)
        res = self.data[scn_name]

        if type(self._population_data) == type(None) or iPopulation == False:  # noqa: E712, E721
            pop = pd.Series(index=self.demand_node_name_list, data=1)
        elif type(self._population_data) == type(None) and iPopulation == True:  # noqa: E712, E721
            raise ValueError('Population data is not available')  # noqa: EM101, TRY003
        else:
            pop = self._population_data

        result = []
        union_ = (
            set(res.node['leak'].columns).union(set(self.demand_node_name_list))
            - (set(res.node['leak'].columns) - set(self.demand_node_name_list))
            - (set(self.demand_node_name_list) - set(res.node['leak'].columns))
        )
        union_ = list(union_)
        refined_result = res.node['demand'][self.demand_node_name_list]
        demands = self.getRequiredDemandForAllNodesandtime(scn_name)
        demands = demands[self.demand_node_name_list]

        leak_res = res.node['leak'][union_]
        leak_data = []
        if consider_leak:
            for name in leak_res:
                demand_name = demands[name]
                leak_res_name = leak_res[name].dropna()
                time_list = set(leak_res_name.index)
                time_list_drop = set(demands.index) - time_list
                demand_name = demand_name.drop(time_list_drop)
                leak_more_than_criteria = leak_res_name >= leak_ratio * demand_name
                if leak_more_than_criteria.any(0):
                    leak_data.append(leak_more_than_criteria)
        leak_data = pd.DataFrame(leak_data).transpose()

        s = refined_result + 0.00000001 >= demands  # sina bug

        for name in s:
            if name in leak_data.columns:
                leak_data_name = leak_data[name]
                for time in leak_data_name.index:
                    if leak_data_name.loc[time] == True:  # noqa: E712
                        s.loc[time, name] = False

        s = s * pop[s.columns]  # noqa: PLR6104
        if ratio == False:  # noqa: E712
            total_pop = 1
        else:
            total_pop = pop.sum()

        result = s.sum(axis=1) / total_pop

        return result  # noqa: RET504

    def getQuantityExceedanceCurve(  # noqa: N802, D102
        self,
        iPopulation='No',  # noqa: ARG002, N803
        ratio=False,  # noqa: FBT002
        consider_leak=False,  # noqa: FBT002
        leak_ratio=0.75,
        result_type='mean',
        daily=False,  # noqa: FBT002
        min_time=0,
        max_time=999999999999999,
    ):
        all_scenarios_qn_data = self.AS_getQNIndexPopulation(
            iPopulation='No',
            ratio=ratio,
            consider_leak=consider_leak,
            leak_ratio=leak_ratio,
        )
        exceedance_curve = self.PR_getCurveExcedence(
            all_scenarios_qn_data,
            result_type=result_type,
            daily=daily,
            min_time=min_time,
            max_time=max_time,
        )
        columns_list = exceedance_curve.columns.to_list()

        dmg_vs_ep_list = {}

        for i in range(0, len(columns_list), 2):
            dmg_col = columns_list[i]
            ep_col = columns_list[i + 1]
            dmg_vs_ep_list[dmg_col] = ep_col
        res = {}

        for dmg_col in dmg_vs_ep_list:  # noqa: PLC0206
            ep_col = dmg_vs_ep_list[dmg_col]
            exceedance_curve_temp = exceedance_curve.set_index(dmg_col)
            exceedance_curve_temp = exceedance_curve_temp[ep_col]
            res[dmg_col] = exceedance_curve_temp

        return res

    def getDeliveryExceedanceCurve(  # noqa: N802, D102
        self,
        iPopulation='No',  # noqa: N803
        ratio=False,  # noqa: FBT002
        consider_leak=False,  # noqa: FBT002
        leak_ratio=0.75,
        result_type='mean',
        daily=False,  # noqa: FBT002
        min_time=0,
        max_time=999999999999999,
    ):
        all_scenarios_qn_data = self.AS_getDLIndexPopulation(
            iPopulation=iPopulation,
            ratio=ratio,
            consider_leak=consider_leak,
            leak_ratio=leak_ratio,
        )
        exceedance_curve = self.PR_getCurveExcedence(
            all_scenarios_qn_data,
            result_type=result_type,
            daily=daily,
            min_time=min_time,
            max_time=max_time,
        )
        columns_list = exceedance_curve.columns.to_list()

        dmg_vs_ep_list = {}

        for i in range(0, len(columns_list), 2):
            dmg_col = columns_list[i]
            ep_col = columns_list[i + 1]
            dmg_vs_ep_list[dmg_col] = ep_col
        res = {}

        for dmg_col in dmg_vs_ep_list:  # noqa: PLC0206
            ep_col = dmg_vs_ep_list[dmg_col]
            exceedance_curve_temp = exceedance_curve.set_index(dmg_col)
            exceedance_curve_temp = exceedance_curve_temp[ep_col]
            res[dmg_col] = exceedance_curve_temp

        return res
