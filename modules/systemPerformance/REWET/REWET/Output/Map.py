"""Created on Mon Oct 24 09:43:00 2022
This file includes all Map Related Results.

Class:
    Map: inherieted by Project_Result{

    functions:
        loadShapeFile
        joinTwoShapeFiles
        createGeopandasPointDataFrameForNodes
        getDLQNExceedenceProbabilityMap
        getOutageTimeGeoPandas_4
}

@author: snaeimi
"""  # noqa: CPY001, D205, INP001

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from Output import Helper

# import time


class Map:  # noqa: D101
    def __init__(self):
        pass

    # def loadShapeFile(shapeFileAddr='Northridge\GIS\Demand\demand_polygons.shp'):
    def loadShapeFile(  # noqa: D102, N802, PLR6301
        self,
        shapeFileAddr=r'Northridge\GIS\Demand\demand_polygons.shp',  # noqa: N803
    ):
        shape_file = gpd.read_file(shapeFileAddr)
        return shape_file  # noqa: RET504

    def joinTwoShapeFiles(self, first, second):  # noqa: D102, N802, PLR6301
        second = second.set_crs(crs=first.crs)
        joined_map = gpd.sjoin(first, second)

        return joined_map  # noqa: RET504

    def createGeopandasPointDataFrameForNodes(self):  # noqa: N802, D102
        s = gpd.GeoDataFrame(index=self.demand_node_name_list)
        point_list = []
        point_name_list = []

        for name in self.demand_node_name_list:
            coord = self.wn.get_node(name).coordinates
            point_list.append(shapely.geometry.Point(coord[0], coord[1]))
            point_name_list.append(name)
        s.geometry = point_list
        return s

    def getDLQNExceedenceProbabilityMap(self, data_frame, ihour, param):  # noqa: N802, D102
        data = data_frame.transpose()
        scn_prob_list = self.scenario_prob
        # DLQN_dmg = pd.DataFrame(data=0, index=data.index, columns=data.columns)

        scn_prob = [scn_prob_list[scn_name] for scn_name in data.index]
        data['prob'] = scn_prob

        res_dict_list = []
        tt = 0
        if ihour:
            for node_name in data_frame.index:
                loop_dmg = data[[node_name, 'prob']]
                loop_dmg = loop_dmg.sort_values(node_name, ascending=False)
                # t1 = time.time()
                # loop_ep  = Helper.EPHelper(loop_dmg['prob'].to_numpy())
                loop_ep = Helper.EPHelper(loop_dmg['prob'].to_numpy(), old=False)
                # loop_ep_2  = Helper.EPHelper(loop_dmg['prob'].to_numpy(), old=True)
                # return (loop_ep, loop_ep_2)
                # t2 = time.time()
                # dt = t2-t1
                # tt += dt
                loop_dmg['ep'] = loop_ep
                inter_ind = param
                if inter_ind >= loop_dmg['ep'].max():
                    max_ind = loop_dmg[loop_dmg['ep'] == loop_dmg['ep'].max()].index[
                        0
                    ]
                    # max_ind = loop_dmg.idxmax()
                    inter_value = loop_dmg.loc[max_ind, node_name]
                elif inter_ind <= loop_dmg['ep'].min():
                    min_ind = loop_dmg[loop_dmg['ep'] == loop_dmg['ep'].min()].index[
                        0
                    ]
                    # min_ind = loop_dmg.idxmin()
                    inter_value = loop_dmg.loc[min_ind, node_name]
                else:
                    loop_dmg.loc['inter', 'ep'] = inter_ind

                    loop_dmg = loop_dmg.sort_values('ep')
                    ep_list = loop_dmg['ep'].to_list()
                    inter_series = pd.Series(
                        index=ep_list, data=loop_dmg[node_name].to_list()
                    )
                    inter_series = inter_series.interpolate(method='linear')
                    inter_value = inter_series.loc[inter_ind]
                    if type(inter_value) != np.float64:  # noqa: E721
                        inter_value = inter_value.mean()

                res_dict_list.append({'node_name': node_name, 'res': inter_value})

        else:
            for node_name in data_frame.index:
                loop_dmg = data[[node_name, 'prob']]

                loop_dmg = loop_dmg.sort_values(node_name, ascending=False)
                loop_ep = Helper.EPHelper(loop_dmg['prob'].to_numpy())
                loop_dmg['ep'] = loop_ep
                inter_ind = param
                if inter_ind >= loop_dmg[node_name].max():
                    max_ind = loop_dmg[
                        loop_dmg[node_name] == loop_dmg[node_name].max()
                    ].index[0]
                    inter_value = loop_dmg.loc[max_ind, 'ep']
                elif inter_ind <= loop_dmg[node_name].min():
                    min_ind = loop_dmg[
                        loop_dmg[node_name] == loop_dmg[node_name].min()
                    ].index[0]
                    inter_value = loop_dmg.loc[min_ind, 'ep']
                else:
                    loop_dmg.loc['inter', node_name] = inter_ind

                    loop_dmg = loop_dmg.sort_values(node_name)
                    hour_list = loop_dmg[node_name].to_list()

                    inter_series = pd.Series(
                        index=hour_list, data=loop_dmg['ep'].to_list()
                    )
                    inter_series = inter_series.interpolate(method='linear')
                    inter_value = inter_series.loc[inter_ind]
                    if type(inter_value) != np.float64:  # noqa: E721
                        inter_value = inter_value.mean()

                res_dict_list.append({'node_name': node_name, 'res': inter_value})

        res = pd.DataFrame.from_dict(res_dict_list)
        res = res.set_index('node_name')['res']

        s = self.createGeopandasPointDataFrameForNodes()
        s['res'] = res

        # polygon = gpd.read_file('Northridge\GIS\Demand\demand_polygons.shp')
        # s = s.set_crs(epsg=polygon.crs.to_epsg())
        # joined_map = gpd.sjoin(polygon, s)
        # joined_map.plot(column='res', legend=True, categorical=True, cmap='Accent', ax=ax)
        # ax.get_legend().set_title('Hours without service')
        # ax.get_legend()._loc=3
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        print(tt)  # noqa: T201
        return s

    def getOutageTimeGeoPandas_4(  # noqa: C901, N802, D102
        self,
        scn_name,
        LOS='DL',  # noqa: N803
        iConsider_leak=False,  # noqa: FBT002, N803
        leak_ratio=0,
        consistency_time_window=7200,
    ):
        # print(repr(LOS) + "   " + repr(iConsider_leak)+"  "+ repr(leak_ratio)+"   "+repr(consistency_time_window  ) )
        self.loadScneariodata(scn_name)
        res = self.data[scn_name]
        map_res = pd.Series(data=0, index=self.demand_node_name_list, dtype=np.int64)

        demands = self.getRequiredDemandForAllNodesandtime(scn_name)
        refined_res = res.node['demand'][self.demand_node_name_list]
        union_ = set(res.node['leak'].columns).union(
            set(self.demand_node_name_list)
            - (set(res.node['leak'].columns))
            - set(self.demand_node_name_list)
        ) - (set(self.demand_node_name_list) - set(res.node['leak'].columns))
        leak_res = res.node['leak'][union_]

        leak_data = []
        if iConsider_leak:
            for name in leak_res:
                demand_name = demands[name]
                current_leak_res = leak_res[name].dropna()
                time_list = set(current_leak_res.dropna().index)
                time_list_drop = set(demands.index) - time_list
                demand_name = demand_name.drop(time_list_drop)
                leak_more_than_criteria = (
                    current_leak_res >= leak_ratio * demand_name
                )
                if leak_more_than_criteria.any(0):
                    leak_data.append(leak_more_than_criteria)
        leak_data = pd.DataFrame(leak_data).transpose()

        demands = demands[self.demand_node_name_list]

        if LOS == 'DL':
            DL_res_not_met_bool = refined_res <= demands * 0.01  # noqa: N806
        elif LOS == 'QN':
            DL_res_not_met_bool = refined_res < demands * 0.98  # noqa: N806

        time_window = consistency_time_window + 1
        time_list = DL_res_not_met_bool.index.to_list()
        time_list.reverse()

        for time in time_list:
            past_time_beg = time - time_window
            window_data = DL_res_not_met_bool.loc[past_time_beg:time]
            window_data = window_data.all()
            window_data_false = window_data[window_data == False]  # noqa: E712
            DL_res_not_met_bool.loc[time, window_data_false.index] = False

        for name in DL_res_not_met_bool:
            if name in leak_data.columns:
                leak_data_name = leak_data[name]
                for time in leak_data_name.index:
                    if leak_data_name.loc[time] == True:  # noqa: E712
                        DL_res_not_met_bool.loc[time, name] = True

        all_node_name_list = refined_res.columns
        only_not_met_bool = DL_res_not_met_bool.any(0)
        only_not_met_any = all_node_name_list[only_not_met_bool]
        DL_res_not_met = DL_res_not_met_bool.filter(only_not_met_any)  # noqa: N806
        DL_res_MET = ~DL_res_not_met  # noqa: N806
        time_window = 2

        for name in only_not_met_any:
            rolled_DL_res_MET = (  # noqa: N806
                DL_res_MET[name].rolling(time_window, center=True).sum()
            )
            rolled_DL_res_MET = rolled_DL_res_MET.sort_index(ascending=False)  # noqa: N806
            rolled_DL_res_MET.dropna(inplace=True)  # noqa: PD002

            false_found, found_index = Helper.helper_outageMap(
                rolled_DL_res_MET.ge(time_window - 1)
            )
            # if name == "SM323":
            # return DL_res_MET[name], rolled_DL_res_MET, false_found, rolled_DL_res_MET.index[found_index], rolled_DL_res_MET.ge(time_window-1), found_index
            if false_found == False:  # noqa: E712
                latest_time = 0
            else:
                if DL_res_MET[name].iloc[-1] == False:  # noqa: E712
                    latest_time = DL_res_MET.index[-1]
                else:
                    latest_time = rolled_DL_res_MET.index[found_index]
                latest_time = rolled_DL_res_MET.index[found_index]

            map_res.loc[name] = latest_time

        # map_res = map_res/(3600*24)
        geopandas_df = self.createGeopandasPointDataFrameForNodes()
        geopandas_df.loc[map_res.index.to_list(), 'restoration_time'] = (
            map_res.to_list()
        )

        return geopandas_df

    def getOutageTimeGeoPandas_5(  # noqa: C901, N802, D102
        self,
        scn_name,
        bsc='DL',
        iConsider_leak=False,  # noqa: FBT002, N803
        leak_ratio=0,
        consistency_time_window=7200,
        sum_time=False,  # noqa: FBT002
    ):
        self.loadScneariodata(scn_name)
        res = self.data[scn_name]
        map_res = pd.Series(data=0, index=self.demand_node_name_list, dtype=np.int64)

        required_demand = self.getRequiredDemandForAllNodesandtime(scn_name)
        delivered_demand = res.node['demand'][self.demand_node_name_list]
        common_nodes_leak = list(
            set(res.node['leak'].columns).intersection(
                set(self.demand_node_name_list)
            )
        )
        leak_res = res.node['leak'][common_nodes_leak]

        common_nodes_demand = list(
            set(delivered_demand.columns).intersection(
                set(self.demand_node_name_list)
            )
        )
        delivered_demand = delivered_demand[common_nodes_demand]
        required_demand = required_demand[common_nodes_demand]

        required_demand.sort_index(inplace=True)  # noqa: PD002
        delivered_demand.sort_index(inplace=True)  # noqa: PD002
        leak_res.sort_index(inplace=True)  # noqa: PD002

        # return delivered_demand, required_demand, leak_res

        if bsc == 'DL':
            bsc_res_not_met_bool = (
                delivered_demand.fillna(0) <= required_demand * 0.0001
            )
        elif bsc == 'QN':
            bsc_res_not_met_bool = (
                delivered_demand.fillna(0) < required_demand * 0.9999
            )
        else:
            raise ValueError('Unknown BSC= ' + str(bsc))

        if iConsider_leak:
            # return leak_res, required_demand
            leak_res_non_available_time_list = set(
                required_demand[leak_res.columns].index
            ) - set(leak_res.index)
            if len(leak_res_non_available_time_list) > 0:
                leak_res_non_available_time_list = list(
                    leak_res_non_available_time_list
                )
                temp_data = pd.DataFrame(
                    [
                        [0 for i in leak_res.columns]
                        for j in range(len(leak_res_non_available_time_list))
                    ],
                    index=leak_res_non_available_time_list,
                    columns=leak_res.columns,
                )
                # leak_res.loc[leak_res_non_available_time_list, : ] = temp_data
                leak_res = leak_res.append(temp_data)
                leak_res.sort_index(inplace=True)  # noqa: PD002
            leak_criteria_exceeded = (
                leak_res.fillna(0) >= leak_ratio * required_demand[leak_res.columns]
            )
            combined_negative_result = (
                bsc_res_not_met_bool | leak_criteria_exceeded
            ).dropna(axis=1)
            # return combined_negative_result
            bsc_res_not_met_bool.loc[:, combined_negative_result.columns] = (
                combined_negative_result
            )

        # end_time = delivered_demand.min()
        end_time = delivered_demand.index.max()
        if consistency_time_window > 1:
            time_beg_step_list = np.arange(0, end_time, consistency_time_window)

            # time_beg_step_list = np.append(time_beg_step_list, [end_time])
            time_end_step_list = time_beg_step_list  # + consistency_time_window
            window_bsc_not_met = pd.DataFrame(
                index=time_end_step_list,
                columns=bsc_res_not_met_bool.columns,
                dtype=bool,
            )
            # return bsc_res_not_met_bool#, delivered_demand, required_demand
            for step_time_beg in time_beg_step_list:
                step_time_end = step_time_beg + consistency_time_window
                window_data = bsc_res_not_met_bool.loc[step_time_beg:step_time_end]
                if len(window_data) > 0:
                    window_data = window_data.all()
                    window_bsc_not_met.loc[step_time_beg, window_data.index] = (
                        window_data
                    )
                else:
                    # print(step_time_beg)
                    window_bsc_not_met.drop(step_time_beg, inplace=True)  # noqa: PD002
        else:
            window_bsc_not_met = bsc_res_not_met_bool

        pre_incident = (window_bsc_not_met.loc[: 3600 * 3]).any()
        non_incident = pre_incident[pre_incident == False].index  # noqa: E712

        not_met_node_name_list = window_bsc_not_met.any()

        # ("****************")
        # print(not_met_node_name_list[not_met_node_name_list==True])

        not_met_node_name_list = not_met_node_name_list[
            not_met_node_name_list == True  # noqa: E712
        ]
        not_met_node_name_list = not_met_node_name_list.index

        if sum_time:
            time_difference = (
                window_bsc_not_met.index[1:] - window_bsc_not_met.index[:-1]
            )
            timed_diference_window_bsc_not_met = (
                time_difference * window_bsc_not_met.iloc[1:].transpose()
            ).transpose()
            timed_diference_window_bsc_not_met.iloc[0] = 0
            sum_window_bsc_not_met = timed_diference_window_bsc_not_met.sum()
            return sum_window_bsc_not_met  # noqa: RET504

        window_bsc_not_met = window_bsc_not_met[not_met_node_name_list]
        cut_time = window_bsc_not_met.index.max()
        non_incident = list(
            set(non_incident).intersection(set(not_met_node_name_list))
        )
        for step_time, row in window_bsc_not_met[non_incident].iterrows():
            if step_time <= 14400:  # noqa: PLR2004
                continue

            if row.any() == False:  # noqa: E712
                print(step_time)  # noqa: T201
                cut_time = step_time
                break

        window_bsc_not_met = window_bsc_not_met.loc[:cut_time]
        window_bsc_not_met = window_bsc_not_met.loc[:cut_time]

        # return window_bsc_not_met
        # print(not_met_node_name_list)
        time_bsc_not_met_time = window_bsc_not_met.sort_index(
            ascending=False
        ).idxmax()
        map_res.loc[time_bsc_not_met_time.index] = time_bsc_not_met_time

        never_reported_nodes = set(self.demand_node_name_list) - set(
            common_nodes_demand
        )
        number_of_unreported_demand_nodes = len(never_reported_nodes)
        if number_of_unreported_demand_nodes > 0:
            warnings.warn(  # noqa: B028
                'REWET WARNING: there are '
                + str(number_of_unreported_demand_nodes)
                + 'unreported nodes'
            )
            map_res.loc[never_reported_nodes] = end_time

        map_res = map_res / (3600 * 24)  # noqa: PLR6104
        return map_res  # noqa: RET504

        s = gpd.GeoDataFrame(index=self.demand_node_name_list)
        point_list = []
        point_name_list = []

        for name in self.demand_node_name_list:
            coord = self.wn.get_node(name).coordinates
            point_list.append(shapely.geometry.Point(coord[0], coord[1]))
            point_name_list.append(name)

        s.geometry = point_list

        s.loc[map_res.index.to_list(), 'restoration_time'] = map_res.to_list()

        polygon = gpd.read_file(r'Northridge\GIS\Demand\demand_polygons.shp')
        s = s.set_crs(crs=polygon.crs)
        joined_map = gpd.sjoin(polygon, s)
        # return   joined_map
        # joined_map.loc[map_res.index.to_list(), 'restoration_time'] = (map_res/3600/24).to_list()

        return joined_map  # noqa: RET504

    def percentOfEffectNodes(  # noqa: C901, N802, D102
        self,
        scn_name,
        bsc='QN',
        iConsider_leak=True,  # noqa: FBT002, N803
        leak_ratio=0.75,
        consistency_time_window=7200,
    ):
        self.loadScneariodata(scn_name)
        res = self.data[scn_name]
        map_res = pd.Series(data=0, index=self.demand_node_name_list, dtype=np.int64)

        required_demand = self.getRequiredDemandForAllNodesandtime(scn_name)
        delivered_demand = res.node['demand'][self.demand_node_name_list]
        common_nodes_leak = set(res.node['leak'].columns).intersection(
            set(self.demand_node_name_list)
        )
        leak_res = res.node['leak'][common_nodes_leak]

        common_nodes_demand = list(
            set(delivered_demand.columns).intersection(
                set(self.demand_node_name_list)
            )
        )
        delivered_demand = delivered_demand[common_nodes_demand]
        required_demand = required_demand[common_nodes_demand]

        required_demand.sort_index(inplace=True)  # noqa: PD002
        delivered_demand.sort_index(inplace=True)  # noqa: PD002
        leak_res.sort_index(inplace=True)  # noqa: PD002

        # return delivered_demand, required_demand, leak_res

        if bsc == 'DL':
            bsc_res_not_met_bool = (
                delivered_demand.fillna(0) <= required_demand * 0.1
            )
        elif bsc == 'QN':
            bsc_res_not_met_bool = (
                delivered_demand.fillna(0) < required_demand * 0.98
            )
        else:
            raise ValueError('Unknown BSC= ' + str(bsc))

        if iConsider_leak:
            # return leak_res, required_demand
            leak_res_non_available_time_list = set(
                required_demand[leak_res.columns].index
            ) - set(leak_res.index)
            if len(leak_res_non_available_time_list) > 0:
                leak_res_non_available_time_list = list(
                    leak_res_non_available_time_list
                )
                temp_data = pd.DataFrame(
                    [
                        [0 for i in leak_res.columns]
                        for j in range(len(leak_res_non_available_time_list))
                    ],
                    index=leak_res_non_available_time_list,
                    columns=leak_res.columns,
                )
                # leak_res.loc[leak_res_non_available_time_list, : ] = temp_data
                leak_res = leak_res.append(temp_data)
                leak_res.sort_index(inplace=True)  # noqa: PD002
            leak_criteria_exceeded = (
                leak_res.fillna(0) >= leak_ratio * required_demand[leak_res.columns]
            )
            combined_negative_result = (
                bsc_res_not_met_bool | leak_criteria_exceeded
            ).dropna(axis=1)
            # return combined_negative_result
            bsc_res_not_met_bool.loc[:, combined_negative_result.columns] = (
                combined_negative_result
            )

        # end_time = delivered_demand.min()
        end_time = delivered_demand.index.max()
        time_beg_step_list = np.arange(0, end_time, consistency_time_window)

        # time_beg_step_list = np.append(time_beg_step_list, [end_time])
        time_end_step_list = time_beg_step_list  # + consistency_time_window
        window_bsc_not_met = pd.DataFrame(
            index=time_end_step_list,
            columns=bsc_res_not_met_bool.columns,
            dtype=bool,
        )
        # return bsc_res_not_met_bool#, delivered_demand, required_demand
        for step_time_beg in time_beg_step_list:
            step_time_end = step_time_beg + consistency_time_window
            window_data = bsc_res_not_met_bool.loc[step_time_beg:step_time_end]
            if len(window_data) > 0:
                window_data = window_data.all()
                window_bsc_not_met.loc[step_time_beg, window_data.index] = (
                    window_data
                )
            else:
                # print(step_time_beg)
                window_bsc_not_met.drop(step_time_beg, inplace=True)  # noqa: PD002
        # return window_bsc_not_met
        pre_incident = (window_bsc_not_met.loc[: 3600 * 3]).any()
        non_incident = pre_incident[pre_incident == False].index  # noqa: E712

        number_of_good_nodes = len(non_incident)

        not_met_node_name_list = window_bsc_not_met.any()

        # ("****************")
        # print(not_met_node_name_list[not_met_node_name_list==True])

        not_met_node_name_list = not_met_node_name_list[
            not_met_node_name_list == True  # noqa: E712
        ]
        not_met_node_name_list = not_met_node_name_list.index
        window_bsc_not_met = window_bsc_not_met[not_met_node_name_list]

        cut_time = window_bsc_not_met.index.max()
        non_incident = list(
            set(non_incident).intersection(set(not_met_node_name_list))
        )
        for step_time, row in window_bsc_not_met[non_incident].iterrows():
            if step_time <= 14400:  # noqa: PLR2004
                continue

            if row.any() == False:  # noqa: E712
                print(step_time)  # noqa: T201
                cut_time = step_time
                break

        cut_time = 24 * 3600
        window_bsc_not_met = window_bsc_not_met.loc[:cut_time]
        window_bsc_not_met = window_bsc_not_met.loc[:cut_time]

        number_of_bad_node_at_damage = (
            window_bsc_not_met[non_incident].loc[14400].sum()
        )
        percent_init = number_of_bad_node_at_damage / number_of_good_nodes * 100
        # return window_bsc_not_met
        # print(not_met_node_name_list)
        time_bsc_not_met_time = window_bsc_not_met.sort_index(
            ascending=False
        ).idxmax()
        map_res.loc[time_bsc_not_met_time.index] = time_bsc_not_met_time

        never_reported_nodes = set(self.demand_node_name_list) - set(
            common_nodes_demand
        )
        number_of_unreported_demand_nodes = len(never_reported_nodes)
        if number_of_unreported_demand_nodes > 0:
            warnings.warn(  # noqa: B028
                'REWET WARNING: there are '
                + str(number_of_unreported_demand_nodes)
                + 'unreported nodes'
            )
            map_res.loc[never_reported_nodes] = end_time

        map_res = map_res / (3600 * 24)  # noqa: PLR6104
        percent = (map_res.loc[non_incident] > 0).sum() / number_of_good_nodes * 100
        return np.round(percent_init, 2), np.round(percent, 2)
