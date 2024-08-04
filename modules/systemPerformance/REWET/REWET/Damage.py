"""Created on Mon Mar 23 13:54:21 2020
This module is responsible for calculating damage to t=different componenst of
the system, including pipe lines. pupmo and so.
@author: snaeimi
"""  # noqa: CPY001, D205, D400, N999

import logging
import math
import pickle  # noqa: S403

import numpy as np
import pandas as pd
import wntrfr
from EnhancedWNTR.morph.link import break_pipe, split_pipe
from scipy.stats import lognorm
from wntrfr.network.model import LinkStatus

logger = logging.getLogger(__name__)


class EarthquakeScenario:  # noqa: D101
    def __init__(self, magnitude, depth, x_coord, y_coord, eq_time):
        self.M = abs(magnitude)
        self.depth = abs(depth)
        self.coordinate = {}
        self.coordinate['X'] = x_coord
        self.coordinate['Y'] = y_coord
        self.time = abs(eq_time)

    def getWNTREarthquakeObject(self):  # noqa: N802, D102
        return wntrfr.scenario.Earthquake(
            (self.coordinate['X'], self.coordinate['Y']), self.M, self.depth
        )


class Damage:  # noqa: D101, PLR0904
    def __init__(self, registry, scenario_set):
        self.scenario_set = scenario_set
        self.pipe_leak = pd.Series(dtype='O')
        self.pipe_break = pd.Series(dtype='O')
        self.pipe_all_damages = None
        self.tank_damage = pd.Series(dtype='O')
        self.node_damage = pd.Series(dtype='O')
        # self._earthquake      = pd.Series(dtype="O")
        self._registry = registry
        self.default_time = 4
        # if damageEndTime==None:
        self.end_time = 100
        # else:
        # self.end_time=damageEndTime
        self.is_timely_sorted = False

        self._pipe_last_ratio = pd.Series(dtype='float64')
        self.damaged_pumps = pd.Series(dtype='float64')
        self.nodal_equavalant_diameter = None

        # self._nodal_damage_method = None
        self._pipe_damage_method = 1

    def readDamageFromPickleFile(  # noqa: N802
        self,
        pickle_file_name,
        csv_file_name,
        csv_index=None,
    ):
        """This function is only for the sake of reading picke file that nafiseg gives to me
        This function shall not be in any distribution that we release

        Parameters
        ----------
        pickle_file_name : string
            name file of path + name file of the pickle file
        csv_file_name : name file of path + name file of the csv file

        Returns
        -------

        """  # noqa: D205, D400, D401, D404, D414
        with open(pickle_file_name, 'rb') as pckf:  # noqa: PTH123
            w = pickle.load(pckf)  # noqa: S301

        name_list = pd.read_csv(csv_file_name, index_col=csv_index)
        damage_name_list = []
        damage_mat_list = []

        for ind, val in w.items():
            if ind[4] == 0 and val == 1:
                refr = ind[2]
                damage_name_list.append(name_list.index[refr])
                damage_mat_list.append(name_list['material'][refr])

        damage_list = pd.DataFrame()
        damage_list['name'] = damage_name_list
        damage_list['material'] = damage_mat_list
        damage_state_probability = {
            'STL': [0.2, 0.8],
            'CI': [0.2, 0.8],
            'DI': [0, 1],
            'CON': [0.2, 0.8],
            'RS': [0.2, 0.8],
        }
        damage_list = self.sampleDamageStatesBasedOnMaterialFragility(
            damage_list, damage_state_probability
        )
        # return damage_list
        self.addPipeDamageByDamageList(damage_list, 1, 0)

        # print(name_list)

    def readPumpDamage(self, file_name):  # noqa: N802, D102
        pump_list = pd.read_csv(file_name)
        self.damaged_pumps = pump_list['Pump_ID']

    def readNodalDamage(self, file_address):  # noqa: N802, D102
        temp = pd.read_csv(file_address)
        for ind, val in temp.iterrows():  # noqa: B007
            temp_data = {}
            temp_data['node_name'] = str(val['NodeID'])
            temp_data['node_RR'] = val['RR']
            temp_data['node_Pre_EQ_Demand'] = (
                val['Pre_EQ_Demand'] * 6.30901964 / 100000
            )  # *0.5
            temp_data['node_Post_EQ_Demand'] = (
                val['Post_EQ_Demand'] * 6.30901964 / 100000
            )  # *0.5 # * 6.30901964/100000*(1+0.01*val['setNumDamages'])
            temp_data['node_Pipe_Length'] = val['PipeLength']
            temp_data['Number_of_damages'] = val['setNumDamages']
            temp_data['node_Customer'] = val['#Customer']
            temp_data['node_LargeUser'] = val['LargeUser']

            self.node_damage = self.node_damage.append(pd.Series(data=[temp_data]))

        self.node_damage.reset_index(drop=True, inplace=True)  # noqa: PD002

    def setNodalDamageModelParameter(self, damage_param):  # noqa: N802, D102
        self._registry.nodal_equavalant_diameter = damage_param

    def readDamageGiraffeFormat(self, break_file_name, leak_file_name):  # noqa: N802, D102
        break_temp = pd.read_csv(break_file_name)
        leak_temp = pd.read_csv(leak_file_name)

        temp_break_pipe_ID = break_temp['PipeID']  # noqa: N806
        temp_leak_pipe_ID = leak_temp['PipeID']  # noqa: N806

        if temp_break_pipe_ID.dtype != 'O':
            temp_break_pipe_ID = temp_break_pipe_ID.apply(lambda x: str(x))  # noqa: N806, PLW0108
            break_temp['PipeID'] = temp_break_pipe_ID

        if temp_leak_pipe_ID.dtype != 'O':
            temp_leak_pipe_ID = temp_leak_pipe_ID.apply(lambda x: str(x))  # noqa: N806, PLW0108
            leak_temp['PipeID'] = temp_leak_pipe_ID

        temp1 = break_temp[['PipeID', 'BreakRatio']]
        temp1._is_copy = None  # noqa: SLF001
        temp1['damage'] = 'break'
        temp1.rename(columns={'BreakRatio': 'ratio'}, inplace=True)  # noqa: PD002

        temp2 = leak_temp[['PipeID', 'LeakRatio']]
        temp2._is_copy = None  # noqa: SLF001
        temp2.rename(columns={'LeakRatio': 'ratio'}, inplace=True)  # noqa: PD002
        temp2['damage'] = 'leak'

        temp = pd.concat([temp1, temp2])

        temp = temp.sort_values(['PipeID', 'ratio'], ascending=(True, False))

        unique_pipe_ID = temp['PipeID'].unique().tolist()  # noqa: N806

        for pipe_ID in unique_pipe_ID:  # noqa: N806
            selected_damage = temp[temp['PipeID'] == pipe_ID]

            if 'break' in selected_damage['damage'].tolist():
                number = len(selected_damage)
                tmp_break = {
                    'pipe_id': pipe_ID,
                    'break_loc': 0.5,
                    'break_time': self.default_time,
                    'number': number,
                }
                self.pipe_break = self.pipe_break.append(
                    pd.Series(
                        data=[tmp_break], index=[int(tmp_break['break_time'] * 3600)]
                    )
                )

            else:
                number = len(selected_damage)
                temp_leak_D = pd.Series(data=selected_damage.index)  # noqa: N806
                temp_leak_D = temp_leak_D.apply(lambda x: leak_temp.loc[x, 'LeakD'])  # noqa: N806

                leak_D = ((temp_leak_D**2).sum()) ** 0.5  # noqa: N806
                tmp_leak = {
                    'pipe_id': pipe_ID,
                    'leak_loc': 0.5,
                    'leakD': leak_D / 100 * 2.54,
                    'leak_type': 1,
                    'leak_time': self.default_time,
                    'number': number,
                }
                self.pipe_leak = self.pipe_leak.append(
                    pd.Series(
                        data=[tmp_leak], index=[int(tmp_leak['leak_time'] * 3600)]
                    )
                )

    def addPipeDamageByDamageList(self, damage_list, leak_type_ref, break_type_ref):  # noqa: ARG002, N802, D102
        # leaked_damage = damage_list[damage_list['damage_state']==leak_type_ref]

        for ind, row in damage_list.iterrows():  # noqa: B007
            if row['damage_state'] == 0:  # break
                tmp_break = {
                    'pipe_id': row['name'],
                    'break_loc': 0.5,
                    'break_time': self.default_time,
                }
                self.pipe_break = self.pipe_break.append(
                    pd.Series(
                        data=[tmp_break], index=[int(tmp_break['break_time'] * 3600)]
                    )
                )
            elif row.damage_state == 1:  # leak
                tmp_leak = {
                    'pipe_id': row['name'],
                    'leak_loc': 0.5,
                    'leak_type': 1,
                    'leak_time': self.default_time,
                }
                self.pipe_leak = self.pipe_leak.append(
                    pd.Series(
                        data=[tmp_leak], index=[int(tmp_leak['leak_time'] * 3600)]
                    )
                )
            else:
                raise ValueError('There is an unknown damage type')  # noqa: EM101, TRY003

    def readDamageFromTextFile(self, path):  # noqa: N802
        """Reads a damage from scenario from a text file and add the information
            to the damage class object.

        Parameters
        ----------
        [path] : str
            The input file name

        """  # noqa: D205, D401
        if path == None:  # noqa: E711
            raise ValueError('None in path')  # noqa: DOC501, EM101, TRY003
        file = open(path)  # noqa: PLW1514, PTH123, SIM115
        lines = file.readlines()
        line_cnt = 0
        for line in lines:
            line_cnt += 1  # noqa: SIM113
            sline = line.split()
            line_length = len(sline)

            if sline[0].lower() == 'leak':
                # print(len(sline))
                temp_leak = {}
                if line_length < 4:  # noqa: PLR2004
                    raise OSError(  # noqa: DOC501
                        'There must be at least 4 arguments in line' + repr(line_cnt)
                    )
                    # print('Probelm 1')
                temp_leak['pipe_id'] = sline[1]
                temp_leak['leak_loc'] = float(sline[2])
                temp_leak['leak_type'] = int(sline[3])
                if line_length > 4:  # noqa: PLR2004
                    temp_leak['leak_time'] = float(sline[4])
                else:
                    temp_leak['leak_time'] = self.default_time
                self.pipe_leak = self.pipe_leak.append(
                    pd.Series(
                        data=[temp_leak], index=[int(temp_leak['leak_time'] * 3600)]
                    )
                )

            elif sline[0].lower() == 'break':
                if line_length < 3:  # noqa: PLR2004
                    raise OSError('Line cannot have more than three arguments')  # noqa: DOC501, EM101, TRY003
                # print('Probelm 2')
                temp_break = {}
                temp_break['pipe_id'] = sline[1]
                temp_break['break_loc'] = float(sline[2])
                if line_length > 3:  # noqa: PLR2004
                    temp_break['break_time'] = float(sline[3])
                else:
                    temp_break['break_time'] = self.default_time
                # print( type(temp_break['break_time']))
                self.pipe_break = self.pipe_break.append(
                    pd.Series(
                        data=[temp_break],
                        index=[int(temp_break['break_time'] * 3600)],
                    )
                )
            else:
                logger.warning(sline)
                logger.warning(
                    'No recogniziable command in damage file, line'  # noqa: G003
                    + repr(line_cnt)
                    + '\n'
                )
        file.close()

    def applyNodalDamage(self, WaterNetwork, current_time):  # noqa: C901, N802, N803
        """Apply Nodal Damage

        Parameters
        ----------
        WaterNetwork : Water Network Model
            DESCRIPTION.

        Returns
        -------
        None.

        """  # noqa: D400
        if self.node_damage.empty:
            print('no node damage at all')  # noqa: T201
            return

        curren_time_node_damage = self.node_damage[current_time]

        if type(curren_time_node_damage) == dict:  # noqa: E721
            curren_time_node_damage = pd.Series(
                [curren_time_node_damage], index=[current_time]
            )
        elif type(curren_time_node_damage) == pd.Series:  # noqa: E721
            if curren_time_node_damage.empty:
                print('No node damage at time ' + str(current_time))  # noqa: T201
                return
        else:
            raise ValueError(  # noqa: DOC501
                'Node damage has a unknown type: '
                + str(type(curren_time_node_damage))
                + ' at time: '
                + str(current_time)
            )

        # self._nodal_damage_method = self._registry.settings['damage_node_model']
        method = self._registry.settings['damage_node_model']
        if method == 'Predefined_demand':
            for ind, val in curren_time_node_damage.items():  # noqa: B007, PERF102
                node_name = val['node_name']
                pre_EQ_Demand = val['node_Pre_EQ_Demand']  # noqa: N806
                post_EQ_Demand = val['node_Post_EQ_Demand']  # noqa: N806

                # if node_name not in WaterNetwork.node_name_list and icheck==True:
                # raise ValueError('Node in damage list is not in water network model: '+repr(node_name))
                # elif icheck==False:
                # continue
                node_cur_dem = (
                    WaterNetwork.get_node(node_name)  # noqa: SLF001
                    .demand_timeseries_list._list[0]
                    .base_value
                )
                # print(str(pre_EQ_Demand) + '  ' + str(node_cur_dem))
                # print(node_name)
                if abs(pre_EQ_Demand - node_cur_dem) > 0.001:  # noqa: PLR2004
                    raise  # noqa: PLE0704

                ratio = post_EQ_Demand / pre_EQ_Demand

                WaterNetwork.get_node(node_name).demand_timeseries_list._list[  # noqa: SLF001
                    0
                ].base_value = node_cur_dem * ratio

                self._registry.addNodalDemandChange(
                    val['node_name'], node_cur_dem, post_EQ_Demand
                )
            demand_damage = self.estimateNodalDamage()
            self._registry.addNodalDamage(demand_damage)

        elif (
            method == 'equal_diameter_emitter'  # noqa: PLR1714
            or method == 'equal_diameter_reservoir'
        ):
            temp1 = []
            temp2 = []
            temp_new_explicit_leak_data = []
            for ind, val in curren_time_node_damage.items():  # noqa: B007, PERF102
                node_name = val['node_name']
                number_of_damages = val['Number_of_damages']
                pipe_length = val['node_Pipe_Length'] * 1000

                if node_name not in WaterNetwork.node_name_list:
                    raise ValueError(  # noqa: DOC501
                        'Node name of damages not in node name list: ' + node_name
                    )

                new_node_name, new_pipe_name, mp, q = (
                    self.addExplicitLeakWithReservoir(
                        node_name, number_of_damages, pipe_length, WaterNetwork
                    )
                )
                # self._registry.nodal_damage_nodes.add(new_node_name)
                # post_EQ_Demand = val['node_Post_EQ_Demand']
                # pre_EQ_Demand  = val['node_Pre_EQ_Demand']
                # self._registry.addNodalDemandChange(val['node_name'], pre_EQ_Demand, post_EQ_Demand)
                temp1.append(node_name)
                temp2.append(val['Number_of_damages'])

                self._registry.active_nodal_damages.update(
                    {new_node_name: node_name}
                )
                temp_data = {
                    'mean_pressure': mp,
                    'new_pipe_name': new_pipe_name,
                    'new_node_name': new_node_name,
                    'pipe_length': pipe_length,
                    'orginal_flow': q,
                }
                temp_new_explicit_leak_data.append(temp_data)

            demand_damage = pd.Series(data=temp2, index=temp1)
            new_pipe_name_list = dict(zip(temp1, temp_new_explicit_leak_data))
            self._registry.addNodalDamage(demand_damage, new_pipe_name_list)
        elif method == 'SDD':
            for ind, val in curren_time_node_damage.items():  # noqa: B007, PERF102
                node_name = val['node_name']
                number_of_damages = val['Number_of_damages']
                pipe_length = val['node_Pipe_Length'] * 1000
                if node_name not in WaterNetwork.node_name_list:
                    raise ValueError(  # noqa: DOC501
                        'Node name of damages not in node name list: ' + node_name
                    )
                maximum_node_demand = 10
                pipe_equal_length = pipe_length / 10
                _hl = 8
                _C = 100  # noqa: N806
                before_damage_pipe_length = pipe_equal_length / 2
                over_designed_diameter = (
                    10.67 * (maximum_node_demand / _C) ** 1.852 * (pipe_length / _hl)
                )
                over_designed_diameter = over_designed_diameter ** (1 / 4.8704)  # noqa: PLR6104

                equavalant_damaged_pipe_diameter = self.scenario_set[
                    'equavalant_damage_diameter'
                ]
                equavalant_pipe_diameter = (
                    np.sqrt(number_of_damages) * equavalant_damaged_pipe_diameter
                )

                node = WaterNetwork.get_node(node_name)
                new_elavation = node.elevation

                # Midlle junction definition
                new_coord = (node.coordinates[0] + 10, node.coordinates[1] + 10)
                middle_node_name = 'lk_mdl_' + node_name
                WaterNetwork.add_junction(
                    middle_node_name, elevation=new_elavation, coordinates=new_coord
                )

                # Leak reservoir definition
                new_coord = (new_coord[0] + 10, new_coord[1] + 10)
                new_resevoir_name = 'lk_aux_' + node_name
                WaterNetwork.add_reservoir(
                    new_resevoir_name, base_head=new_elavation, coordinates=new_coord
                )

                # Node-to-middle-junction pipe definition
                OVD_pipe_name = 'lk_ODP_' + node_name  # noqa: N806
                WaterNetwork.add_pipe(
                    OVD_pipe_name,
                    node_name,
                    middle_node_name,
                    length=before_damage_pipe_length,
                    diameter=over_designed_diameter,
                    roughness=_C,
                )

                # Middle_node_to_reservoir pipe definition
                new_pipe_name = 'lk_pipe_' + node_name
                WaterNetwork.add_pipe(
                    new_pipe_name,
                    middle_node_name,
                    new_resevoir_name,
                    length=1,
                    diameter=equavalant_pipe_diameter,
                    roughness=1000000,
                )

                self._registry.explicit_nodal_damages[node_name] = {
                    'ODD',
                    over_designed_diameter,
                }
        else:
            raise ValueError('Unknown nodal damage method')  # noqa: DOC501, EM101, TRY003

        # return WaterNetwork

    def getNd(self, mp, number_of_damages, sum_of_length):  # noqa: N802, D102
        rr = number_of_damages / sum_of_length * 1000

        node_damage_parametrs = self._registry.settings['node_damage_model']
        # {'a':0.0036, 'aa':1, 'b':0, 'bb':0, 'c':-0.877, 'cc':1, 'd':0, 'dd':0, 'e':0.0248, 'ee1':1, 'ee2':1, 'f':0, 'ff1':0, 'ff2':0, "damage_node_model": "equal_diameter_emitter"}
        x = node_damage_parametrs['x']
        a = node_damage_parametrs['a']
        aa = node_damage_parametrs['aa']
        b = node_damage_parametrs['b']
        bb = node_damage_parametrs['bb']
        c = node_damage_parametrs['c']
        cc = node_damage_parametrs['cc']
        d = node_damage_parametrs['d']
        dd = node_damage_parametrs['dd']
        e = node_damage_parametrs['e']
        ee1 = node_damage_parametrs['ee1']
        ee2 = node_damage_parametrs['ee2']
        f = node_damage_parametrs['f']
        ff1 = node_damage_parametrs['ff1']
        ff2 = node_damage_parametrs['ff2']

        # nd = 0.0036*mp + 0.9012 + (0.0248*mp-0.877)*rr
        nd = (
            a * mp**aa
            + b * mp**bb
            + c * rr**cc
            + d * rr**dd
            + e * (mp**ee1) * (rr**ee2)
            + f * (mp**ff1) * (rr**ff2)
            + x
        )
        nd = 0.0036 * float(mp) + 0.9012 + (0.0248 * float(mp) - 0.877) * float(rr)
        return nd  # noqa: RET504

    def getNd2(self, mp, number_of_damages, sum_of_length):  # noqa: N802, D102
        rr = number_of_damages / sum_of_length * 1000

        node_damage_parametrs = self._registry.settings['node_damage_model']
        # {'a':0.0036, 'aa':1, 'b':0, 'bb':0, 'c':-0.877, 'cc':1, 'd':0, 'dd':0, 'e':0.0248, 'ee1':1, 'ee2':1, 'f':0, 'ff1':0, 'ff2':0, "damage_node_model": "equal_diameter_emitter"}
        x = node_damage_parametrs['x']
        a = node_damage_parametrs['a']
        aa = node_damage_parametrs['aa']
        b = node_damage_parametrs['b']
        bb = node_damage_parametrs['bb']
        c = node_damage_parametrs['c']
        cc = node_damage_parametrs['cc']
        d = node_damage_parametrs['d']
        dd = node_damage_parametrs['dd']
        e = node_damage_parametrs['e']
        ee1 = node_damage_parametrs['ee1']
        ee2 = node_damage_parametrs['ee2']
        f = node_damage_parametrs['f']
        ff1 = node_damage_parametrs['ff1']
        ff2 = node_damage_parametrs['ff2']

        nd = (
            a * mp**aa
            + b * mp**bb
            + c * rr**cc
            + d * rr**dd
            + e * (mp**ee1) * (rr**ee2)
            + f * (mp**ff1) * (rr**ff2)
            + x
        )

        return nd  # noqa: RET504

    def getEmitterCdAndElevation(  # noqa: N802, D102
        self,
        real_node_name,
        wn,
        number_of_damages,
        sum_of_length,
        mp,
        q,
    ):
        mp = (  # noqa: PLR6104
            mp * 1.4223
        )  # this is because our CURRENT relationship is base on psi
        rr = number_of_damages / sum_of_length * 1000  # noqa: F841
        nd = self.getNd(mp, number_of_damages, sum_of_length)
        # equavalant_pipe_diameter = ( ((nd-1)*q)**2 /(0.125*9.81*3.14**2 * mp/1.4223) )**(1/4) * 1

        if real_node_name == 'CC1381':
            print(nd)  # noqa: T201
            nd2 = self.getNd2(mp, number_of_damages, sum_of_length)
            print(nd2)  # noqa: T201

        node = wn.get_node(real_node_name)  # noqa: F841
        # new_elavation = node.elevation

        nd = nd - 1  # noqa: PLR6104
        # nd0 = 0.0036*0 + 0.9012 + (0.0248*0-0.877)*rr
        nd0 = self.getNd(0, number_of_damages, sum_of_length)
        if real_node_name == 'CC1381':
            print(nd0)  # noqa: T201
            nd02 = self.getNd2(0, number_of_damages, sum_of_length)
            print(nd02)  # noqa: T201
        nd0 = nd0 - 1  # noqa: PLR6104
        alpha = (nd - nd0) / (mp)
        mp0 = -1 * (nd0) / alpha
        mp0 = mp0 / 1.4223  # noqa: PLR6104
        cd = alpha * q
        return cd, mp0

    def addExplicitLeakWithReservoir(  # noqa: N802, D102
        self,
        node_name,
        number_of_damages,
        sum_of_length,
        wn,
    ):
        method = self._registry.settings['damage_node_model']
        if (
            method == 'equal_diameter_emitter'  # noqa: PLR1714
            or method == 'equal_diameter_reservoir'
        ):
            node = wn.get_node(node_name)
            new_elavation = node.elevation
            new_coord = (node.coordinates[0] + 10, node.coordinates[1] + 10)

            pressure = self._registry.result.node['pressure'][node_name]
            mp = pressure.mean()

            if mp < 0:
                mp = 1
            node = wn.get_node(node_name)
            new_elavation = node.elevation
            new_coord = (node.coordinates[0] + 10, node.coordinates[1] + 10)

            new_node_name = 'lk_aux_' + node_name
            new_pipe_name = 'lk_pipe_' + node_name
            new_C = 100000000000  # noqa: N806

            equavalant_pipe_diameter = 1
            q = node.demand_timeseries_list[0].base_value
            if method == 'equal_diameter_emitter':
                cd, mp0 = self.getEmitterCdAndElevation(
                    node_name, wn, number_of_damages, sum_of_length, mp, q
                )
                wn.add_junction(
                    new_node_name,
                    elevation=new_elavation + mp0,
                    coordinates=new_coord,
                )
                nn = wn.get_node(new_node_name)
                nn._emitter_coefficient = cd  # noqa: SLF001
                wn.options.hydraulic.emitter_exponent = 1
                wn.add_pipe(
                    new_pipe_name,
                    node_name,
                    new_node_name,
                    diameter=equavalant_pipe_diameter,
                    length=1,
                    roughness=new_C,
                    check_valve=True,
                )
                # wn.add_reservoir(new_node_name+'_res', base_head = new_elavation + 10000, coordinates = new_coord)
                # wn.add_pipe(new_pipe_name+'_res', node_name, new_node_name+'_res', diameter=1, length=1, roughness=new_C, check_valve_flag=True)

            elif method == 'equal_diameter_reservoir':
                nd = self.getNd(mp, number_of_damages, sum_of_length)
                equavalant_pipe_diameter = (
                    ((nd - 1) * q) ** 2 / (0.125 * 9.81 * math.pi**2 * mp)
                ) ** (1 / 4) * 1
                wn.add_reservoir(
                    new_node_name, base_head=new_elavation, coordinates=new_coord
                )
                wn.add_pipe(
                    new_pipe_name,
                    node_name,
                    new_node_name,
                    diameter=equavalant_pipe_diameter,
                    length=1,
                    roughness=new_C,
                    check_valve=True,
                    minor_loss=1,
                )
            self._registry.addEquavalantDamageHistory(
                node_name,
                new_node_name,
                new_pipe_name,
                equavalant_pipe_diameter,
                number_of_damages,
            )

        elif method == 'SOD':
            pass
            # first_pipe_length = sum_of_length/5*2
            # second_pipe_length = sum_of_length/5*3
            # new_coord_mid     = (node.xxcoordinates[0]+10,node.coordinates[1]+10)
            # new_coord_dem     = (node.coordinates[0]+20,node.coordinates[1]+20)
            # new_coord_res     = (node.coordinates[0]+10,node.coordinates[1]+20)

        else:
            raise ValueError('Unkown Method')  # noqa: EM101, TRY003
        return new_node_name, new_pipe_name, mp, q

    def estimateNodalDamage(self):  # noqa: N802, D102
        # res = pd.Series()
        temp1 = []
        temp2 = []
        for ind, val in self.node_damage.items():  # noqa: B007, PERF102
            pipes_length = val['node_Pipe_Length']
            pipes_RR = val['node_RR']  # noqa: N806
            temp1.append(val['node_name'])
            temp2.append(int(np.round(pipes_RR * pipes_length)))
        res = pd.Series(data=temp2, index=temp1)
        return res  # noqa: RET504

    def getPipeDamageListAt(self, time):  # noqa: N802, D102
        damaged_pipe_name_list = []

        if self.pipe_all_damages.empty:
            return damaged_pipe_name_list

        current_time_pipe_damages = self.pipe_all_damages[time]
        if type(current_time_pipe_damages) == pd.core.series.Series:  # noqa: E721
            current_time_pipe_damages = current_time_pipe_damages.to_list()
        else:
            current_time_pipe_damages = [current_time_pipe_damages]

        damaged_pipe_name_list = [
            cur_damage['pipe_id'] for cur_damage in current_time_pipe_damages
        ]
        damaged_pipe_name_list = list(set(damaged_pipe_name_list))
        return damaged_pipe_name_list  # noqa: RET504

    def applyPipeDamages(self, WaterNetwork, current_time):  # noqa: C901, N802, N803
        """Apply the damage that we have in damage object. the damage is either
            predicted or read from somewhere.

        Parameters
        ----------
        WaterNetwork : wntrfr.network.model.WaterNetworkModel
            water network model to be modified according to the damage

        registry : Registry object

        current_time : int
            current time

        """  # noqa: D205
        last_pipe_id = None
        same_pipe_damage_cnt = None

        if self.pipe_all_damages.empty:
            print('No Pipe damages at all')  # noqa: T201
            return

        current_time_pipe_damages = self.pipe_all_damages[current_time]
        if type(current_time_pipe_damages) == dict:  # noqa: E721
            current_time_pipe_damages = pd.Series(
                [current_time_pipe_damages], index=[current_time]
            )
        elif type(current_time_pipe_damages) == pd.Series:  # noqa: E721
            if current_time_pipe_damages.empty:
                print('No Pipe damages at time ' + str(current_time))  # noqa: T201
                return
        else:
            raise ValueError(  # noqa: DOC501
                'Pipe damage has a unknown type: '
                + str(type(current_time_pipe_damages))
                + ' at time: '
                + str(current_time)
            )

        all_damages = current_time_pipe_damages.to_list()
        for cur_damage in all_damages:
            # print(cur_damage)

            pipe_id = cur_damage['pipe_id']
            # same_pipe_damage_cnt = 1
            if pipe_id == last_pipe_id:
                same_pipe_damage_cnt += 1
            else:
                last_pipe_id = pipe_id
                same_pipe_damage_cnt = 1

            if cur_damage['type'] == 'leak':
                damage_time = current_time / 3600  # cur_damage['damage_time']
                new_node_id = pipe_id + '_leak_' + repr(same_pipe_damage_cnt)
                new_pipe_id = pipe_id + '_leak_B_' + repr(same_pipe_damage_cnt)
                material = cur_damage['Material']
                area = None
                if 'leakD' in cur_damage:
                    diam = cur_damage['leakD']
                    area = math.pi * (diam / 2) ** 2
                else:
                    # diam = 100*WaterNetwork.get_link(pipe_id).diameter
                    # area= 0.6032*diam/10000
                    # pipe_damage_factor = self.scenario_set['pipe_damage_diameter_factor']
                    diam_m = WaterNetwork.get_link(pipe_id).diameter

                    # print(material)
                    if material in self._registry.settings['pipe_damage_model']:
                        damage_parameters = self._registry.settings[
                            'pipe_damage_model'
                        ][material]
                    else:
                        damage_parameters = self._registry.settings[
                            'default_pipe_damage_model'
                        ]
                    alpha = damage_parameters['alpha']
                    beta = damage_parameters['beta']
                    gamma = damage_parameters['gamma']
                    a = damage_parameters['a']
                    b = damage_parameters['b']

                    dd = alpha * diam_m**a + beta * diam_m**b + gamma
                    dd = dd * 1.2  # noqa: PLR6104

                    area = math.pi * dd**2 / 4
                last_ratio = 1
                if pipe_id in self._pipe_last_ratio:
                    last_ratio = self._pipe_last_ratio.loc[pipe_id]

                ratio = cur_damage['damage_loc'] / last_ratio
                if ratio >= 1:
                    raise ValueError(  # noqa: DOC501
                        'IN LEAK: ratio is bigger than or equal to 1 for pipe:'
                        + repr(pipe_id)
                        + '  '
                        + repr(ratio)
                        + '  '
                        + repr(cur_damage['damage_loc'])
                        + '  '
                        + repr(last_ratio)
                    )
                self._pipe_last_ratio.loc[pipe_id] = ratio

                number = 1
                if 'number' in cur_damage:
                    number = cur_damage['number']

                sub_type = 1
                if 'sub_type' in cur_damage:
                    sub_type = cur_damage['sub_type']

                WaterNetwork = split_pipe(  # noqa: N806
                    WaterNetwork,
                    pipe_id,
                    new_pipe_id,
                    new_node_id,
                    split_at_point=ratio,
                    return_copy=False,
                )
                leak_node = WaterNetwork.get_node(new_node_id)
                leak_node.add_leak(
                    WaterNetwork,
                    area=area,
                    discharge_coeff=1,
                    start_time=damage_time,
                    end_time=self.end_time + 1,
                )
                self._registry.addPipeDamageToRegistry(
                    new_node_id,
                    {
                        'number': number,
                        'damage_type': 'leak',
                        'damage_subtype': sub_type,
                        'pipe_A': pipe_id,
                        'pipe_B': new_pipe_id,
                        'orginal_pipe': pipe_id,
                    },
                )
                # self._registry.addPipeDamageToDamageRestorationData(pipe_id, 'leak', damage_time)

            elif cur_damage['type'] == 'break':
                last_ratio = 1
                if pipe_id in self._pipe_last_ratio:
                    last_ratio = self._pipe_last_ratio.loc[pipe_id]

                ratio = cur_damage['damage_loc'] / last_ratio
                if ratio >= 1:
                    raise ValueError(  # noqa: DOC501
                        'IN BREAK: ratio is bigger than or equal to 1 for pipe:'
                        + repr(pipe_id)
                        + '  '
                        + repr(ratio)
                        + '  '
                        + repr(cur_damage['damage_loc'])
                        + '  '
                        + repr(last_ratio)
                    )

                self._pipe_last_ratio.loc[pipe_id] = ratio

                number = 1
                if 'number' in cur_damage:
                    number = cur_damage['number']

                damage_time = current_time / 3600
                logger.debug(
                    'trying to break: ' + cur_damage['pipe_id'] + repr(damage_time)  # noqa: G003
                )
                # Naming new nodes and new pipe
                new_node_id_for_old_pipe = (
                    pipe_id + '_breakA_' + repr(same_pipe_damage_cnt)
                )
                new_node_id_for_new_pipe = (
                    pipe_id + '_breakB_' + repr(same_pipe_damage_cnt)
                )
                new_pipe_id = pipe_id + '_Break_' + repr(same_pipe_damage_cnt)
                new_node_id = new_node_id_for_old_pipe
                # breaking the node
                WaterNetwork = break_pipe(  # noqa: N806
                    WaterNetwork,
                    pipe_id,
                    new_pipe_id,
                    new_node_id_for_old_pipe,
                    new_node_id_for_new_pipe,
                    split_at_point=ratio,
                    return_copy=False,
                )

                diam = WaterNetwork.get_link(pipe_id).diameter
                area = (diam**2) * math.pi / 4
                break_node_for_old_pipe = WaterNetwork.get_node(
                    new_node_id_for_old_pipe
                )
                break_node_for_old_pipe.add_leak(
                    WaterNetwork,
                    area=area,
                    discharge_coeff=1,
                    start_time=float(damage_time),
                    end_time=self.end_time + 0.1,
                )
                break_node_for_new_pipe = WaterNetwork.get_node(
                    new_node_id_for_new_pipe
                )
                break_node_for_new_pipe.add_leak(
                    WaterNetwork,
                    area=area,
                    start_time=float(damage_time),
                    end_time=self.end_time + 0.1,
                )

                self._registry.addPipeDamageToRegistry(
                    new_node_id_for_old_pipe,
                    {
                        'number': number,
                        'damage_type': 'break',
                        'pipe_A': pipe_id,
                        'pipe_B': new_pipe_id,
                        'orginal_pipe': pipe_id,
                        'node_A': new_node_id_for_old_pipe,
                        'node_B': new_node_id_for_new_pipe,
                    },
                )
                # self._registry.addPipeDamageToDamageRestorationData(pipe_id, 'break', damage_time)
            else:
                raise ValueError(  # noqa: DOC501
                    'undefined damage type: '
                    + repr(cur_damage['type'])
                    + ". Accpetale type of famages are either 'creack' or 'break'."
                )
            self._registry.addRestorationDataOnPipe(
                new_node_id, damage_time, cur_damage['type']
            )
        # return WaterNetwork

    def applyTankDamages(self, WaterNetwork, current_time):  # noqa: N802, N803, D102
        if self.tank_damage.empty:
            print('No Tank Damage at all')  # noqa: T201
            return

        current_time_tank_damage = self.tank_damage[current_time]
        if type(current_time_tank_damage) != str:  # noqa: E721
            if current_time_tank_damage.empty:
                print('No Tank Damage at time ' + str(current_time))  # noqa: T201
                return
        else:
            current_time_tank_damage = pd.Series(
                [current_time_tank_damage], index=[current_time]
            )
        # print(current_time_tank_damage)
        for ind, value in current_time_tank_damage.items():  # noqa: B007, PERF102
            # if value not in WaterNetwork.tank_name_list:
            # continue #contibue if there is not a tank with such damage
            # connected_link_list = []
            link_name_list_connected_to_node = WaterNetwork.get_links_for_node(
                value
            )  # must be here
            # for link_name in link_name_list_connected_to_node:

            # link = WaterNetwork.get_link(link_name)
            # if value == link.start_node.name:
            # connected_link_list.append((0, link_name))
            # elif value == link.end_node.name:
            # connected_link_list.append((1, link_name) )

            tank = WaterNetwork.get_node(value)
            coord = tank.coordinates
            new_coord = (coord[0] + 10, coord[1] + 10)
            elevation = tank.elevation
            new_mid_node_name = value + '_tank_mid'
            WaterNetwork.add_junction(
                new_mid_node_name, elevation=elevation, coordinates=new_coord
            )

            new_pipe_name = value + '_tank_mid_pipe'
            # print(value + str("  -> " ) + new_pipe_name)
            WaterNetwork.add_pipe(
                new_pipe_name, value, new_mid_node_name, initial_status='CLOSED'
            )

            new_node = WaterNetwork.get_node(new_mid_node_name)

            for link_name in link_name_list_connected_to_node:
                link = WaterNetwork.get_link(link_name)

                if value == link.start_node.name:
                    link.start_node = new_node
                elif value == link.end_node.name:
                    link.end_node = new_node
                else:
                    raise  # noqa: PLE0704

    def applyPumpDamages(self, WaterNetwork, current_time):  # noqa: N802, N803, D102
        # print(type(self.damaged_pumps))
        if self.damaged_pumps.empty:
            print('No pump damage at all')  # noqa: T201
            return

        pump_damage_at_time = self.damaged_pumps[current_time]
        if type(pump_damage_at_time) != str:  # noqa: E721
            if pump_damage_at_time.empty:
                print('No Pump Damage at time ' + str(current_time))  # noqa: T201
                return
        else:
            pump_damage_at_time = pd.Series(
                [pump_damage_at_time], index=[current_time]
            )
        for ind, values in pump_damage_at_time.items():  # noqa: B007, PERF102
            WaterNetwork.get_link(values).initial_status = LinkStatus(0)

    def read_earthquake(self, earthquake_file_name):
        """Parameters
        ----------
        earthquake_file_name : str
            path to the text file that include earthquake definition file

        Raises
        ------
        ValueError
            If the file name is not provided, a valueError will be returned
        IOError
            If the information inside the text file is not valid, then IOError
            will be returned

        Returns
        -------
        None.

        """  # noqa: D205, DOC502
        if type(earthquake_file_name) != str:  # noqa: E721
            raise ValueError('string is wanted for earthqiake fie name')  # noqa: EM101, TRY003

        file = open(earthquake_file_name)  # noqa: PLW1514, PTH123, SIM115
        lines = file.readlines()
        ct = 0
        for line in lines:
            ct += 1  # noqa: SIM113
            sline = line.split()
            line_length = len(sline)
            if line_length != 5:  # noqa: PLR2004
                raise OSError(  # noqa: DOC501
                    'there should be 5 values in line '
                    + repr(ct)
                    + '\n M[SPACE]depth[SPACE]X coordinate[SPACE]Y coordinate{SPACE]Time'
                )
            temp_EQ = EarthquakeScenario(  # noqa: N806
                float(sline[0]),
                float(sline[1]),
                float(sline[2]),
                float(sline[3]),
                float(sline[4]),
            )
            self._earthquake = self._earthquake.append(
                pd.Series(temp_EQ, index=[int(temp_EQ.time)])
            )
        file.close()
        self.sortEarthquakeListTimely()

    def sortEarthquakeListTimely(self):  # noqa: N802
        """This functions sorts the list of earthquakes in a timely manner

        Returns
        -------
        None.

        """  # noqa: D400, D401, D404
        self._earthquake.sort_index()
        self.is_timely_sorted = True

    def predictDamage(self, wn, iClear=False):  # noqa: FBT002, N802, N803
        """This function predict the water network model damage based on  probabilistic method.

        Parameters
        ----------
        wn : wntrfr.network.model.WaterNetworkModel
            Water Network Model to be used to model the damages
        clear : TYPE, optional
            Boolian value, determining if the leak and break list must be
            cleared before predicting and adding. The default is False.

        Returns
        -------
        None.

        """  # noqa: D401, D404
        if iClear:
            self.pipe_leak = pd.Series()
            self.pipe_break = pd.Series()

        for eq_in, eq in self._earthquake.items():  # noqa: B007, PERF102
            wntr_eq = eq.getWNTREarthquakeObject()
            distance_to_pipes = wntr_eq.distance_to_epicenter(
                wn, element_type=wntrfr.network.Pipe
            )
            pga = wntr_eq.pga_attenuation_model(distance_to_pipes)
            pgv = wntr_eq.pgv_attenuation_model(distance_to_pipes)
            repair_rate = wntr_eq.repair_rate_model(pgv)  # noqa: F841
            fc = wntrfr.scenario.FragilityCurve()
            fc.add_state('leak', 1, {'Default': lognorm(0.5, scale=0.2)})
            fc.add_state('break', 2, {'Default': lognorm(0.5, scale=0.5)})
            failure_probability = fc.cdf_probability(pga)
            damage_state = fc.sample_damage_state(failure_probability)

            for pipe_ID, ds in damage_state.items():  # noqa: N806
                # if wn.get_link(pipe_ID).status==0:
                # continue
                if ds == None:  # noqa: E711
                    continue
                if ds.lower() == 'leak':
                    temp = {
                        'pipe_id': pipe_ID,
                        'leak_loc': 0.5,
                        'leak_type': 1,
                        'leak_time': eq.time / 3600,
                    }
                    self.pipe_leak = self.pipe_leak.append(
                        pd.Series(data=[temp], index=[int(eq.time)])
                    )
                if ds.lower() == 'break':
                    temp = {
                        'pipe_id': pipe_ID,
                        'break_loc': 0.5,
                        'break_time': eq.time / 3600,
                    }
                    self.pipe_break = self.pipe_break.append(
                        pd.Series(data=[temp], index=[int(eq.time)])
                    )

    def get_damage_distinct_time(self):
        """Get distinct time for all kind of damages

        Returns
        -------
        damage_time_list : list
            Distinct time for all kind of damages

        """  # noqa: D400
        pipe_damage_unique_time = self.pipe_all_damages.index.unique().tolist()
        node_damage_unique_time = self.node_damage.index.unique().tolist()
        tank_damage_unique_time = self.tank_damage.index.unique().tolist()
        pump_damage_unique_time = self.damaged_pumps.index.unique().tolist()

        all_damages_time = []
        all_damages_time.extend(pipe_damage_unique_time)
        all_damages_time.extend(node_damage_unique_time)
        all_damages_time.extend(tank_damage_unique_time)
        all_damages_time.extend(pump_damage_unique_time)

        all_damages_time = list(set(all_damages_time))
        all_damages_time.sort()

        # damage_time_list     = all_pipe_damage_time.unique().tolist()
        # damage_time_list.sort()
        return all_damages_time

    def get_earthquake_distict_time(self):
        """Checks if the earthquake time are in order. Then the it will get
        distinct earthquake time sand return it

        Raises
        ------
        ValueError
            when the earthquake in not in order.

        Returns
        -------
        pandas.Series()
            a list of distinct time of earthquake.

        """  # noqa: D205, D400, D401, DOC502
        reg = []
        if self.is_timely_sorted == False:  # noqa: E712
            self.sortEarthquakeListTimely()

        time_list = self._earthquake.index
        last_value = None
        for time in iter(time_list):
            if last_value == None or last_value < time:  # noqa: E711
                reg.append(time)
                last_value = time

        return pd.Series(reg)
