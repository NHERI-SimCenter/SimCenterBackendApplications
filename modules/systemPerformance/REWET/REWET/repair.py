"""Created on Tue Feb  2 20:22:09 2021

@author: snaeimi
"""  # noqa: D400, D415

from collections import OrderedDict

from wntrfr.network.model import LinkStatus

LINK_TYPE_COLLECTIVES = {
    'BYPASS_PIPE',
    'ADDED_PIPE_A',
    'ADDED_PIPE_B',
    'ADDED_PIPE_C',
    'ADDED_PIPE_D',
    'ADDED_PUMP_A',
    'ADDED_PUMP_B',
    'PIPE_CLOSED_FROM_OPEN',
    'PIPE_CLOSED_FROM_CV',
}
NODE_TYPE_COLLECTIVES = {
    'MIDDLE_NODE_A',
    'MIDDLE_NODE_B',
    'ADDED_RESERVOIR_A',
    'ADDED_RESERVOIR_B',
}
NON_COLLECTIVES = {
    'NODE_DEMAND_AFTER',
    'NODE_DEMAND_BEFORE',
    'NODE_A_DEMAND_BEFORE',
    'NODE_A_DEMAND_AFTER',
    'NODE_B_DEMAND_BEFORE',
    'NODE_B_DEMAND_AFTER',
    'NODE_A',
    'NODE_B',
    'NON_COL_ADDED_PIPE',
    'NON_COL_PIPE_CLOSED_FROM_OPEN',
    'NON_COL_PIPE_CLOSED_FROM_CV',
}
NC_FALSE_FLAG = {
    'NODE_A',
    'NODE_B',
    'NODE_A_DEMAND_BEFORE',
    'NODE_A_DEMAND_AFTER',
    'NODE_B_DEMAND_BEFORE',
    'NODE_B_DEMAND_AFTER',
    'NODE_DEMAND_AFTER',
    'NODE_DEMAND_BEFORE',
}


class Repair:  # noqa: D101
    def __init__(self, registry):  # noqa: ANN001, ANN204, D107
        self._registry = registry

    def closeSecondLeakingPipe(self, damage_node_name, wn):  # noqa: ANN001, ANN201, N802, D102
        if (
            self._registry.getDamageData('PIPE', False).loc[  # noqa: FBT003
                damage_node_name, 'damage_type'
            ]
            != 'leak'
        ):
            raise ValueError('Damage type is not leak in node ' + damage_node_name)

        pipe_A_name, pipe_B_name, orginal_pipe_name = self._registry.getLeakData(  # noqa: N806
            damage_node_name
        )
        pipe_B = wn.get_link(pipe_B_name)  # noqa: N806

        pipe_B.status = LinkStatus.Closed
        pipe_B.initial_status = LinkStatus.Closed

    def bypassPipe(  # noqa: ANN201, N802, D102, PLR0913
        self,
        damage_node_name,  # noqa: ANN001
        middle_pipe_size,  # noqa: ANN001
        damage_type,  # noqa: ANN001
        wn,  # noqa: ANN001
        length=None,  # noqa: ANN001
        friction=None,  # noqa: ANN001
    ):
        # if self._registry.getDamageData('PIPE',False).loc[damage_node_name,'damage_type'] != 'leak':  # noqa: E501
        # raise ValueError('Damage type is not leak in node '+damage_node_name)  # noqa: ERA001

        if damage_type == 'leak':
            pipe_A_name, pipe_B_name, orginal_pipe_name = self._registry.getLeakData(  # noqa: N806
                damage_node_name
            )
        elif damage_type == 'break':
            pipe_A_name, pipe_B_name, orginal_pipe_name, node_A_name, node_B_name = (  # noqa: N806
                self._registry.getBreakData(damage_node_name)
            )
        org_pipe_data = self._registry.getOriginalPipenodes(orginal_pipe_name)

        orginal_node_A_name = org_pipe_data['start_node_name']  # noqa: N806
        orginal_node_B_name = org_pipe_data['end_node_name']  # noqa: N806
        orginal_pipe_length = org_pipe_data['length']
        orginal_pipe_roughness = org_pipe_data['roughness']

        if length != None:  # noqa: SIM108, E711
            pipe_length = length
        else:
            pipe_length = orginal_pipe_length

        if friction != None:  # noqa: SIM108, E711
            pipe_friction = friction
        else:
            pipe_friction = orginal_pipe_roughness
        new_pipe_name = orginal_pipe_name + '-Byp'
        if middle_pipe_size > 0:
            wn.add_pipe(
                new_pipe_name,
                orginal_node_A_name,
                orginal_node_B_name,
                length=pipe_length,
                diameter=middle_pipe_size,
                roughness=pipe_friction,
            )

            # For the sake of multiple damages in one pipe the following line is marked the the line after it is added  # noqa: E501

        damage_data = self._registry.getDamageData('pipe', iCopy=False)
        redefined_damage_data = damage_data[
            damage_data['Orginal_element'] == orginal_pipe_name
        ]

        for cur_damage_node_name, cur_damage in redefined_damage_data.iterrows():
            history = OrderedDict()
            if middle_pipe_size > 0:
                history['BYPASS_PIPE'] = (
                    new_pipe_name  # Bypass pipe doesn't get removed unless all damages in the orginal pipe is removed  # noqa: E501
                )

            cur_damage_type = cur_damage['damage_type']
            if cur_damage_type == 'leak':
                pipe_A_name, pipe_B_name, orginal_pipe_name = (  # noqa: N806
                    self._registry.getLeakData(cur_damage_node_name)
                )

                pipe_B = wn.get_link(pipe_B_name)  # noqa: N806, F841

            elif cur_damage_type == 'break':
                pass

            else:
                raise ValueError('Unrecognozed damaged type: ' + cur_damage_type)

            self._registry.addFunctionDataToRestorationRegistry(
                cur_damage_node_name, history, 'bypass'
            )

    # local reconnection, for instance, for fire truck reconnection
    def reconnectPipe(self, damage_node_name, middle_pipe_size, damage_type, wn):  # noqa: ANN001, ANN201, N802, D102
        history = OrderedDict()

        if damage_type == 'leak':
            pipe_A_name, pipe_B_name, orginal_pipe_name = self._registry.getLeakData(  # noqa: N806
                damage_node_name
            )

            pipe_A = wn.get_link(pipe_A_name)  # noqa: N806
            pipe_B = wn.get_link(pipe_B_name)  # noqa: N806

            if pipe_A.status == 1:
                history['NON_COL_PIPE_CLOSED_FROM_OPEN'] = pipe_A_name
            elif pipe_A.status == 3:  # noqa: PLR2004
                history['NON_COL_PIPE_CLOSED_FROM_CV'] = pipe_A_name
            pipe_A.initial_status = LinkStatus(0)

            if middle_pipe_size == None:  # noqa: E711
                middle_pipe_size = pipe_A.diameter

            beg_node_of_pipe_A = pipe_A.start_node  # noqa: N806
            end_node_of_pipe_B = pipe_B.end_node  # noqa: N806
            new_length = pipe_A.length + pipe_B.length

            # For the sake of multiple damages in one pipe the following line is marked the the line after it is added  # noqa: E501
            new_pipe_name = pipe_B_name + '-Red'

            wn.add_pipe(
                new_pipe_name,
                beg_node_of_pipe_A.name,
                end_node_of_pipe_B.name,
                length=new_length,
                diameter=middle_pipe_size,
                roughness=pipe_A.roughness,
            )

            history['NON_COL_ADDED_PIPE'] = new_pipe_name

        elif damage_type == 'break':
            pipe_A_name, pipe_B_name, orginal_pipe_name, node_A_name, node_B_name = (  # noqa: N806
                self._registry.getBreakData(damage_node_name)
            )

            pipe_A = wn.get_link(pipe_A_name)  # noqa: N806
            pipe_B = wn.get_link(pipe_B_name)  # noqa: N806

            if middle_pipe_size == None:  # noqa: E711
                middle_pipe_size = pipe_A.diameter

            beg_node_of_pipe_A = pipe_A.start_node  # noqa: N806
            end_node_of_pipe_B = pipe_B.end_node  # noqa: N806
            new_length = pipe_A.length + pipe_B.length

            # For the sake of multiple damages in one pipe the following line is marked the the line after it is added  # noqa: E501
            new_pipe_name = pipe_B_name + '-Red'

            wn.add_pipe(
                new_pipe_name,
                beg_node_of_pipe_A.name,
                end_node_of_pipe_B.name,
                length=new_length,
                diameter=middle_pipe_size,
                roughness=pipe_A.roughness,
            )
            history['NON_COL_ADDED_PIPE'] = new_pipe_name

        else:
            raise ValueError('Unrecognozed damaged type: ' + damage_type)

        self._registry.addFunctionDataToRestorationRegistry(
            damage_node_name, history, 'reconnect'
        )

    def removeLeak(self, damage_node_name, damage_type, wn, factor=1):  # noqa: ANN001, ANN201, C901, N802, D102, PLR0912, PLR0915
        history = OrderedDict()

        opening = 1 - factor

        damage_data = self._registry.getDamageData('pipe', iCopy=False)
        orginal_pipe_name = damage_data.loc[damage_node_name, 'Orginal_element']
        refined_damage_data = damage_data[
            damage_data['Orginal_element'] == orginal_pipe_name
        ]

        damaged_node_name_list_on_orginal_pipe = refined_damage_data.index.to_list()
        damage_type_list = refined_damage_data['damage_type'].to_dict()

        for cur_damage_node_name in damaged_node_name_list_on_orginal_pipe:
            cur_damage_type = damage_type_list[cur_damage_node_name]

            if cur_damage_type == 'leak':
                pipe_A_name, pipe_B_name, orginal_pipe_name = (  # noqa: N806
                    self._registry.getLeakData(cur_damage_node_name)
                )

                node_A = wn.get_node(cur_damage_node_name)  # noqa: N806

                if pipe_B_name in wn.pipe_name_list:
                    pipe_B = wn.get_link(pipe_B_name)  # noqa: N806

                    if pipe_B.status == 1:
                        history['PIPE_CLOSED_FROM_OPEN'] = pipe_B_name
                    elif pipe_B.status == 3:  # noqa: PLR2004
                        history['PIPE_CLOSED_FROM_CV'] = pipe_B_name

                    pipe_B.initial_status = LinkStatus(0)
                history['NODE_A_DEMAND_BEFORE'] = node_A._leak_area  # noqa: SLF001
                node_A_leak_area = opening * node_A._leak_area  # noqa: SLF001, N806
                node_A.add_leak(wn, node_A_leak_area, discharge_coeff=1)
                history['NODE_A_DEMAND_AFTER'] = node_A._leak_area  # noqa: SLF001

                if abs(opening) < 0.001:  # noqa: PLR2004
                    node_A.remove_leak(wn)
                    history['NODE_A'] = 'REMOVED'
                else:
                    history['NODE_A'] = 'REDUCED'

            elif cur_damage_type == 'break':
                (
                    pipe_A_name,  # noqa: N806
                    pipe_B_name,  # noqa: N806
                    orginal_pipe_name,
                    node_A_name,  # noqa: N806
                    node_B_name,  # noqa: N806
                ) = self._registry.getBreakData(cur_damage_node_name)
                if cur_damage_node_name != node_A_name:
                    raise ValueError(
                        'Cur damage and pipe_name are not the same: '
                        + repr(cur_damage_node_name)
                        + ' - '
                        + repr(node_A_name)
                    )

                node_A = wn.get_node(cur_damage_node_name)  # noqa: N806

                history['NODE_A_DEMAND_BEFORE'] = node_A._leak_area  # noqa: SLF001
                node_A_leak_area = opening * node_A._leak_area  # noqa: SLF001, N806
                node_A.add_leak(wn, node_A_leak_area, discharge_coeff=1)

                history['NODE_A_DEMAND_AFTER'] = node_A._leak_area  # noqa: SLF001

                if abs(opening) < 0.001:  # noqa: PLR2004
                    node_A.remove_leak(wn)
                    node_A._leak_area = 0  # noqa: SLF001
                    history['NODE_A'] = 'REMOVED'
                else:
                    history['NODE_A'] = 'REDUCED'
                node_B = wn.get_node(node_B_name)  # noqa: N806

                history['NODE_B_DEMAND_BEFORE'] = node_B._leak_area  # noqa: SLF001
                node_B_leak_area = opening * node_B._leak_area  # noqa: SLF001, N806
                node_B.add_leak(wn, node_B_leak_area, discharge_coeff=1)
                history['NODE_B_DEMAND_AFTER'] = node_B._leak_area  # noqa: SLF001

                if abs(opening) < 0.001:  # noqa: PLR2004
                    node_B.remove_leak(wn)
                    node_B._leak_area = 0  # noqa: SLF001
                    history['NODE_B'] = 'REMOVED'
                else:
                    history['NODE_B'] = 'REDUCED'

            else:
                raise ValueError('Unknown Damage type:' + repr(damage_type))

        self._registry.addFunctionDataToRestorationRegistry(
            damage_node_name, history, 'removeLeak'
        )

    def addReservoir(self, damage_node_name, damage_type, _type, pump, wn):  # noqa: ANN001, ANN201, C901, N802, D102, PLR0912, PLR0915
        history = OrderedDict()

        if damage_type == 'leak':
            pipe_A_name, pipe_B_name, orginal_pipe_name = self._registry.getLeakData(  # noqa: N806
                damage_node_name
            )
        elif damage_type == 'break':
            pipe_A_name, pipe_B_name, orginal_pipe_name, node_A_name, node_B_name = (  # noqa: N806
                self._registry.getBreakData(damage_node_name)
            )
        else:
            raise ValueError(
                'Unknown damage type in ' + damage_node_name + ', ' + damage_type
            )

        pipe_A = wn.get_link(pipe_A_name)  # noqa: N806
        pipe_B = wn.get_link(pipe_B_name)  # noqa: N806
        first_node_pipe_A = pipe_A.start_node  # noqa: N806
        second_node_pipe_B = pipe_B.end_node  # noqa: N806

        _coord_A = (  # noqa: N806
            first_node_pipe_A.coordinates[0] + 10,
            first_node_pipe_A.coordinates[1] + 10,
        )
        new_reservoir_A = first_node_pipe_A.name + '-added'  # noqa: N806
        wn.add_reservoir(
            new_reservoir_A,
            base_head=first_node_pipe_A.elevation,
            coordinates=_coord_A,
        )

        _coord_B = (  # noqa: N806
            second_node_pipe_B.coordinates[0] + 10,
            second_node_pipe_B.coordinates[1] + 10,
        )
        new_reservoir_B = second_node_pipe_B.name + '-added'  # noqa: N806
        wn.add_reservoir(
            new_reservoir_B,
            base_head=second_node_pipe_B.elevation,
            coordinates=_coord_B,
        )
        history['ADDED_RESERVOIR_A'] = new_reservoir_A
        history['ADDED_RESERVOIR_B'] = new_reservoir_B

        if _type == None:  # noqa: E711
            _pipe_size = pipe_A.diameter
            new_pipe_name_1 = damage_node_name + '-lK1'
            new_pipe_name_2 = damage_node_name + '-lK2'
            wn.add_pipe(
                new_pipe_name_1,
                new_reservoir_A,
                first_node_pipe_A.name,
                diameter=_pipe_size,
                length=5,
                check_valve=True,
            )
            history['ADDED_PIPE_A'] = (
                new_pipe_name_1  # ٌIt Added Pipe is collective now. Won't be removed till all damaegs in the pipe is removed  # noqa: E501
            )
            wn.add_pipe(
                new_pipe_name_2,
                new_reservoir_B,
                second_node_pipe_B.name,
                diameter=_pipe_size,
                length=5,
                check_valve=True,
            )
            history['ADDED_PIPE_B'] = (
                new_pipe_name_2  # ٌIt Added Pipe is collective now. Won't be removed till all damaegs in the pipe is removed  # noqa: E501
            )

        elif _type == 'PUMP':
            if 'POWER' in pump:
                _power = pump['POWER']
                new_pump_name_1 = damage_node_name + '-RP1'
                new_pump_name_2 = damage_node_name + '-RP2'
                wn.add_pump(
                    new_pump_name_1,
                    new_reservoir_A,
                    first_node_pipe_A.name,
                    pump_parameter=_power,
                )
                wn.add_pump(
                    new_pump_name_2,
                    new_reservoir_B,
                    second_node_pipe_B.name,
                    pump_parameter=_power,
                )
                history['ADDED_PUMP_A'] = (
                    new_pump_name_1  # ٌIt Added Pumps is collective now. Won;t be removed till all damaegs in the pipe is removed  # noqa: E501
                )
                history['ADDED_PUMP_B'] = (
                    new_pump_name_2  # ٌIt Added Pumps is collective now. Won;t be removed till all damaegs in the pipe is removed  # noqa: E501
                )
            else:
                raise ValueError('Invalid Pump Type: ' + repr(pump.keys()))
        elif _type == 'ADDEDELEVATION':
            _pipe_size = pipe_A.diameter
            new_pipe_name_1 = damage_node_name + '-RP1'
            new_pipe_name_2 = damage_node_name + '-RP2'

            new_valve_name_1 = damage_node_name + '-RV1'
            new_valve_name_2 = damage_node_name + '-RV2'

            new_RP_middle_name1 = damage_node_name + '-mn1'  # noqa: N806
            new_RP_middle_name2 = damage_node_name + '-mn2'  # noqa: N806

            coord1 = (
                first_node_pipe_A.coordinates[0] + 5,
                first_node_pipe_A.coordinates[1] + 5,
            )
            coord2 = (
                second_node_pipe_B.coordinates[0] + 5,
                second_node_pipe_B.coordinates[1] + 5,
            )

            elavation1 = first_node_pipe_A.elevation
            elavation2 = second_node_pipe_B.elevation

            wn.add_junction(
                new_RP_middle_name1, elevation=elavation1, coordinates=coord1
            )
            wn.add_junction(
                new_RP_middle_name2, elevation=elavation2, coordinates=coord2
            )

            wn.add_pipe(
                new_pipe_name_1,
                new_reservoir_A,
                new_RP_middle_name1,
                diameter=_pipe_size,
                length=1,
                roughness=100000000,
                minor_loss=7,
                check_valve=True,
            )
            wn.add_pipe(
                new_pipe_name_2,
                new_reservoir_B,
                new_RP_middle_name2,
                diameter=_pipe_size,
                length=1,
                roughness=100000000,
                minor_loss=7,
                check_valve=True,
            )

            wn.add_valve(
                new_valve_name_1,
                new_RP_middle_name1,
                first_node_pipe_A.name,
                valve_type='FCV',
                setting=0.2500,
            )
            wn.add_valve(
                new_valve_name_2,
                new_RP_middle_name2,
                second_node_pipe_B.name,
                valve_type='FCV',
                setting=0.2500,
            )

            res_A = wn.get_node(new_reservoir_A)  # noqa: N806
            res_B = wn.get_node(new_reservoir_B)  # noqa: N806

            res_A.base_head = res_A.base_head + 20
            res_B.base_head = res_B.base_head + 20

            history['MIDDLE_NODE_A'] = (
                new_RP_middle_name1  # ٌIt Added Pipe is collective now. Won't be removed till all damaegs in the pipe is removed  # noqa: E501
            )
            history['MIDDLE_NODE_B'] = (
                new_RP_middle_name2  # ٌIt Added Pipe is collective now. Won't be removed till all damaegs in the pipe is removed  # noqa: E501
            )
            history['ADDED_PIPE_A'] = (
                new_pipe_name_1  # ٌIt Added Pipe is collective now. Won't be removed till all damaegs in the pipe is removed  # noqa: E501
            )
            history['ADDED_PIPE_B'] = (
                new_pipe_name_2  # ٌIt Added Pipe is collective now. Won't be removed till all damaegs in the pipe is removed  # noqa: E501
            )
            history['ADDED_PIPE_C'] = (
                new_valve_name_1  # ٌIt Added Pipe is collective now. Won't be removed till all damaegs in the pipe is removed  # noqa: E501
            )
            history['ADDED_PIPE_D'] = (
                new_valve_name_2  # ٌIt Added Pipe is collective now. Won't be removed till all damaegs in the pipe is removed  # noqa: E501
            )

        else:
            raise ValueError('Unknown Reservoir type')  # noqa: EM101, TRY003

        damage_data = self._registry.getDamageData('pipe', iCopy=False)
        redefined_damage_data = damage_data[
            damage_data['Orginal_element'] == orginal_pipe_name
        ]

        for cur_damage_node_name, cur_damage in redefined_damage_data.iterrows():
            cur_damage_type = cur_damage['damage_type']
            if cur_damage_type == 'leak':
                pipe_A_name, pipe_B_name, orginal_pipe_name = (  # noqa: N806
                    self._registry.getLeakData(cur_damage_node_name)
                )

                pipe_B = wn.get_link(pipe_B_name)  # noqa: N806

                if pipe_B.status == 1:
                    history['PIPE_CLOSED_FROM_OPEN'] = pipe_B_name
                elif pipe_B.status == 3:  # noqa: PLR2004
                    history['PIPE_CLOSED_FROM_CV'] = pipe_B_name

                pipe_B.initial_status = LinkStatus(0)

            elif cur_damage_type == 'break':
                pass
            else:
                raise ValueError('Unrecognozed damaged type: ' + cur_damage_type)

        self._registry.addFunctionDataToRestorationRegistry(
            damage_node_name, history, 'addReservoir'
        )

    def removeDemand(self, node_name, factor, wn):  # noqa: ANN001, ANN201, N802, D102
        history = OrderedDict()

        if factor < 0 or factor > 1:
            raise ValueError(
                'In node ' + node_name + ' factor is not valid: ' + repr(factor)
            )

        demand_after_removal_factor = 1 - factor
        node = wn.get_node(node_name)
        cur_demand = node.demand_timeseries_list[0].base_value
        # self._registry.getDamageData('DISTNODE', iCopy=False).loc[node_name,'Demand1'] = cur_demand  # noqa: ERA001, E501
        history['NODE_DEMAND_BEFORE'] = cur_demand
        if abs(cur_demand) < 0:
            return ValueError(
                'Node ' + repr(node_name) + ' is has zerovalue: ' + repr(cur_demand)
            )

        new_demand = demand_after_removal_factor * cur_demand
        node.demand_timeseries_list[0].base_value = new_demand
        history['NODE_DEMAND_AFTER'] = new_demand

        self._registry.addFunctionDataToRestorationRegistry(  # noqa: RET503
            node_name, history, 'removeDemand'
        )

    def removeExplicitNodalLeak(self, node_name, factor, wn):  # noqa: ANN001, ANN201, N802, D102
        history = OrderedDict()
        damage_data = self._registry.getEquavalantDamageHistory(node_name)
        pipe_name = damage_data['new_pipe_name']

        current_number_of_damages = damage_data['current_number_of_damage']
        if factor == 1:
            pipe = wn.get_link(pipe_name)
            pipe.cv = False
            pipe.initial_status = LinkStatus(0)
            history['EXPLICIT_PIPE_CLOSED_FROM_CV'] = pipe_name
        else:
            ned = self._registry.nodal_equavalant_diameter
            pipe = wn.get_link(pipe_name)
            diameter = pipe.diameter
            diameter = (factor**0.5) * current_number_of_damages * ned
            history['EXPLICIT_PIPE_DIAMETER_CHANAGED'] = diameter
            pipe.diameter = diameter

        self._registry.addFunctionDataToRestorationRegistry(
            node_name, history, 'removeExplicitLeak'
        )

    def removeNodeTemporaryRepair(self, damage_node_name, wn):  # noqa: ANN001, ANN201, N802, D102
        if_damage_removed = False

        restoration_table = self._registry._restoration_table  # noqa: SLF001
        selected_restoration_table = restoration_table[
            restoration_table['node_name'] == damage_node_name
        ]

        for ind, rec_id in selected_restoration_table.record_index.items():  # noqa: B007, PERF102
            change_list = self._registry._record_registry[rec_id]  # noqa: SLF001

            for change, name in ((k, change_list[k]) for k in reversed(change_list)):  # noqa: B007
                if change == 'removeExplicitLeak':
                    pass

                elif change == 'NODE_DEMAND_AFTER' or change == 'NODE_DEMAND_BEFORE':  # noqa: PLR1714
                    if (
                        self._registry.settings['damage_node_model']
                        == 'Predefined_demand'
                    ):
                        self.repair.reduceDemand()
                    elif (
                        self._registry.settings['damage_node_model']
                        == 'equal_diameter_emitter'
                    ) or (
                        self._registry.settings['damage_node_model']
                        == 'equal_diameter_reservoir'
                    ):
                        self.restoreDistributionOrginalDemand(damage_node_name, wn)
                    else:
                        raise ValueError('unknow method')  # noqa: EM101, TRY003

        if if_damage_removed == False:  # noqa: E712
            self.removeDISTNodeExplicitLeak(damage_node_name, wn)

    def removePipeRepair(self, damaged_node_name, wn, action):  # noqa: ANN001, ANN201, C901, N802, D102, PLR0912
        restoration_table = self._registry._restoration_table  # noqa: SLF001
        selected_restoration_table = restoration_table[
            restoration_table['node_name'] == damaged_node_name
        ]

        for ind, rec_id in selected_restoration_table.record_index.items():
            change_list = self._registry._record_registry[rec_id]  # noqa: SLF001

            to_pop_list = []

            for change, name in ((k, change_list[k]) for k in reversed(change_list)):
                flag = True
                if change == 'ADDED_PIPE' or change == 'ADDED_PUMP':  # noqa: PLR1714
                    wn.remove_link(name)

                i_link_collective = False
                i_node_collective = False
                if change in LINK_TYPE_COLLECTIVES:
                    i_link_collective = True
                if change in NODE_TYPE_COLLECTIVES:
                    i_node_collective = True

                if i_link_collective or i_node_collective:
                    damage_data = self._registry.getDamageData('pipe', iCopy=False)
                    orginal_pipe_name = damage_data.loc[
                        damaged_node_name, 'Orginal_element'
                    ]
                    refined_damage_data = damage_data[
                        (damage_data['Orginal_element'] == orginal_pipe_name)
                        & (damage_data['discovered'] == True)  # noqa: E712
                    ]
                    if (refined_damage_data[action] == True).all():  # noqa: E712
                        if i_link_collective:
                            if (
                                change == 'BYPASS_PIPE'  # noqa: PLR1714
                                or change == 'ADDED_PIPE_A'
                                or (
                                    change == 'ADDED_PIPE_B'  # noqa: PLR1714
                                    or change == 'ADDED_PIPE_C'
                                )
                                or (
                                    change == 'ADDED_PIPE_D'  # noqa: PLR1714
                                    or change == 'ADDED_PUMP_A'
                                    or change == 'ADDED_PUMP_B'
                                )
                            ):
                                wn.remove_link(name)
                            elif change == 'PIPE_CLOSED_FROM_OPEN':
                                if name in wn.pipe_name_list:
                                    wn.get_link(name).initial_status = LinkStatus(1)
                            elif change == 'PIPE_CLOSED_FROM_CV':
                                if name in wn.pipe_name_list:
                                    wn.get_link(name).initial_status = LinkStatus(3)
                            else:
                                raise ValueError(
                                    'Unknown change indicator in restoration registry: '  # noqa: E501
                                    + repr(change)
                                )

                        elif i_node_collective:
                            wn.remove_node(name)
                        else:
                            raise ValueError(
                                'Unknown change indicator in restoration registry: '
                                + repr(change)
                            )

                elif change in NON_COLLECTIVES:
                    if change == 'NON_COL_ADDED_PIPE':
                        wn.remove_link(name)
                    elif change == 'NON_COL_PIPE_CLOSED_FROM_OPEN':
                        wn.get_link(name).initial_status = LinkStatus(1)
                    elif change in NC_FALSE_FLAG:
                        flag = False
                    else:
                        raise ValueError(
                            'Unknown change indicator in restoration registry: '
                            + repr(change)
                        )

                else:
                    raise ValueError(
                        'Unknown change indicator in restoration registry: '
                        + repr(change)
                    )

                if flag:
                    to_pop_list.append(change)

            for pop_key in to_pop_list:
                change_list.pop(pop_key)

            if len(change_list) == 0:
                restoration_table.drop(ind, inplace=True)  # noqa: PD002

    def repairPipe(self, damage_node_name, damage_type, wn):  # noqa: ANN001, ANN201, N802, D102
        if damage_type == 'leak':
            pipe_A_name, pipe_B_name = self._registry.getCertainLeakData(  # noqa: N806
                damage_node_name, wn
            )

            pipe_A = wn.get_link(pipe_A_name)  # noqa: N806
            pipe_B = wn.get_link(pipe_B_name)  # noqa: N806

            end_node_of_pipe_B = pipe_B.end_node  # noqa: N806
            new_length = pipe_A.length + pipe_B.length

            pipe_A.length = new_length
            pipe_A.end_node = end_node_of_pipe_B

            wn.remove_link(pipe_B_name)
            wn.remove_node(damage_node_name, with_control=True)

        elif damage_type == 'break':
            pipe_A_name, pipe_B_name, node_A_name, node_B_name = (  # noqa: N806
                self._registry.getCertainBreakData(damage_node_name, wn)
            )

            pipe_A = wn.get_link(pipe_A_name)  # noqa: N806
            pipe_B = wn.get_link(pipe_B_name)  # noqa: N806

            end_node_of_pipe_B = pipe_B.end_node  # noqa: N806
            new_length = pipe_A.length + pipe_B.length

            pipe_A.length = new_length
            pipe_A.end_node = end_node_of_pipe_B

            wn.remove_link(pipe_B_name)
            wn.remove_node(node_A_name, with_control=True)
            wn.remove_node(node_B_name, with_control=True)

    def restorePumps(self, pump_name_list, wn):  # noqa: ANN001, ANN201, N802, D102
        for pump_name in pump_name_list:
            wn.get_link(pump_name).initial_status = LinkStatus(1)

    def restoreTanks(self, tank_name_list, wn):  # noqa: ANN001, ANN201, N802, D102
        for tank_name in tank_name_list:
            made_up_mid_node_name = tank_name + '_tank_mid'
            made_up_pipe_name = tank_name + '_tank_mid_pipe'

            wn.remove_link(made_up_pipe_name)

            link_name_list_connected_to_node = wn.get_links_for_node(
                made_up_mid_node_name
            )

            tank_node = wn.get_node(tank_name)
            for link_name in link_name_list_connected_to_node:
                link = wn.get_link(link_name)
                if made_up_mid_node_name == link.start_node.name:
                    link.start_node = tank_node
                elif made_up_mid_node_name == link.end_node.name:
                    link.end_node = tank_node

            wn.remove_node(made_up_mid_node_name, with_control=True)

    def removeDISTNodeIsolation(self, damaged_node_name, wn):  # noqa: ANN001, ANN201, N802, D102
        post_incident_node_demand = self._registry.getDamageData('DISTNODE').loc[
            damaged_node_name, 'Demand2'
        ]

        node = wn.get_node(damaged_node_name)
        node.demand_timeseries_list[0].base_value = post_incident_node_demand

    def restoreDistributionOrginalDemand(self, damaged_node_name, wn):  # noqa: ANN001, ANN201, N802, D102
        if self._registry.settings['damage_node_model'] == 'Predefined_demand':
            pre_incident_node_demand = self._registry.getDamageData(
                'DISTNODE', iCopy=False
            ).loc[damaged_node_name, 'Demand1']
        elif (
            self._registry.settings['damage_node_model'] == 'equal_diameter_emitter'
            or self._registry.settings['damage_node_model']
            == 'equal_diameter_reservoir'
        ):
            damage_table = self._registry.getDamageData('DISTNODE', iCopy=False)
            virtual_nodes_damage_tabel = damage_table[
                damage_table['virtual_of'] == damaged_node_name
            ]
            pre_incident_node_demand = virtual_nodes_damage_tabel.iloc[0]['Demand1']
        else:
            raise ValueError('unknow method')  # noqa: EM101, TRY003

        node = wn.get_node(damaged_node_name)
        node.demand_timeseries_list[0].base_value = pre_incident_node_demand

    def removeDISTNodeExplicitLeak(self, damaged_node_name, wn):  # noqa: ANN001, ANN201, N802, D102
        temp = self._registry.active_nodal_damages
        value_key = {v: k for k, v in temp.items()}
        _key = value_key[damaged_node_name]
        self._registry.active_nodal_damages.pop(_key)

        temp = self._registry.getEquavalantDamageHistory(damaged_node_name)
        pipe_name = temp['new_pipe_name']
        reservoir_name = temp['new_node_name']
        wn.remove_link(pipe_name)
        wn.remove_node(reservoir_name, with_control=True)
        if reservoir_name in wn.node_name_list:
            raise  # noqa: PLE0704

        self._registry.removeEquavalantDamageHistory(damaged_node_name)

    def modifyDISTNodeDemandLinearMode(  # noqa: ANN201, N802, D102
        self, damage_node_name, real_node_name, wn, repaired_number, total_number  # noqa: ANN001
    ):
        damage_table = self._registry.getDamageData('DISTNODE', iCopy=False)
        pre_incident_demand = damage_table.loc[damage_node_name, 'Demand1']
        post_incident_demand = damage_table.loc[damage_node_name, 'Demand2']
        delta = (
            (total_number - repaired_number)
            / total_number
            * (post_incident_demand - pre_incident_demand)
        )
        new_demand = pre_incident_demand + delta
        node = wn.get_node(real_node_name)
        node.demand_timeseries_list[0].base_value = new_demand

    def modifyDISTNodeExplicitLeakEmitter(  # noqa: ANN201, N802, D102
        self, damage_node_name, real_node_name, wn, repaired_number, total_number  # noqa: ANN001
    ):
        nodal_data = self._registry._nodal_data[real_node_name]  # noqa: SLF001
        pipe_length = nodal_data['pipe_length']
        mean_pressure = nodal_data['mean_pressure']
        new_node_name = nodal_data['new_node_name']
        orginal_flow = nodal_data['orginal_flow']
        number_of_damages = total_number - repaired_number
        cd, mp0 = self._registry.damage.getEmitterCdAndElevation(
            real_node_name,
            wn,
            number_of_damages,
            pipe_length,
            mean_pressure,
            orginal_flow,
        )
        node = wn.get_node(new_node_name)

        # print(real_node_name)  # noqa: ERA001
        if cd >= node._emitter_coefficient:  # noqa: SLF001
            raise ValueError(
                'something wrong here: '
                + repr(cd)
                + ' - '
                + repr(node._emitter_coefficient)  # noqa: SLF001
                + ' '
                + str(damage_node_name)
                + ' '
                + str(real_node_name)
            )

        node._emitter_coefficient = cd  # noqa: SLF001

    def modifyDISTNodeExplicitLeakReservoir(  # noqa: ANN201, N802, D102
        self, damage_node_name, real_node_name, wn, repaired_number, total_number  # noqa: ANN001, ARG002
    ):
        nodal_data = self._registry._nodal_data[real_node_name]  # noqa: SLF001
        pipe_length = nodal_data['pipe_length']
        mean_pressure = nodal_data['mean_pressure']
        pipe_name = nodal_data['new_pipe_name']
        orginal_flow = nodal_data['orginal_flow']
        number_of_damages = total_number - repaired_number
        cd, mp0 = self._registry.damage.getEmitterCdAndElevation(
            real_node_name,
            wn,
            number_of_damages,
            pipe_length,
            mean_pressure,
            orginal_flow,
        )
        node = wn.get_node(real_node_name)  # noqa: F841

        q = orginal_flow
        nd = self._registry.damage.getNd(
            mean_pressure, number_of_damages, total_number
        )
        equavalant_pipe_diameter = (  # noqa: F841
            ((nd - 1) * q) ** 2 / (0.125 * 9.81 * 3.14**2 * mean_pressure)
        ) ** (1 / 4) * 1
        pipe = wn.get_link(pipe_name)
        # if equavalant_pipe_diameter >= pipe.diameter:
        # raise ValueError("something wrong here: "+repr(equavalant_pipe_diameter)+" - "+repr(pipe.diameter))  # noqa: ERA001, E501
        pipe.diameter = pipe.diameter / 2

    def modifyDISTNodeExplicitLeak(  # noqa: ANN201, N802, D102
        self, real_damage_node_name, virtual_node_name, wn, method, damaged_number  # noqa: ANN001, ARG002
    ):
        if method == 'equal_diameter':
            emitter_name = self._registry.virtual_node_data[virtual_node_name][
                'emitter_node'
            ]
            node = wn.get_node(emitter_name)  # noqa: F841
