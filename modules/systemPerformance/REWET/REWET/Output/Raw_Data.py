# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:27:03 2022

@author: snaeimi
"""


class Raw_Data:
    def __init__():
        pass

    def saveDetailedDemandNodeData(
        self, scn_name, data_type, file_address, file_type
    ):
        if data_type not in ['pressure', 'head', 'demand', 'quality']:
            raise ValueError(
                'data type is not recognized for demand nodes: ' + repr(data_type)
            )
        data = self.getDetailedData(scn_name, data_type)
        data = data[self.demand_node_name_list]
        self.saveDataFrame(data, file_address, file_type=file_type)

    def saveDetailedJunctionData(self, scn_name, data_type, file_address, file_type):
        if data_type not in ['pressure', 'head', 'demand', 'quality']:
            raise ValueError(
                'data type is not recognized for junctiosn: ' + repr(data_type)
            )
        data = self.getDetailedData(scn_name, data_type)
        data = data[self.wn.junction_name_list]
        self.saveDataFrame(data, file_address, file_type=file_type)

    def saveDetailedTankData(self, scn_name, data_type, file_address, file_type):
        if data_type not in ['pressure', 'head', 'demand', 'quality']:
            raise ValueError(
                'data type is not recognized for tanks: ' + repr(data_type)
            )
        data = self.getDetailedData(scn_name, data_type)
        data = data[self.wn.tank_name_list]
        self.saveDataFrame(data, file_address, file_type=file_type)

    def saveDetailedReservoirData(
        self, scn_name, data_type, file_address, file_type
    ):
        if data_type not in ['pressure', 'head', 'demand', 'quality']:
            raise ValueError(
                'data type is not recognized for demand nodes: ' + repr(data_type)
            )
        data = self.getDetailedData(scn_name, data_type)
        data = data[self.wn.tank_name_list]
        self.saveDataFrame(data, file_address, file_type=file_type)

    def saveDetailedPipeData(self, scn_name, data_type, file_address, file_type):
        if data_type not in [
            'linkquality',
            'flowrate',
            'headloss',
            'velocity',
            'status',
            'setting',
            'frictionfact',
            'rxnrate',
        ]:
            raise ValueError(
                'data type is not recognized for pipes: ' + repr(data_type)
            )
        data = self.getDetailedData(scn_name, data_type)
        data = data[self.wn.pipe_name_list]
        self.saveDataFrame(data, file_address, file_type=file_type)

    def saveDetailedPumpData(self, scn_name, data_type, file_address, file_type):
        if data_type not in [
            'linkquality',
            'flowrate',
            'headloss',
            'velocity',
            'status',
            'setting',
            'frictionfact',
            'rxnrate',
        ]:
            raise ValueError(
                'data type is not recognized for pumps: ' + repr(data_type)
            )
        data = self.getDetailedData(scn_name, data_type)
        data = data[self.wn.pump_name_list]
        self.saveDataFrame(data, file_address, file_type=file_type)

    def saveDetailedValveData(self, scn_name, data_type, file_address, file_type):
        if data_type not in [
            'linkquality',
            'flowrate',
            'headloss',
            'velocity',
            'status',
            'setting',
            'frictionfact',
            'rxnrate',
        ]:
            raise ValueError(
                'data type is not recognized for valves: ' + repr(data_type)
            )
        data = self.getDetailedData(scn_name, data_type)
        data = data[self.wn.valve_name_list]
        self.saveDataFrame(data, file_address, file_type=file_type)

    def getDetailedData(self, scn_name, data_type):
        cur_scn_data = None
        if data_type in [
            'linkquality',
            'flowrate',
            'headloss',
            'velocity',
            'status',
            'setting',
            'frictionfact',
            'rxnrate',
        ]:
            cur_scn_data = self.data[scn_name].link[data_type]
        elif data_type in ['pressure', 'head', 'demand', 'quality']:
            cur_scn_data = self.data[scn_name].node[data_type]
        else:
            raise ValueError('Unknown Data Type For output')
        return cur_scn_data

    def saveDataFrame(dataframe, file_address, file_type='xlsx'):
        if file_type == 'xlsx':
            dataframe.to_excel(file_address)
        elif file_type == 'csv':
            dataframe.to_csv(file_address)
        else:
            raise ValueError('Unknown file type: ' + repr(file_type))
