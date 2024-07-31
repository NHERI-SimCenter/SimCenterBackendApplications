"""Created on Fri Oct 28 12:50:24 2022

@author: snaeimi
"""  # noqa: N999, D400, D415

import os
import pickle

import pandas as pd
from PyQt5 import QtCore, QtWidgets

from .Node_Damage_Model_Designer import Node_Damage_Model_Designer
from .Pipe_Damage_Model_Designer import Pipe_Damage_Model_Designer
from .Scenario_Dialog_Designer import Scenario_Dialog_Designer


class Damage_Tab_Designer:  # noqa: N801, D101
    def __init__(self):  # noqa: ANN204
        # self.pipe_damage_model = {"CI":{"alpha":-0.0038, "beta":0.1096, "gamma":0.0196, "a":2, "b":1 }, "DI":{"alpha":-0.0038, "beta":0.05, "gamma":0.04, "a":2, "b":1 } }
        # self.node_damage_model = {'a':0.0036, 'aa':1, 'b':0, 'bb':0, 'c':-0.877, 'cc':1, 'd':0, 'dd':0, 'e':0.0248, 'ee1':1, 'ee2':1, 'f':0, 'ff1':0, 'ff2':0, "damage_node_model": "equal_diameter_emitter"}
        """These are variables that are shared between ui and settings."""  # noqa: D401
        self.setDamageSettings(self.settings, self.scenario_list)

        """
        Reassignment of shared variables.
        """
        self.damage_input_directory = os.getcwd()  # noqa: PTH109
        self.current_xlsx_directory = os.getcwd()  # noqa: PTH109
        if self.scenario_list == None:  # noqa: E711
            self.scenario_list = pd.DataFrame(
                columns=[
                    'Scenario Name',
                    'Pipe Damage',
                    'Nodal Damage',
                    'Pump Damage',
                    'Tank Damage',
                    'Probability',
                ]
            )
        self.scneraio_validated = False
        self.damage_pipe_model_reviewed = False

        """
        ui value assigments.
        """
        self.setDamageUI()

        """
        Signals connection.
        """
        self.add_scenario_button.clicked.connect(self.addNewScenarioByButton)
        self.damage_directory_browse_button.clicked.connect(
            self.browseDamageDirectoryByButton
        )
        self.remove_scenario_button.clicked.connect(self.removeScenarioByButton)
        self.load_scenario_button.clicked.connect(self.loadScenarioByButton)
        self.save_scenario_button.clicked.connect(self.saveScenarioByButton)
        self.validate_scenario_button.clicked.connect(self.validateScenarioByButton)
        self.pipe_damage_modeling_button.clicked.connect(
            self.pipeDamageSettingByButton
        )
        self.node_damage_modeling_button.clicked.connect(
            self.nodeDamageSettingByButton
        )
        self.file_type_excel_radio.toggled.connect(self.fileTypeChanged)
        self.file_type_pickle_radio.toggled.connect(self.fileTypeChanged)

    def getDamageSettings(self):  # noqa: ANN201, N802, D102
        if len(self.scenario_list) < 1:
            self.errorMSG('REWET', 'Damage scenario list is empty.')
            return False
        if self.damage_input_directory == '':
            self.errorMSG('REWET', 'No Damage Input Directory is selected.')
            return False

        self.settings.scenario['pipe_damage_model'] = self.pipe_damage_model
        self.settings.scenario['node_damage_model'] = self.node_damage_model
        self.settings.scenario['Pipe_damage_input_method'] = (
            self.pipe_damage_input_method
        )
        self.settings.process['pipe_damage_file_directory'] = (
            self.damage_input_directory
        )
        # self.scenario_list -- already set
        return True

    def setDamageUI(self):  # noqa: ANN201, N802, D102
        self.damage_direcotry_line.setText(self.damage_input_directory)
        self.clearScnearioTable()
        self.populateScenarioTable()

    def setDamageSettings(self, settings, scenario_list):  # noqa: ANN001, ANN201, N802, D102
        self.pipe_damage_model = settings.scenario['pipe_damage_model']
        self.node_damage_model = settings.scenario['node_damage_model']
        self.pipe_damage_input_method = settings.scenario['Pipe_damage_input_method']
        self.damage_input_directory = settings.process['pipe_damage_file_directory']
        self.scenario_list = scenario_list

    def addNewScenarioByButton(self):  # noqa: ANN201, N802, D102
        new_scenario_dialoge = Scenario_Dialog_Designer()

        error = True
        while error:
            error = False
            return_value = new_scenario_dialoge._window.exec_()  # noqa: SLF001

            if return_value == 0:
                return

            scenario_name = new_scenario_dialoge.scenario_name_line.text()
            pipe_damage_name = new_scenario_dialoge.pipe_damage_line.text()
            node_damage_name = new_scenario_dialoge.node_damage_line.text()
            pump_damage_name = new_scenario_dialoge.pump_damage_line.text()
            tank_damage_name = new_scenario_dialoge.tank_damage_line.text()
            probability = new_scenario_dialoge.probability_line.text()
            probability = float(probability)

            if len(scenario_name) < 1:
                self.errorMSG('Empty Scneario Name', 'Please enter a scenario name')
                error = True

            if len(pipe_damage_name) < 1:
                self.errorMSG(
                    'Empty Pipe Damage Name', 'Please enter a pipe damage name'
                )
                error = True

            if len(node_damage_name) < 1:
                self.errorMSG(
                    'Empty Node Damage Name', 'Please enter a node damage name'
                )
                error = True

            if len(pump_damage_name) < 1:
                self.errorMSG(
                    'Empty Pump Damage Name', 'Please enter a pump damage name'
                )
                error = True

            if len(tank_damage_name) < 1:
                self.errorMSG(
                    'Empty Tank Damage Name', 'Please enter a tank damage name'
                )
                error = True

            if (self.scenario_list['Scenario Name'] == scenario_name).any():
                self.errorMSG(
                    'Duplicate Scneario Name',
                    'Please Have the Scenario Name changed',
                )
                error = True
        new_row = {
            'Scenario Name': scenario_name,
            'Pipe Damage': pipe_damage_name,
            'Nodal Damage': node_damage_name,
            'Pump Damage': pump_damage_name,
            'Tank Damage': tank_damage_name,
            'Probability': probability,
        }
        self.scenario_list = self.scenario_list.append(new_row, ignore_index=True)
        self.clearScnearioTable()
        self.populateScenarioTable()
        self.scneraio_validated = False
        self.damage_pipe_model_reviewed = False

    def fileTypeChanged(self, checked):  # noqa: ANN001, ANN201, ARG002, N802, D102
        if self.file_type_excel_radio.isChecked():
            self.pipe_damage_input_method = 'excel'
        else:
            self.pipe_damage_input_method = 'pickle'

    def removeScenarioByButton(self):  # noqa: ANN201, N802, D102
        items = self.scenario_table.selectedItems()
        if len(items) < 1:
            return

        row_number = []
        for i in range(len(items)):
            selected_row = items[i].row()
            row_number.append(selected_row)

        row_number = list(set(row_number))

        for selected_row in row_number:
            scenario_name = self.scenario_table.item(selected_row, 0).text()
            to_be_removed_index = (
                self.scenario_list[self.scenario_list == scenario_name]
            ).index[0]
            self.scenario_list = self.scenario_list.drop(to_be_removed_index)
        self.scenario_list = self.scenario_list.reset_index(drop=True)
        self.clearScnearioTable()
        self.populateScenarioTable()
        self.scneraio_validated = False
        self.damage_pipe_model_reviewed = False

    def loadScenarioByButton(self):  # noqa: ANN201, N802, D102
        file = QtWidgets.QFileDialog.getOpenFileName(
            self.asli_MainWindow,
            'Open file',
            self.current_xlsx_directory,
            'scenrario file (*.xlsx)',
        )
        if file[0] == '':
            return
        split_addr = os.path.split(file[0])

        temp = self.getScnearioListFromXLSX(file[0])
        if temp is None:
            return
        self.scenario_list = temp

        self.current_xlsx_directory = split_addr[0]
        self.wdn_addr_line.setText(file[0])
        self.clearScnearioTable()
        self.populateScenarioTable()
        self.scneraio_validated = False
        self.damage_pipe_model_reviewed = False

    def saveScenarioByButton(self):  # noqa: ANN201, N802, D102
        file = QtWidgets.QFileDialog.getSaveFileName(
            self.asli_MainWindow,
            'Save file',
            self.current_xlsx_directory,
            'Excel file (*.xlsx)',
        )
        split_addr = os.path.split(file[0])
        self.current_xlsx_directory = split_addr[0]

        self.scenario_list.to_excel(file[0])

    def validateScenarioByButton(self):  # noqa: ANN201, C901, N802, D102, PLR0912, PLR0915
        self.status_text.setText('Validating Damage Scnearios')
        if_validate_successful = True
        text_output = ''
        scneario_list = self.scenario_list

        all_pipe_material = set()

        damage_pipe_not_exist_List = []  # noqa: N806
        damage_nodal_not_exist_List = []  # noqa: N806
        damage_pump_not_exist_List = []  # noqa: N806
        damage_tank_not_exist_List = []  # noqa: N806

        for index, row in scneario_list.iterrows():  # noqa: B007
            damage_pipe_name = row['Pipe Damage']
            damage_pipe_addr = os.path.join(  # noqa: PTH118
                self.damage_input_directory, damage_pipe_name
            )
            if not os.path.exists(damage_pipe_addr):  # noqa: PTH110
                damage_pipe_not_exist_List.append(damage_pipe_name)

            damage_node_name = row['Nodal Damage']
            damage_nodal_addr = os.path.join(  # noqa: PTH118
                self.damage_input_directory, damage_node_name
            )
            if not os.path.exists(damage_nodal_addr):  # noqa: PTH110
                damage_nodal_not_exist_List.append(damage_node_name)

            damage_pump_name = row['Pump Damage']
            damage_pump_addr = os.path.join(  # noqa: PTH118
                self.damage_input_directory, damage_pump_name
            )
            if not os.path.exists(damage_pump_addr):  # noqa: PTH110
                damage_pump_not_exist_List.append(damage_pump_name)

            damage_tank_name = row['Tank Damage']
            damage_tank_addr = os.path.join(  # noqa: PTH118
                self.damage_input_directory, damage_tank_name
            )
            if not os.path.exists(damage_tank_addr):  # noqa: PTH110
                damage_tank_not_exist_List.append(damage_tank_name)

        if len(damage_pipe_not_exist_List) > 0:
            text_output += (
                'The follwing pipe damage files could not be found.\n'
                + repr(damage_pipe_not_exist_List)
                + '\n'
            )
            if_validate_successful = False
        if len(damage_nodal_not_exist_List) > 0:
            text_output += (
                'The follwing node damage files could not be found.\n'
                + repr(damage_nodal_not_exist_List)
                + '\n'
            )
            if_validate_successful = False
        if len(damage_pump_not_exist_List) > 0:
            text_output += (
                'The follwing pump damage files could not be found.\n'
                + repr(damage_pump_not_exist_List)
                + '\n'
            )
            if_validate_successful = False
        if len(damage_tank_not_exist_List) > 0:
            text_output += (
                'The follwing tank damage files could not be found.\n'
                + repr(damage_tank_not_exist_List)
                + '\n'
            )
            if_validate_successful = False

        try:
            must_have_pipe_columns = set(  # noqa: C405
                ['time', 'pipe_id', 'damage_loc', 'type', 'Material']
            )
            for index, row in scneario_list.iterrows():  # noqa: B007
                damage_pipe_name = row['Pipe Damage']
                if self.pipe_damage_input_method == 'excel':
                    pipe_damage = pd.read_excel(
                        os.path.join(self.damage_input_directory, damage_pipe_name)  # noqa: PTH118
                    )
                elif self.pipe_damage_input_method == 'pickle':
                    with open(  # noqa: PTH123
                        os.path.join(self.damage_input_directory, damage_pipe_name),  # noqa: PTH118
                        'rb',
                    ) as f:
                        pipe_damage = pickle.load(f)  # noqa: S301
                        index_list = pipe_damage.index
                        pipe_damage = pd.DataFrame.from_dict(pipe_damage.to_list())
                        pipe_damage.loc[:, 'time'] = index_list
                        if len(index_list) == 0:
                            available_columns = set(pipe_damage.columns)
                            not_available_columns = (
                                must_have_pipe_columns - available_columns
                            )
                            pipe_damage.loc[:, not_available_columns] = None
                        # print(pipe_damage)
                        # pipe_damage = pd.DataFrame.from_dict( )
                        # pipe_damage.index.name = 'time'
                        # pipe_damage = pipe_damage.reset_index(drop=False)
                    # print(pipe_damage)
                available_columns = set(pipe_damage.columns)
                not_available_columns = must_have_pipe_columns - available_columns
                if len(not_available_columns) > 0:
                    text_output += (
                        'In pipe damage file= '
                        + repr(damage_pipe_name)
                        + 'the following headers are missing: '
                        + repr(not_available_columns)
                        + '\n'
                    )
                    if_validate_successful = False

                new_material_set = set(pipe_damage['Material'].unique())
                all_pipe_material = all_pipe_material.union(new_material_set)

            must_have_node_columns = set(  # noqa: C405
                ['time', 'node_name', 'Number_of_damages', 'node_Pipe_Length']
            )
            for index, row in scneario_list.iterrows():  # noqa: B007
                damage_node_name = row['Nodal Damage']
                if self.pipe_damage_input_method == 'excel':
                    node_damage = pd.read_excel(
                        os.path.join(self.damage_input_directory, damage_node_name)  # noqa: PTH118
                    )
                elif self.pipe_damage_input_method == 'pickle':
                    with open(  # noqa: PTH123
                        os.path.join(self.damage_input_directory, damage_node_name),  # noqa: PTH118
                        'rb',
                    ) as f:
                        node_damage = pickle.load(f)  # noqa: S301
                        index_list = node_damage.index
                        node_damage = pd.DataFrame.from_dict(node_damage.to_list())
                        node_damage.loc[:, 'time'] = index_list
                        if len(index_list) == 0:
                            available_columns = set(node_damage.columns)
                            not_available_columns = (
                                must_have_node_columns - available_columns
                            )
                            pipe_damage.loc[:, not_available_columns] = None
                available_columns = set(node_damage.columns)
                not_available_columns = must_have_node_columns - available_columns
                if len(not_available_columns) > 0:
                    text_output += (
                        'In node damage file= '
                        + repr(damage_node_name)
                        + 'the following headers are missing: '
                        + repr(not_available_columns)
                        + '\n'
                    )
                    if_validate_successful = False

            must_have_pump_columns = set(['time', 'Pump_ID', 'Restore_time'])  # noqa: C405
            for index, row in scneario_list.iterrows():  # noqa: B007
                damage_pump_name = row['Pump Damage']
                if self.pipe_damage_input_method == 'excel':
                    pump_damage = pd.read_excel(
                        os.path.join(self.damage_input_directory, damage_pump_name)  # noqa: PTH118
                    )
                elif self.pipe_damage_input_method == 'pickle':
                    with open(  # noqa: PTH123
                        os.path.join(self.damage_input_directory, damage_pump_name),  # noqa: PTH118
                        'rb',
                    ) as f:
                        pump_damage = pickle.load(f)  # noqa: S301
                        pump_damage = pump_damage.reset_index(drop=False)
                        available_columns = set(pump_damage.columns)
                        not_available_columns = (
                            must_have_pump_columns - available_columns
                        )
                        pump_damage.loc[:, not_available_columns] = None
                        # index_list  = pump_damage.index
                        # pump_damage = pd.DataFrame.from_dict(pump_damage.to_list() )

                available_columns = set(pump_damage.columns)
                not_available_columns = must_have_pump_columns - available_columns

                if len(not_available_columns) > 0 and len(pump_damage) > 0:
                    text_output += (
                        'In pump damage file= '
                        + repr(damage_pump_name)
                        + 'the following headers are missing: '
                        + repr(not_available_columns)
                        + '\n'
                    )
                    if_validate_successful = False

            must_have_tank_columns = set(['time', 'Tank_ID', 'Restore_time'])  # noqa: C405
            for index, row in scneario_list.iterrows():  # noqa: B007
                damage_tank_name = row['Tank Damage']
                if self.pipe_damage_input_method == 'excel':
                    tank_damage = pd.read_excel(
                        os.path.join(self.damage_input_directory, damage_tank_name)  # noqa: PTH118
                    )
                elif self.pipe_damage_input_method == 'pickle':
                    with open(  # noqa: PTH123
                        os.path.join(self.damage_input_directory, damage_tank_name),  # noqa: PTH118
                        'rb',
                    ) as f:
                        tank_damage = pickle.load(f)  # noqa: S301
                        tank_damage = tank_damage.reset_index(drop=False)
                        available_columns = set(tank_damage.columns)
                        not_available_columns = (
                            must_have_tank_columns - available_columns
                        )
                        tank_damage.loc[:, not_available_columns] = None

                available_columns = set(tank_damage.columns)
                not_available_columns = must_have_tank_columns - available_columns
                if len(not_available_columns) > 0 and len(damage_tank_name) > 0:
                    text_output += (
                        'In tank damage file= '
                        + repr(damage_tank_name)
                        + 'the following headers are missing: '
                        + repr(not_available_columns)
                        + '\n'
                    )
                    if_validate_successful = False
        except Exception as exp:  # noqa: TRY302
            raise exp  # noqa: TRY201
            if_validate_successful = False
            text_output += (
                'An error happened. File type might be wrong in addition to other problems. More information:\n'
                + repr(exp)
            )

        if if_validate_successful == True:  # noqa: E712
            text_output += 'Damage Scenario List Validated Sucessfully'
            not_defined_materials = all_pipe_material - set(
                self.pipe_damage_model.keys()
            )
            if len(not_defined_materials) > 0:
                default_material_model = self.settings.scenario[
                    'default_pipe_damage_model'
                ]
                new_material_model = dict(
                    zip(
                        not_defined_materials,
                        [
                            default_material_model
                            for i in range(len(not_defined_materials))
                        ],
                    )
                )
                self.pipe_damage_model.update(new_material_model)
            self.scneraio_validated = True

        self.status_text.setText(text_output)

    def pipeDamageSettingByButton(self):  # noqa: ANN201, N802, D102
        if self.scneraio_validated == False:  # noqa: E712
            self.errorMSG(
                'REWET',
                'You must validate damage scenarios sucessfully before reviewing pipe damage models.',
            )
            return
        pipe_designer = Pipe_Damage_Model_Designer(self.pipe_damage_model)
        return_value = pipe_designer._window.exec_()  # noqa: SLF001

        if return_value == 1:
            self.pipe_damage_model = pipe_designer.pipe_damage_model

        self.damage_pipe_model_reviewed = True

    def nodeDamageSettingByButton(self):  # noqa: ANN201, N802, D102
        node_designer = Node_Damage_Model_Designer(self.node_damage_model)
        return_value = node_designer._window.exec_()  # noqa: SLF001

        if return_value == 1:
            self.node_damage_model = node_designer.node_damage_model

    def browseDamageDirectoryByButton(self):  # noqa: ANN201, N802, D102
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self.asli_MainWindow, 'Select Directory', self.current_xlsx_directory
        )
        if directory == '':
            return
        self.current_xlsx_directory = self.current_xlsx_directory
        self.damage_input_directory = directory
        self.damage_direcotry_line.setText(directory)

    def getScnearioListFromXLSX(self, scenario_file_addr):  # noqa: ANN001, ANN201, N802, D102
        scn_list = pd.read_excel(scenario_file_addr)

        must_be_headers = [
            'Scenario Name',
            'Pipe Damage',
            'Nodal Damage',
            'Pump Damage',
            'Tank Damage',
            'Probability',
        ]
        available_headers = scn_list.columns.tolist()

        not_available_headers = set(must_be_headers) - set(available_headers)
        if len(not_available_headers) > 1:
            self.status_text.setText(
                'failed to open the scenario file. the folowing columns are missing and need to be in teh file: '
                + repr(not_available_headers)
            )
            return None
        else:  # noqa: RET505
            self.status_text.setText('Opened file Sucessfully.')
        scn_list = scn_list[must_be_headers]

        return scn_list  # noqa: RET504

    def populateScenarioTable(self):  # noqa: ANN201, N802, D102
        for index, row in self.scenario_list.iterrows():  # noqa: B007
            number_of_rows = self.scenario_table.rowCount()
            self.scenario_table.insertRow(number_of_rows)

            scenario_item = QtWidgets.QTableWidgetItem(row['Scenario Name'])
            pipe_damage_item = QtWidgets.QTableWidgetItem(row['Pipe Damage'])
            node_damage_item = QtWidgets.QTableWidgetItem(row['Nodal Damage'])
            pump_damage_item = QtWidgets.QTableWidgetItem(row['Pump Damage'])
            tank_damage_item = QtWidgets.QTableWidgetItem(row['Tank Damage'])
            probability_item = QtWidgets.QTableWidgetItem(str(row['Probability']))

            scenario_item.setFlags(
                QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
            )
            pipe_damage_item.setFlags(
                QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
            )
            node_damage_item.setFlags(
                QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
            )
            pump_damage_item.setFlags(
                QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
            )
            tank_damage_item.setFlags(
                QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
            )
            probability_item.setFlags(
                QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
            )

            self.scenario_table.setItem(number_of_rows, 0, scenario_item)
            self.scenario_table.setItem(number_of_rows, 1, pipe_damage_item)
            self.scenario_table.setItem(number_of_rows, 2, node_damage_item)
            self.scenario_table.setItem(number_of_rows, 3, pump_damage_item)
            self.scenario_table.setItem(number_of_rows, 4, tank_damage_item)
            self.scenario_table.setItem(number_of_rows, 5, probability_item)

    def clearScnearioTable(self):  # noqa: ANN201, N802, D102
        for i in range(self.scenario_table.rowCount()):  # noqa: B007
            self.scenario_table.removeRow(0)
