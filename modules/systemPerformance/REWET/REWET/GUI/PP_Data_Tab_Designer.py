"""Created on Thu Dec 29 15:41:03 2022

@author: snaeimi
"""  # noqa: N999, D400

import os

import pandas as pd
from PyQt5 import QtWidgets
from Result_Project import Project_Result


class PP_Data_Tab:  # noqa: D101
    def __init__(self, project):
        self.pp_project = project
        # self.__settings            = settings
        self.result_scenarios = []
        self.population_data = None
        # self.pp_result_folder_addr = result_folder_addr
        self.load_results_button.clicked.connect(self.resultLoadButtonPressed)
        self.population_browser_button.clicked.connect(self.browsePopulationData)
        self.population_load_button.clicked.connect(self.loadPopulationData)
        # self.results_tabs_widget.setTabEnabled(1, False)
        self.project_result = None
        self.current_population_directory = ''

    def initalizeResultData(self):  # noqa: N802, D102
        if self.project == None:  # noqa: E711
            self.errorMSG(
                'Error', 'No project is found. open or save a new project.'
            )
            return
        result_directory = self.result_folder_addr_line.text()
        self.project_result = Project_Result(
            self.project,
            iObject=True,
            result_file_dir=result_directory,
            ignore_not_found=True,
        )
        self.clearResultData()

        print(self.project_result.scn_name_list_that_result_file_not_found)  # noqa: T201
        for index, row in self.scenario_list.iterrows():  # noqa: B007
            number_of_rows = self.result_file_status_table.rowCount()
            scenario_name = row['Scenario Name']
            scenario_item = QtWidgets.QTableWidgetItem(scenario_name)
            if (
                scenario_name
                in self.project_result.scn_name_list_that_result_file_not_found
            ):
                status_item = QtWidgets.QTableWidgetItem('NOT Available')
            else:
                self.result_scenarios.append(scenario_name)
                status_item = QtWidgets.QTableWidgetItem('Available')

            self.result_file_status_table.insertRow(number_of_rows)
            self.result_file_status_table.setItem(number_of_rows, 0, scenario_item)
            self.result_file_status_table.setItem(number_of_rows, 1, status_item)

        if self.read_files_check_box.isChecked():
            for scenario_name in self.result_scenarios:
                try:
                    self.project_result.loadScneariodata(scenario_name)
                except Exception:  # noqa: BLE001, PERF203
                    self.errorMSG('Error', 'Error occurred in reading data')
                    self.clearResultData()
                    raise Exception  # noqa: B904, TRY002
                    return

        self.results_tabs_widget.setTabEnabled(1, True)  # noqa: FBT003

    def clearResultData(self):  # noqa: N802, D102
        for i in range(self.result_file_status_table.rowCount()):  # noqa: B007
            self.result_file_status_table.removeRow(0)

    def resultLoadButtonPressed(self):  # noqa: N802, D102
        # data_retrived = False
        # if self.getSimulationSettings():
        # if self.getHydraulicSettings():
        # if self.getDamageSettings():
        # if self.getRestorationSettings():
        # data_retrived = True

        # if not data_retrived:
        # return

        self.initalizeResultData()

    def browsePopulationData(self):  # noqa: N802, D102
        file = QtWidgets.QFileDialog.getOpenFileName(
            self.asli_MainWindow,
            'Open file',
            self.current_population_directory,
            'Excel file (*.xlsx);;CSV File (*.csv)',
        )
        if file[0] == '':
            return
        split_addr = os.path.split(file[0])
        self.current_population_directory = split_addr[0]

        self.population_addr_line.setText(file[0])

        print(file)  # noqa: T201
        if file[1] == 'Excel file (*.xlsx)':
            self.population_data = pd.read_excel(file[0])
        elif file[1] == 'CSV File (*.csv)':
            self.population_data = pd.read_scv(file[0])
        else:
            raise ValueError('Unknown population file type: ' + repr(file[1]))

        self.population_node_ID_header_combo.clear()
        self.population_node_ID_header_combo.addItems(
            self.population_data.columns.to_list()
        )
        self.population_population_header_combo.clear()
        self.population_population_header_combo.addItems(
            self.population_data.columns.to_list()
        )

        if len(self.population_data.columns.to_list()) >= 2:  # noqa: PLR2004
            self.population_population_header_combo.setCurrentIndex(1)

    def loadPopulationData(self):  # noqa: N802, D102
        node_id_header = self.population_node_ID_header_combo.currentText()
        population_header = self.population_population_header_combo.currentText()

        if node_id_header == population_header:
            self.errorMSG(
                'Error', 'Node ID Header and Population Header cannot be the same'
            )
            return

        if node_id_header == '' or population_header == '':
            self.errorMSG(
                'Error',
                'Node ID Header or/and Population Header is not selected. Maybe an empty population file?',
            )
            return

        if self.project_result == None:  # noqa: E711
            self.errorMSG(
                'Error', 'No project and data is loaded. Please load the data first.'
            )
            return

        self.project_result.loadPopulation(
            self.population_data, node_id_header, population_header
        )

    def errorMSG(self, error_title, error_msg, error_more_msg=None):  # noqa: N802, D102
        error_widget = QtWidgets.QMessageBox()
        error_widget.setIcon(QtWidgets.QMessageBox.Critical)
        error_widget.setText(error_msg)
        error_widget.setWindowTitle(error_title)
        error_widget.setStandardButtons(QtWidgets.QMessageBox.Ok)
        if error_more_msg != None:  # noqa: E711
            error_widget.setInformativeText(error_more_msg)
        error_widget.exec_()
