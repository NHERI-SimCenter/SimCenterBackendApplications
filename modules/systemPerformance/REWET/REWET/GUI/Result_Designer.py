"""Created on Thu Nov 10 18:29:50 2022

@author: snaeimi
"""  # noqa: N999, D400, D415

import pandas as pd
from PyQt5 import QtGui, QtWidgets

single_scenario_curve_options = ['', 'Quantity', 'Delivery', 'SSI']
multi_scenario_curve_options = ['', 'Quantity Exceedance', 'Delivery Exceedance']
curve_settings = {
    'Quantity Exceedance': [
        {'Label': 'Time', 'Type': 'Time', 'Default': 'seconds'},
        {'Label': 'Population', 'Type': 'Yes-No_Combo', 'Default': 'No'},
        {
            'Label': 'Percentage',
            'Type': 'Custom_Combo',
            'Default': 'Yes',
            'Content': ['Yes', 'No'],
        },
        {'Label': 'LDN leak', 'Type': 'Yes-No_Combo', 'Default': 'No'},
        {'Label': 'leak Criteria', 'Type': 'Float Line', 'Default': '0.75'},
        {
            'Label': 'Group method',
            'Type': 'Custom_Combo',
            'Default': 'Mean',
            'Content': ['Mean', 'Min', 'Max'],
        },
        {'Label': 'Daily bin', 'Type': 'Yes-No_Combo', 'Default': 'No'},
        {'Label': 'Min time', 'Type': 'Int Line', 'Default': '0'},
        {'Label': 'Max time', 'Type': 'Int Line', 'Default': '9999999999'},
    ],
    'Delivery Exceedance': [
        {'Label': 'Time', 'Type': 'Time', 'Default': 'seconds'},
        {'Label': 'Population', 'Type': 'Yes-No_Combo', 'Default': 'No'},
        {
            'Label': 'Percentage',
            'Type': 'Custom_Combo',
            'Default': 'Yes',
            'Content': ['Yes', 'No'],
        },
        {'Label': 'LDN leak', 'Type': 'Yes-No_Combo', 'Default': 'No'},
        {'Label': 'leak Criteria', 'Type': 'Float Line', 'Default': '0.75'},
        {
            'Label': 'Group method',
            'Type': 'Custom_Combo',
            'Default': 'Mean',
            'Content': ['Mean', 'Min', 'Max'],
        },
        {'Label': 'Daily bin', 'Type': 'Yes-No_Combo', 'Default': 'No'},
        {'Label': 'Min time', 'Type': 'Int Line', 'Default': '0'},
        {'Label': 'Max time', 'Type': 'Int Line', 'Default': '9999999999'},
    ],
    'Quantity': [
        {'Label': 'Time', 'Type': 'Time', 'Default': 'seconds'},
        {'Label': 'Population', 'Type': 'Yes-No_Combo', 'Default': 'No'},
        {'Label': 'Percentage', 'Type': 'Yes-No_Combo', 'Default': 'No'},
        {'Label': 'LDN leak', 'Type': 'Yes-No_Combo', 'Default': 'No'},
        {'Label': 'leak Criteria', 'Type': 'Float Line', 'Default': '0.75'},
    ],
    'Delivery': [
        {'Label': 'Time', 'Type': 'Time', 'Default': 'seconds'},
        {'Label': 'Population', 'Type': 'Yes-No_Combo', 'Default': 'No'},
        {'Label': 'Percentage', 'Type': 'Yes-No_Combo', 'Default': 'No'},
        {'Label': 'LDN leak', 'Type': 'Yes-No_Combo', 'Default': 'No'},
        {'Label': 'leak Criteria', 'Type': 'Float Line', 'Default': '1.25'},
    ],
    'SSI': [
        {'Label': 'Time', 'Type': 'Time', 'Default': 'seconds'},
        {'Label': 'Population', 'Type': 'Yes-No_Combo', 'Default': 'No'},
    ],
}


class Time_Unit_Combo(QtWidgets.QComboBox):  # noqa: N801, D101
    def __init__(self):  # noqa: ANN204, D107
        super().__init__()
        time_units = ['second', 'hour', 'day']

        self.addItems(time_units)

    def changeCurveTimeUnit(self, raw_time_curve):  # noqa: ANN001, ANN201, N802, D102
        res = {}
        if type(raw_time_curve) == pd.core.series.Series:  # noqa: E721
            time_justified_curve = raw_time_curve.copy()
            res = self.applyUnitToSeries(time_justified_curve)
        else:
            for k in raw_time_curve:
                time_justified_curve = raw_time_curve[k].copy()
                res[k] = self.applyUnitToSeries(time_justified_curve)
        return res

    def applyUnitToSeries(self, data):  # noqa: ANN001, ANN201, N802, D102
        time_unit = self.currentText()
        if time_unit == 'second':
            pass
        elif time_unit == 'hour':
            data.index = data.index / 3600
        elif time_unit == 'day':
            data.index = data.index / 3600 / 24
        else:
            raise ValueError('Unknown unit time: ' + repr(time_unit))

        return data


class Yes_No_Combo(QtWidgets.QComboBox):  # noqa: N801, D101
    def __init__(self):  # noqa: ANN204, D107
        super().__init__()
        self.addItems(['No', 'Yes'])


class Result_Designer:  # noqa: N801, D101
    def __init__(self):  # noqa: ANN204, D107
        self.current_raw_curve = None
        self.current_curve = None
        self.curve_settings_widgets = {}
        self.main_tab.currentChanged.connect(self.tabChanged)
        self.all_scenarios_checkbox.stateChanged.connect(
            self.curveAllScenarioCheckboxChanged
        )
        self.save_curve_button.clicked.connect(self.saveCurrentCurveByButton)
        self.scenario_combo.currentTextChanged.connect(self.resultScenarioChanged)
        self.curve_type_combo.currentTextChanged.connect(self.curveTypeChanegd)

        self.initalize_result()

    def initalize_result(self):  # noqa: ANN201, D102
        self.setCurveAllScenarios(True)
        self.all_scenarios_checkbox.setChecked(True)
        self.scenario_combo.clear()
        self.scenario_combo.addItems(self.result_scenarios)
        # self.current_curve_data = None  # noqa: ERA001

    def curveAllScenarioCheckboxChanged(self, state):  # noqa: ANN001, ANN201, N802, D102
        if state == 0:
            self.setCurveAllScenarios(False)
        elif state == 2:  # noqa: PLR2004
            self.setCurveAllScenarios(True)

    def clearCurvePlot(self):  # noqa: ANN201, N802, D102
        self.mpl_curve.canvas.ax.cla()

    def plot_data(self):  # noqa: ANN201, D102
        x = range(10)
        y = range(0, 20, 2)
        self.mpl_curve.canvas.ax.plot(x, y)
        self.mpl_curve.canvas.draw()
        # self.mpl_curve.canvas.ax.set_ylabel("y_label")  # noqa: ERA001
        # self.mpl_curve.canvas.ax.set_xlabel("x_label")  # noqa: ERA001
        # self.mpl_curve.canvas.fig.tight_layout()  # noqa: ERA001

    def plotCurve(self, y_label=None, x_label=None):  # noqa: ANN001, ANN201, N802, D102
        if y_label == None:  # noqa: E711
            y_label = self.mpl_curve.canvas.ax.get_ylabel()
        if x_label == None:  # noqa: E711
            x_label = self.mpl_curve.canvas.ax.get_xlabel()

        self.mpl_curve.canvas.ax.clear()
        data = self.current_curve

        if type(data) == pd.core.series.Series:  # noqa: E721
            self.mpl_curve.canvas.ax.plot(
                self.current_curve.index, self.current_curve.to_list()
            )
        else:
            for k in data:
                self.mpl_curve.canvas.ax.plot(data[k].index, data[k].to_list())

        self.mpl_curve.canvas.ax.set_ylabel(y_label)
        self.mpl_curve.canvas.ax.set_xlabel(x_label)
        self.mpl_curve.canvas.draw()
        self.mpl_curve.canvas.fig.tight_layout()

    def setCurveAllScenarios(self, flag):  # noqa: ANN001, ANN201, N802, D102
        if flag == True:  # noqa: E712
            self.all_scenarios_checkbox.setChecked(True)
            self.scenario_combo.setEnabled(False)
            self.curve_type_combo.clear()
            self.curve_type_combo.addItems(multi_scenario_curve_options)
            self.clearCurvePlot()
        elif flag == False:  # noqa: E712
            self.all_scenarios_checkbox.setChecked(False)
            self.scenario_combo.setEnabled(True)
            self.curve_type_combo.clear()
            self.curve_type_combo.addItems(single_scenario_curve_options)
            self.clearCurvePlot()
        else:
            raise ValueError('Unknown flag: ' + repr(flag))

    def resultScenarioChanged(self, text):  # noqa: ANN001, ANN201, N802, D102
        self.result_current_scenario = text  # self.scenario_combo.getText()
        # self.current_curve_data = None  # noqa: ERA001

    def curveTypeChanegd(self, text):  # noqa: ANN001, ANN201, N802, D102
        if self.project_result == None:  # noqa: E711
            return
        self.current_curve_type = text
        self.setCurveSettingBox(text)
        self.calculateCurrentCurve()

    def calculateCurrentCurve(self):  # noqa: ANN201, C901, N802, D102, PLR0912, PLR0915
        curve_type = self.current_curve_type
        if curve_type == 'Quantity Exceedance':
            iPopulation = self.curve_settings_widgets['Population'].currentText()  # noqa: N806
            iRatio = self.curve_settings_widgets['Percentage'].currentText()  # noqa: N806
            iConsider_leak = self.curve_settings_widgets['LDN leak'].currentText()  # noqa: N806
            leak_ratio = self.curve_settings_widgets['leak Criteria'].text()
            group_method = self.curve_settings_widgets['Group method'].currentText()
            daily_bin = self.curve_settings_widgets['Daily bin'].currentText()
            min_time = self.curve_settings_widgets['Min time'].text()
            max_time = self.curve_settings_widgets['Max time'].text()

            if iConsider_leak == 'Yes':  # noqa: SIM108
                iConsider_leak = True  # noqa: N806
            else:
                iConsider_leak = False  # noqa: N806

            if iRatio == 'Yes':  # noqa: SIM108
                iRatio = True  # noqa: N806
            else:
                iRatio = False  # noqa: N806

            if daily_bin == 'Yes':  # noqa: SIM108
                daily_bin = True
            else:
                daily_bin = False

            group_method = group_method.lower()
            min_time = int(float(min_time))
            max_time = int(float(max_time))

            self.current_raw_curve = self.project_result.getQuantityExceedanceCurve(
                iPopulation=iPopulation,
                ratio=iRatio,
                consider_leak=iConsider_leak,
                leak_ratio=leak_ratio,
                result_type=group_method,
                daily=daily_bin,
                min_time=min_time,
                max_time=max_time,
            )
            self.current_curve = self.time_combo.changeCurveTimeUnit(
                self.current_raw_curve
            )
            self.plotCurve('Exceedance Probability', 'Time')

        elif curve_type == 'Delivery Exceedance':
            iPopulation = self.curve_settings_widgets['Population'].currentText()  # noqa: N806
            iRatio = self.curve_settings_widgets['Percentage'].currentText()  # noqa: N806
            iConsider_leak = self.curve_settings_widgets['LDN leak'].currentText()  # noqa: N806
            leak_ratio = self.curve_settings_widgets['leak Criteria'].text()
            group_method = self.curve_settings_widgets['Group method'].currentText()
            daily_bin = self.curve_settings_widgets['Daily bin'].currentText()
            min_time = self.curve_settings_widgets['Min time'].text()
            max_time = self.curve_settings_widgets['Max time'].text()

            if iConsider_leak == 'Yes':  # noqa: SIM108
                iConsider_leak = True  # noqa: N806
            else:
                iConsider_leak = False  # noqa: N806

            if iRatio == 'Yes':  # noqa: SIM108
                iRatio = True  # noqa: N806
            else:
                iRatio = False  # noqa: N806

            if daily_bin == 'Yes':  # noqa: SIM108
                daily_bin = True
            else:
                daily_bin = False

            group_method = group_method.lower()
            min_time = int(float(min_time))
            max_time = int(float(max_time))

            self.current_raw_curve = self.project_result.getDeliveryExceedanceCurve(
                iPopulation=iPopulation,
                ratio=iRatio,
                consider_leak=iConsider_leak,
                leak_ratio=leak_ratio,
                result_type=group_method,
                daily=daily_bin,
                min_time=min_time,
                max_time=max_time,
            )
            self.current_curve = self.time_combo.changeCurveTimeUnit(
                self.current_raw_curve
            )
            self.plotCurve('Exceedance Probability', 'Time')
        elif curve_type == 'Quantity':
            iPopulation = self.curve_settings_widgets['Population'].currentText()  # noqa: N806
            # iPopulation             = self.curve_population_settings_combo.currentText()  # noqa: ERA001, E501
            iRatio = self.curve_settings_widgets['Percentage'].currentText()  # noqa: N806
            iConsider_leak = self.curve_settings_widgets['LDN leak'].currentText()  # noqa: N806
            leak_ratio = self.curve_settings_widgets['leak Criteria'].text()

            if iConsider_leak == 'Yes':  # noqa: SIM108
                iConsider_leak = True  # noqa: N806
            else:
                iConsider_leak = False  # noqa: N806

            if iRatio == 'Yes':  # noqa: SIM108
                iRatio = True  # noqa: N806
            else:
                iRatio = False  # noqa: N806

            scn_name = self.scenario_combo.currentText()
            self.current_raw_curve = self.project_result.getQNIndexPopulation_4(
                scn_name,
                iPopulation=iPopulation,
                ratio=iRatio,
                consider_leak=iConsider_leak,
                leak_ratio=leak_ratio,
            )
            self.current_curve = self.time_combo.changeCurveTimeUnit(
                self.current_raw_curve
            )
            self.plotCurve('Quantity', 'Time')

        elif curve_type == 'Delivery':
            # self.current_curve_data = (curve_type, pd.DataFrame())  # noqa: ERA001
            iPopulation = self.curve_settings_widgets['Population'].currentText()  # noqa: N806
            # iPopulation             = self.curve_population_settings_combo.currentText()  # noqa: ERA001, E501
            iRatio = self.curve_settings_widgets['Percentage'].currentText()  # noqa: N806
            iConsider_leak = self.curve_settings_widgets['LDN leak'].currentText()  # noqa: N806
            leak_ratio = self.curve_settings_widgets['leak Criteria'].text()

            if iConsider_leak == 'Yes':  # noqa: SIM108
                iConsider_leak = True  # noqa: N806
            else:
                iConsider_leak = False  # noqa: N806

            if iRatio == 'Yes':  # noqa: SIM108
                iRatio = True  # noqa: N806
            else:
                iRatio = False  # noqa: N806

            scn_name = self.scenario_combo.currentText()
            self.current_raw_curve = self.project_result.getDLIndexPopulation_4(
                scn_name,
                iPopulation=iPopulation,
                ratio=iRatio,
                consider_leak=iConsider_leak,
                leak_ratio=leak_ratio,
            )
            self.current_curve = self.time_combo.changeCurveTimeUnit(
                self.current_raw_curve
            )
            self.plotCurve('Delivery', 'Time')

        elif curve_type == 'SSI':
            # self.current_curve_data = (curve_type, pd.DataFrame())  # noqa: ERA001
            iPopulation = self.curve_settings_widgets['Population'].currentText()  # noqa: N806
            scn_name = self.scenario_combo.currentText()
            self.current_raw_curve = (
                self.project_result.getSystemServiceabilityIndexCurve(
                    scn_name, iPopulation=iPopulation
                )
            )
            self.current_curve = self.time_combo.changeCurveTimeUnit(
                self.current_raw_curve
            )
            self.plotCurve('SSI', 'Time')

    def setCurveSettingBox(self, curve_type):  # noqa: ANN001, ANN201, N802, D102
        for i in range(self.curve_settings_table.rowCount()):  # noqa: B007
            self.curve_settings_table.removeRow(0)

        if curve_type in curve_settings:
            self.populateCurveSettingsTable(curve_settings[curve_type])
        else:
            pass
            # raise ValueError("Unknown Curve type: "+repr(curve_type))  # noqa: ERA001

    def populateCurveSettingsTable(self, settings_content):  # noqa: ANN001, ANN201, C901, N802, D102, PLR0912, PLR0915
        self.curve_settings_widgets.clear()
        vertical_header = []
        cell_type_list = []
        default_list = []
        content_list = []
        for row in settings_content:
            for k in row:
                if k == 'Label':
                    vertical_header.append(row[k])
                elif k == 'Type':
                    cell_type_list.append(row[k])
                elif k == 'Default':
                    default_list.append(row[k])

            if 'Content' in row:
                content_list.append(row['Content'])
            else:
                content_list.append(None)

        self.curve_settings_table.setColumnCount(1)
        self.curve_settings_table.setRowCount(len(settings_content))
        self.curve_settings_table.setVerticalHeaderLabels(vertical_header)

        i = 0
        for cell_type in cell_type_list:
            if cell_type == 'Time':
                self.time_combo = Time_Unit_Combo()
                self.curve_settings_table.setCellWidget(i, 0, self.time_combo)
                self.time_combo.currentTextChanged.connect(
                    self.curveTimeSettingsChanged
                )

            elif cell_type == 'Yes-No_Combo':
                current_widget = Yes_No_Combo()
                self.curve_settings_table.setCellWidget(i, 0, current_widget)
                current_widget.currentTextChanged.connect(self.curveSettingChanged)

                default_value = default_list[i]
                current_widget.setCurrentText(default_value)

                self.curve_settings_widgets[vertical_header[i]] = current_widget

            elif cell_type == 'Custom_Combo':
                current_widget = QtWidgets.QComboBox()
                contents = content_list[i]
                current_widget.addItems(contents)
                self.curve_settings_table.setCellWidget(i, 0, current_widget)
                current_widget.currentTextChanged.connect(self.curveSettingChanged)

                default_value = default_list[i]
                current_widget.setCurrentText(default_value)

                self.curve_settings_widgets[vertical_header[i]] = current_widget

            elif cell_type == 'Float Line':
                current_widget = QtWidgets.QLineEdit()
                self.curve_settings_table.setCellWidget(i, 0, current_widget)
                current_widget.editingFinished.connect(self.curveSettingChanged)
                current_widget.setValidator(
                    QtGui.QDoubleValidator(
                        0,
                        1000000,
                        20,
                        notation=QtGui.QDoubleValidator.StandardNotation,
                    )
                )

                default_value = default_list[i]
                current_widget.setText(default_value)
                self.curve_settings_widgets[vertical_header[i]] = current_widget

            elif cell_type == 'Int Line':
                current_widget = QtWidgets.QLineEdit()
                self.curve_settings_table.setCellWidget(i, 0, current_widget)
                current_widget.editingFinished.connect(self.curveSettingChanged)
                current_widget.setValidator(QtGui.QIntValidator(0, 3600 * 24 * 1000))

                default_value = default_list[i]
                current_widget.setText(default_value)
                self.curve_settings_widgets[vertical_header[i]] = current_widget
            else:
                raise ValueError(repr(cell_type))

            i += 1  # noqa: SIM113
        # for label in settings_content:

    def curveTimeSettingsChanged(self, x):  # noqa: ANN001, ANN201, ARG002, N802, D102
        self.current_curve = self.time_combo.changeCurveTimeUnit(
            self.current_raw_curve
        )
        self.plotCurve()

    def curveSettingChanged(self):  # noqa: ANN201, N802, D102
        if 'Population' in self.curve_settings_widgets:
            new_population_setting = self.curve_settings_widgets[
                'Population'
            ].currentText()
            if new_population_setting == 'Yes' and type(  # noqa: E721
                self.project_result._population_data  # noqa: SLF001
            ) == type(None):
                self.errorMSG('Error', 'Population data is not loaded')
                self.curve_settings_widgets['Population'].setCurrentText('No')
                return
        self.calculateCurrentCurve()

    def tabChanged(self, index):  # noqa: ANN001, ANN201, N802, D102
        if index == 1:
            self.initalize_result()

    def saveCurrentCurveByButton(self):  # noqa: ANN201, N802, D102
        # if self.current_curve_data == None:
        if type(self.current_curve) == type(None):  # noqa: E721
            self.errorMSG('REWET', 'No curve is ploted')
            return

        file_addr = QtWidgets.QFileDialog.getSaveFileName(
            self.asli_MainWindow,
            'Save File',
            self.project_file_addr,
            'Excel Workbook (*.xlsx)',
        )
        if file_addr[0] == '':
            return

        # self.current_curve_data[1].to_excel(file_addr[0])  # noqa: ERA001
        self.current_curve.to_excel(file_addr[0])
