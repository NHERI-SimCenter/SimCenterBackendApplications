"""Created on Tue Nov  1 23:25:30 2022

@author: snaeimi
"""

import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

from .Damage_Discovery_Window import Ui_damage_discovery


class Damage_Discovery_Designer(Ui_damage_discovery):
    def __init__(self, damage_discovery_model):
        self._window = QtWidgets.QDialog()
        self.setupUi(self._window)
        self.damage_discovery_model = damage_discovery_model.copy()

        """
        Field Validators
        """
        self.leak_amount_line.setValidator(
            QtGui.QDoubleValidator(
                0, 1000000, 20, notation=QtGui.QDoubleValidator.StandardNotation
            )
        )
        self.leak_time_line.setValidator(QtGui.QIntValidator(0, 10000 * 3600))
        self.time_line.setValidator(QtGui.QIntValidator(0, 10000 * 3600))

        if self.damage_discovery_model['method'] == 'leak_based':
            self.enableLeakBased()
            leak_amount = self.damage_discovery_model['leak_amount']
            leak_time = self.damage_discovery_model['leak_time']
            self.leak_amount_line.setText(str(leak_amount))
            self.leak_time_line.setText(str(leak_time))

        elif self.damage_discovery_model['method'] == 'time_based':
            self.enableTimeBased()
            time_discovery_ratio = self.damage_discovery_model[
                'time_discovery_ratio'
            ]
            self.populateTimeDiscoveryRatioTable(time_discovery_ratio)

        """
        Signal Connections
        """
        self.discovery_ratio_line.setValidator(
            QtGui.QDoubleValidator(
                0, 1, 20, notation=QtGui.QDoubleValidator.StandardNotation
            )
        )
        self.discovery_ratio_line.textChanged.connect(
            self.discoveryRatioValidatorHelper
        )
        self.leak_based_radio.toggled.connect(self.methodRadioButtonToggled)
        self.time_based_radio.toggled.connect(self.methodRadioButtonToggled)
        self.add_button.clicked.connect(self.addTimeDiscoveryRatioByButton)
        self.remove_button.clicked.connect(self.removeTimeDiscoveryRatioByButton)
        self.buttonBox.accepted.connect(self.okButtonPressed)

    def discoveryRatioValidatorHelper(self, x):
        discovery_ratio = float(self.discovery_ratio_line.text())

        if discovery_ratio > 1:
            self.discovery_ratio_line.setText(str(1.0))

    def enableLeakBased(self):
        self.leak_based_radio.setChecked(True)

        self.leak_anount_label.setEnabled(True)
        self.leak_amount_line.setEnabled(True)
        self.leak_time_label.setEnabled(True)
        self.leak_time_line.setEnabled(True)

        self.time_line.setEnabled(False)
        self.discovery_ratio_line.setEnabled(False)
        self.time_discovery_ratio_table.setEnabled(False)
        self.add_button.setEnabled(False)
        self.remove_button.setEnabled(False)

    def enableTimeBased(self):
        self.time_based_radio.setChecked(True)

        self.leak_anount_label.setEnabled(False)
        self.leak_amount_line.setEnabled(False)
        self.leak_time_label.setEnabled(False)
        self.leak_time_line.setEnabled(False)

        self.time_line.setEnabled(True)
        self.discovery_ratio_line.setEnabled(True)
        self.time_discovery_ratio_table.setEnabled(True)
        self.add_button.setEnabled(True)
        self.remove_button.setEnabled(True)

    def clearTimeDiscoveryRatioTable(self):
        for i in range(self.time_discovery_ratio_table.rowCount()):
            self.time_discovery_ratio_table.removeRow(0)

    def okButtonPressed(self):
        if self.leak_based_radio.isChecked():
            leak_amount = self.leak_amount_line.text()
            leak_time = self.leak_time_line.text()

            if leak_amount == '':
                self.errorMSG('Empty Vlaue', "Please fill the 'Leak Amont' field.")
                return
            elif leak_time == '':
                self.errorMSG('Empty Vlaue', "Please fill the 'Leak Time' field.")
                return

            leak_amount = float(leak_amount)
            leak_time = int(float(leak_amount))

            self.damage_discovery_model['leak_amount'] = leak_amount
            self.damage_discovery_model['leak_time'] = leak_time

        elif self.time_based_radio.isChecked():
            if 'time_discovery_ratio' not in self.damage_discovery_model:
                self.errorMSG(
                    'Discovery Ratio Error', 'Discovery Ratio Table is empty'
                )
                return

            if self.damage_discovery_model['time_discovery_ratio'].empty:
                self.errorMSG(
                    'Discovery Ratio Error', 'Discovery Ratio Table is empty'
                )
                return

            if (
                self.damage_discovery_model[
                    'time_discovery_ratio'
                ].is_monotonic_increasing
                == False
            ):
                self.errorMSG(
                    'Discovery Ratio Error',
                    'Discovery Ratio data must be monotonic through time',
                )
                return
        if self.leak_based_radio.isChecked():
            if 'time_discovery_ratio' in self.damage_discovery_model:
                self.damage_discovery_model.pop('time_discovery_ratio')

            self.damage_discovery_model['method'] = 'leak_based'
        elif self.time_based_radio.isChecked():
            if 'leak_amount' in self.damage_discovery_model:
                self.damage_discovery_model.pop('leak_amount')

            if 'leak_time' in self.damage_discovery_model:
                self.damage_discovery_model.pop('leak_time')

            self.damage_discovery_model['method'] = 'time_based'

        self._window.accept()

    def populateTimeDiscoveryRatioTable(self, time_discovery_ratio):
        for time, discovery_ratio in time_discovery_ratio.iteritems():
            number_of_rows = self.time_discovery_ratio_table.rowCount()
            self.time_discovery_ratio_table.insertRow(number_of_rows)

            time_item = QtWidgets.QTableWidgetItem(str(time))
            discovery_ratio_item = QtWidgets.QTableWidgetItem(str(discovery_ratio))

            time_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            discovery_ratio_item.setFlags(
                QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
            )

            self.time_discovery_ratio_table.setItem(number_of_rows, 0, time_item)
            self.time_discovery_ratio_table.setItem(
                number_of_rows, 1, discovery_ratio_item
            )

    def addTimeDiscoveryRatioByButton(self):
        time = self.time_line.text()
        discovery_ratio = self.discovery_ratio_line.text()

        if time == '' or discovery_ratio == '':
            return
        if 'time_discovery_ratio' not in self.damage_discovery_model:
            self.damage_discovery_model['time_discovery_ratio'] = pd.Series()
        time_discovery_ratio = self.damage_discovery_model['time_discovery_ratio']

        if int(float(time)) in time_discovery_ratio.index:
            self.errorMSG(
                'Duplicate Time',
                'There is a duplicate time. Please remove the time first',
            )
            return

        time = int(float(time))
        discovery_ratio = float(discovery_ratio)

        time_discovery_ratio.loc[time] = discovery_ratio

        self.damage_discovery_model['time_discovery_ratio'] = (
            time_discovery_ratio.sort_index()
        )
        self.clearTimeDiscoveryRatioTable()
        self.populateTimeDiscoveryRatioTable(
            self.damage_discovery_model['time_discovery_ratio']
        )

    def removeTimeDiscoveryRatioByButton(self):
        items = self.time_discovery_ratio_table.selectedItems()
        if len(items) < 1:
            return

        row_number = []
        for i in range(len(items)):
            selected_row = items[i].row()
            row_number.append(selected_row)

        row_number = list(set(row_number))

        time_discovery_ratio = self.damage_discovery_model['time_discovery_ratio']

        for selected_row in row_number:
            time = self.time_discovery_ratio_table.item(selected_row, 0).text()
            time_discovery_ratio = time_discovery_ratio.drop(time)
        self.damage_discovery_model['time_discovery_ratio'] = time_discovery_ratio
        self.clearTimeDiscoveryRatioTable()
        self.populateTimeDiscoveryRatioTable

    def methodRadioButtonToggled(self):
        if self.leak_based_radio.isChecked():
            self.enableLeakBased()
        elif self.time_based_radio.isChecked():
            self.enableTimeBased()

    def errorMSG(self, error_title, error_msg, error_more_msg=None):
        error_widget = QtWidgets.QMessageBox()
        error_widget.setIcon(QtWidgets.QMessageBox.Critical)
        error_widget.setText(error_msg)
        error_widget.setWindowTitle(error_title)
        error_widget.setStandardButtons(QtWidgets.QMessageBox.Ok)
        if error_more_msg != None:
            error_widget.setInformativeText(error_more_msg)
        error_widget.exec_()
