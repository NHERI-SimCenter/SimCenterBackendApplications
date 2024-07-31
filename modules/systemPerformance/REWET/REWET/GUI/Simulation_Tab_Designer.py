"""Created on Thu Oct 27 19:00:30 2022

@author: snaeimi
"""

import os
import tempfile

from PyQt5 import QtGui, QtWidgets


class Simulation_Tab_Designer:
    def __init__(self):
        """These are variables that are shared between ui and settings."""
        self.setSimulationSettings(self.settings)

        """
        Reassignment of shared variables.
        """
        self.result_folder_addr = os.getcwd()
        self.temp_folder_addr = tempfile.mkdtemp()

        """
        ui value assigments.
        """
        self.setSimulationUI()

        """
        Field Validators.
        """
        self.simulation_time_line.setValidator(
            QtGui.QIntValidator(0, 10000 * 24 * 3600)
        )
        self.simulation_time_step_line.setValidator(
            QtGui.QIntValidator(0, 10000 * 24 * 3600)
        )

        """
        Signals connection.
        """
        self.result_directory_browser_button.clicked.connect(
            self.ResultFileBrowserClicked
        )
        self.temp_browser_button.clicked.connect(self.tempFileBrowserClicked)
        self.simulation_time_line.textChanged.connect(
            self.SimulationTimeValidatorHelper
        )
        self.simulation_time_step_line.textChanged.connect(
            self.SimulationTimeValidatorHelper
        )

    def getSimulationSettings(self):
        if self.result_folder_addr == '':
            self.errorMSG('REWET', 'Result folder must be provided')
            return False
        if self.temp_folder_addr == '':
            self.errorMSG('REWET', 'Temp folder must be provided')
            return False

        self.simulation_time = int(self.simulation_time_line.text())
        self.simulation_time_step = int(self.simulation_time_step_line.text())
        if self.single_radio.isChecked():
            self.number_of_damages = 'single'
        elif self.multiple_radio.isChecked():
            self.number_of_damages = 'multiple'
        else:
            raise ValueError(
                'Borh of Run-Type Buttons are not selected which is an error.'
            )
        # self.result_folder_addr -- already set
        # self.temp_folder_addr -- already set

        if self.save_time_step_yes_radio.isChecked():
            self.save_time_step = True
        elif self.save_time_step_no_radio.isChecked():
            self.save_time_step = False
        else:
            raise ValueError(
                'Both of Time-Save Buttons are not selected which is an error.'
            )

        self.settings.process['RUN_TIME'] = self.simulation_time
        self.settings.process['simulation_time_step'] = self.simulation_time_step
        self.settings.process['number_of_damages'] = self.number_of_damages
        self.settings.process['result_directory'] = self.result_folder_addr
        self.settings.process['temp_directory'] = self.temp_folder_addr
        self.settings.process['save_time_step'] = self.save_time_step

        return True

    def setSimulationUI(self):
        self.simulation_time_line.setText(str(int(self.simulation_time)))
        self.simulation_time_step_line.setText(str(int(self.simulation_time_step)))
        self.result_folder_addr_line.setText(self.result_folder_addr)
        self.temp_folder_addr_line.setText(self.temp_folder_addr)

        if self.number_of_damages == 'single':
            self.single_radio.setChecked(True)
        elif self.number_of_damages == 'multiple':
            self.multiple_radio.setChecked(True)
        else:
            raise ValueError('Unknown runtype: ' + repr(self.number_of_damages))

        if self.save_time_step == True:
            self.save_time_step_yes_radio.setChecked(True)
        elif self.save_time_step == False:
            self.save_time_step_no_radio.setChecked(True)
        else:
            raise ValueError(
                'Unknown time save value: ' + repr(self.save_time_step_no_radio)
            )

    def setSimulationSettings(self, settings):
        self.simulation_time = settings.process['RUN_TIME']
        self.simulation_time_step = settings.process['simulation_time_step']
        self.number_of_damages = settings.process['number_of_damages']
        self.result_folder_addr = settings.process['result_directory']
        self.temp_folder_addr = settings.process['temp_directory']
        self.save_time_step = settings.process['save_time_step']

    def ResultFileBrowserClicked(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self.asli_MainWindow, 'Select Directory'
        )
        if directory == '':
            return
        self.result_folder_addr = directory
        self.result_folder_addr_line.setText(self.result_folder_addr)

    def tempFileBrowserClicked(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self.asli_MainWindow, 'Select Directory'
        )
        if directory == '':
            return
        self.temp_folder_addr = directory
        self.temp_folder_addr_line.setText(self.temp_folder_addr)

    def SimulationTimeValidatorHelper(self, text):
        try:
            simulation_time = int(float(self.simulation_time_line.text()))
        except:
            simulation_time = 0
        try:
            simulation_time_step = int(float(self.simulation_time_step_line.text()))
        except:
            simulation_time_step = 0

        if text == self.simulation_time_line.text():
            sim_time_changed = True
        else:
            sim_time_changed = False
        # print(simulation_time_step)
        # print(simulation_time)
        if simulation_time_step > simulation_time:
            if sim_time_changed:
                self.simulation_time_line.setText(str(simulation_time_step))
            else:
                self.simulation_time_step_line.setText(str(simulation_time))
