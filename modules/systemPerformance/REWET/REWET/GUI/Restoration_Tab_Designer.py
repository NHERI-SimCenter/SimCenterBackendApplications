"""Created on Wed Nov  2 00:24:43 2022

@author: snaeimi
"""

import os

from PyQt5 import QtGui, QtWidgets

from .Node_Damage_Discovery_Designer import Node_Damage_Discovery_Designer
from .Pipe_Damage_Discovery_Designer import Pipe_Damage_Discovery_Designer
from .Pump_Damage_Discovery_Designer import Pump_Damage_Discovery_Designer
from .Tank_Damage_Discovery_Designer import Tank_Damage_Discovery_Designer


class Restoration_Tab_Designer:
    def __init__(self):
        """These are variables that are shared between ui and settings."""
        self.setRestorationSettings(self.settings)

        """
        Reassignment of shared variables.
        """
        self.current_policy_directory = os.getcwd()

        """
        ui value assignments.
        """
        self.setRestorationUI()

        """
        Field Validators.
        """
        self.minimum_job_time_line.setValidator(QtGui.QIntValidator(0.0, 2147483647))

        """
        Signals connection.
        """
        self.policy_browse_button.clicked.connect(self.browsePolicyDefinitionFile)
        self.pipe_damage_discovery_button.clicked.connect(
            self.pipeDamageDiscoveryByButton
        )
        self.node_damage_discovery_button.clicked.connect(
            self.nodeDamageDiscoveryByButton
        )
        self.pump_damage_discovery_button.clicked.connect(
            self.pumpDamageDiscoveryByButton
        )
        self.tank_damage_discovery_button.clicked.connect(
            self.tankDamageDiscoveryByButton
        )

    def getRestorationSettings(self):
        if self.restoration_on_radio.isChecked():
            self.restoration_on = True
        elif self.restoration_off_radio.isChecked():
            self.restoration_on = False
        else:
            raise ValueError(
                'None of Restoration-on/off buttons are checked which is an error.'
            )

        if self.script_txt_radio.isChecked():
            self.restoraion_policy_type = 'script'
        elif self.script_rrp_radio.isChecked():
            self.restoraion_policy_type = 'binary'
        else:
            raise ValueError(
                'None of File-Type buttons are checked which is an error.'
            )

        self.minimum_job_time = int(float(self.minimum_job_time_line.text()))

        if self.restoraion_policy_addr == '':
            self.errorMSG('REWET', 'Policy Definition file is needed.')
            return False

        self.settings.process['Restoration_on'] = self.restoration_on
        self.settings.scenario['Restoraion_policy_type'] = (
            self.restoraion_policy_type
        )
        self.settings.scenario['Restortion_config_file'] = (
            self.restoraion_policy_addr
        )
        self.settings.process['minimum_job_time'] = self.minimum_job_time
        self.settings.scenario['pipe_damage_discovery_model'] = (
            self.pipe_damage_discovery_model
        )
        self.settings.scenario['node_damage_discovery_model'] = (
            self.node_damage_discovery_model
        )
        self.settings.scenario['pump_damage_discovery_model'] = (
            self.pump_damage_discovery_model
        )
        self.settings.scenario['tank_damage_discovery_model'] = (
            self.tank_damage_discovery_model
        )
        self.settings.scenario['crew_out_of_zone_travel'] = self.out_of_zone_allowed

        return True

    def setRestorationUI(self):
        if self.restoration_on == True:
            self.restoration_on_radio.setChecked(True)
        elif self.restoration_on == False:
            self.restoration_off_radio.setChecked(True)
        else:
            raise ValueError(
                'Unknown restoration-on status: ' + repr(self.restoration_on)
            )

        self.script_rrp_radio.setEnabled(False)
        self.policy_designer.setEnabled(False)
        self.policy_definition_addr_line.setText(self.restoraion_policy_addr)

        if self.restoraion_policy_type == 'script':
            self.script_txt_radio.setChecked(True)
        elif self.restoraion_policy_type == 'binary':
            self.script_rrp_radio.setChecked(True)
        else:
            raise ValueError(
                'Uknown policy type: ' + repr(self.restoraion_policy_type)
            )

        self.minimum_job_time_line.setText(str(self.minimum_job_time))

        if self.out_of_zone_allowed == True:
            self.out_of_zone_travel_yes.setChecked(True)
        elif self.out_of_zone_allowed == False:
            self.out_of_zone_travel_no.setChecked(True)
        else:
            raise ValueError(
                'Unknown out-of-zone travel value: '
                + repr(self.out_of_zone_travel_no)
            )

    def setRestorationSettings(self, settings):
        self.restoration_on = settings.process['Restoration_on']
        self.restoraion_policy_type = settings.scenario['Restoraion_policy_type']
        self.restoraion_policy_addr = settings.scenario['Restortion_config_file']
        self.minimum_job_time = settings.process['minimum_job_time']
        self.pipe_damage_discovery_model = settings.scenario[
            'pipe_damage_discovery_model'
        ]
        self.node_damage_discovery_model = settings.scenario[
            'node_damage_discovery_model'
        ]
        self.pump_damage_discovery_model = settings.scenario[
            'pump_damage_discovery_model'
        ]
        self.tank_damage_discovery_model = settings.scenario[
            'tank_damage_discovery_model'
        ]
        self.out_of_zone_allowed = settings.scenario['crew_out_of_zone_travel']

    def browsePolicyDefinitionFile(self):
        if self.script_txt_radio.isChecked():
            file_type = 'scenrario text file (*.txt)'
        elif self.script_rrp_radio.isChecked():
            file_type = 'scenrario binary (*.rrp)'

        file = QtWidgets.QFileDialog.getOpenFileName(
            self.asli_MainWindow,
            'Open file',
            self.current_policy_directory,
            file_type,
        )
        if file[0] == '':
            return
        split_addr = os.path.split(file[0])
        self.current_policy_directory = split_addr[0]
        self.restoraion_policy_addr = file[0]
        self.policy_definition_addr_line.setText(file[0])

    def pipeDamageDiscoveryByButton(self):
        pipe_damage_discovery_designer = Pipe_Damage_Discovery_Designer(
            self.pipe_damage_discovery_model
        )
        return_value = pipe_damage_discovery_designer._window.exec_()

        if return_value == 1:
            self.pipe_damage_discovery_model = (
                pipe_damage_discovery_designer.damage_discovery_model
            )

    def nodeDamageDiscoveryByButton(self):
        node_damage_discovery_designer = Node_Damage_Discovery_Designer(
            self.node_damage_discovery_model
        )
        return_value = node_damage_discovery_designer._window.exec_()

        if return_value == 1:
            self.node_damage_discovery_model = (
                node_damage_discovery_designer.damage_discovery_model
            )

    def pumpDamageDiscoveryByButton(self):
        pump_damage_discovery_designer = Pump_Damage_Discovery_Designer(
            self.pump_damage_discovery_model
        )
        return_value = pump_damage_discovery_designer._window.exec_()

        if return_value == 1:
            self.pump_damage_discovery_model = (
                pump_damage_discovery_designer.damage_discovery_model
            )

    def tankDamageDiscoveryByButton(self):
        tank_damage_discovery_designer = Tank_Damage_Discovery_Designer(
            self.tank_damage_discovery_model
        )
        return_value = tank_damage_discovery_designer._window.exec_()

        if return_value == 1:
            self.tank_damage_discovery_model = (
                tank_damage_discovery_designer.damage_discovery_model
            )
