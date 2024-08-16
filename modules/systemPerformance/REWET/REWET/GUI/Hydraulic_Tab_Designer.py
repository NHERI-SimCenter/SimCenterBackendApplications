"""Created on Thu Oct 27 19:19:02 2022

@author: snaeimi
"""  # noqa: N999, D400

import os

from PyQt5 import QtGui, QtWidgets


class Hydraulic_Tab_Designer:  # noqa: D101
    def __init__(self):
        """These are variables that are shared between ui and settings."""  # noqa: D401
        self.setHydraulicSettings(self.settings)

        """
        ui value assignments.
        """
        self.setHydraulicUI()
        """
        Field Validators.
        """
        self.demand_ratio_line.setValidator(
            QtGui.QDoubleValidator(
                0.0, 1.0, 3, notation=QtGui.QDoubleValidator.StandardNotation
            )
        )
        self.minimum_pressure_line.setValidator(QtGui.QIntValidator(0.0, 2147483647))
        self.required_pressure_line.setValidator(
            QtGui.QIntValidator(0.0, 2147483647)
        )
        self.hydraulic_time_step_line.setValidator(
            QtGui.QIntValidator(0.0, 2147483647)
        )

        """
        Signals connection.
        """
        self.demand_ratio_line.textChanged.connect(self.demandRatioValidatorHelper)
        self.wdn_browse_button.clicked.connect(self.wdnFileBroweserClicked)
        self.hydraulic_time_step_line.textEdited.connect(
            self.hydraulicTimeStepValidatorHelper
        )
        self.current_inp_directory = os.getcwd()  # noqa: PTH109

    def getHydraulicSettings(self):  # noqa: N802, D102
        if self.wn_inp == '':
            self.errorMSG(
                'REWET', 'Water distribution network File must be provided'
            )
            return False

        # self.wn_inp -- already set
        self.demand_ratio = float(self.demand_ratio_line.text())
        self.minimum_pressure = float(self.minimum_pressure_line.text())
        self.required_pressure = float(self.required_pressure_line.text())
        self.hydraulic_time_step = int(float(self.hydraulic_time_step_line.text()))

        self.settings.process['WN_INP'] = self.wn_inp
        self.settings.process['demand_ratio'] = self.demand_ratio
        self.settings.process['solver_type'] = self.solver
        self.settings.scenario['minimum_pressure'] = self.minimum_pressure
        self.settings.scenario['required_pressure'] = self.required_pressure
        self.settings.scenario['hydraulic_time_step'] = self.hydraulic_time_step

        return True

    def setHydraulicUI(self):  # noqa: N802, D102
        self.wdn_addr_line.setText(self.wn_inp)
        self.last_demand_ratio_value = self.demand_ratio
        self.demand_ratio_line.setText(str(self.last_demand_ratio_value))

        if self.solver == 'ModifiedEPANETV2.2':
            self.modified_epanet_radio.setChecked(True)
        elif self.solver == 'WNTR':
            self.wntr_solver_radio.setChecked(True)
        else:
            raise ValueError('Unknown value for solver: ' + repr(self.solver))

        self.minimum_pressure_line.setText(str(self.minimum_pressure))
        self.required_pressure_line.setText(str(self.required_pressure))
        self.hydraulic_time_step_line.setText(str(self.hydraulic_time_step))

    def setHydraulicSettings(self, settings):  # noqa: N802, D102
        self.wn_inp = settings.process['WN_INP']
        self.demand_ratio = settings.process['demand_ratio']
        self.solver = settings.process['solver_type']
        self.minimum_pressure = settings.scenario['minimum_pressure']
        self.required_pressure = settings.scenario['required_pressure']
        self.hydraulic_time_step = settings.scenario['hydraulic_time_step']

    def demandRatioValidatorHelper(self, x):  # noqa: N802, D102
        if float(x) > 1:
            self.demand_ratio_line.setText(self.last_demand_ratio_value)
        else:
            self.last_demand_ratio_value = x
        # print(x)

    def hydraulicTimeStepValidatorHelper(self, x):  # noqa: ARG002, N802, D102
        try:
            hydraulic_time_step = int(float(self.hydraulic_time_step_line.text()))
        except:  # noqa: E722
            hydraulic_time_step = 0
        simulation_time_step = int(float(self.simulation_time_step_line.text()))

        if hydraulic_time_step > simulation_time_step:
            self.hydraulic_time_step_line.setText(str(simulation_time_step))

    def wdnFileBroweserClicked(self):  # noqa: N802, D102
        file = QtWidgets.QFileDialog.getOpenFileName(
            self.asli_MainWindow,
            'Open file',
            self.current_inp_directory,
            'inp file (*.inp)',
        )
        if file[0] == '':
            return
        split_addr = os.path.split(file[0])
        self.current_inp_directory = split_addr[0]
        self.wn_inp = file[0]

        self.wdn_addr_line.setText(file[0])
