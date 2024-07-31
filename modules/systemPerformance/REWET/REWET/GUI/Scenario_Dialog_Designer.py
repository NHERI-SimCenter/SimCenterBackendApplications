# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:09:49 2022

@author: snaeimi
"""

import os
from PyQt5 import QtCore, QtGui, QtWidgets
from .Scenario_Dialog_Window import Ui_Scenario_Dialog


class Scenario_Dialog_Designer(Ui_Scenario_Dialog):
    def __init__(self):
        self._window = QtWidgets.QDialog()
        self.setupUi(self._window)
        self.last_probability = 1
        self.probability_line.setValidator(
            QtGui.QDoubleValidator(
                0.0, 1, 3, notation=QtGui.QDoubleValidator.StandardNotation
            )
        )
        self.probability_line.textChanged.connect(self.probabilityValidatorHelper)

    def probabilityValidatorHelper(self, text):
        if float(text) > 1:
            self.probability_line.setText(self.last_probability)
        else:
            self.last_probability = text
