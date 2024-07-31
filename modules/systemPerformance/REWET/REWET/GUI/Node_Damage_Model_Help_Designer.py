# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 21:35:02 2022

@author: snaeimi
"""

from PyQt5 import QtWidgets
from .Node_Damage_Model_Help_Window import Ui_Node_Damage_Model_Help


class Node_Damage_Model_Help_Designer(Ui_Node_Damage_Model_Help):
    def __init__(self):
        self._window = QtWidgets.QDialog()
        self.setupUi(self._window)
        self.buttonBox.rejected.connect(self._window.close)
