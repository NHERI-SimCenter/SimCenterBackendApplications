"""Created on Tue Nov  1 21:35:02 2022

@author: snaeimi
"""  # noqa: N999, D400

from PyQt5 import QtWidgets

from .Node_Damage_Model_Help_Window import Ui_Node_Damage_Model_Help


class Node_Damage_Model_Help_Designer(Ui_Node_Damage_Model_Help):  # noqa: D101
    def __init__(self):
        self._window = QtWidgets.QDialog()
        self.setupUi(self._window)
        self.buttonBox.rejected.connect(self._window.close)
