"""Created on Wed Nov  2 13:25:40 2022

@author: snaeimi
"""

from PyQt5 import QtWidgets

from .Main_Help_Window import Ui_Main_Help_Window


class Main_Help_Designer(Ui_Main_Help_Window):
    def __init__(self):
        self._window = QtWidgets.QDialog()
        self.setupUi(self._window)

        self.buttonBox.rejected.connect(self._window.close)
