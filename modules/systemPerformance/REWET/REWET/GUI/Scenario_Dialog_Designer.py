"""Created on Fri Oct 28 14:09:49 2022

@author: snaeimi
"""  # noqa: CPY001, D400, N999

from PyQt5 import QtGui, QtWidgets

from .Scenario_Dialog_Window import Ui_Scenario_Dialog


class Scenario_Dialog_Designer(Ui_Scenario_Dialog):  # noqa: D101
    def __init__(self):  # noqa: ANN204
        self._window = QtWidgets.QDialog()
        self.setupUi(self._window)
        self.last_probability = 1
        self.probability_line.setValidator(
            QtGui.QDoubleValidator(
                0.0, 1, 3, notation=QtGui.QDoubleValidator.StandardNotation
            )
        )
        self.probability_line.textChanged.connect(self.probabilityValidatorHelper)

    def probabilityValidatorHelper(self, text):  # noqa: ANN001, ANN201, N802, D102
        if float(text) > 1:
            self.probability_line.setText(self.last_probability)
        else:
            self.last_probability = text
