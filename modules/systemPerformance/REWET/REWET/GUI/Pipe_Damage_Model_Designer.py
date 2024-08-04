"""Created on Tue Nov  1 18:32:32 2022

@author: snaeimi
"""  # noqa: CPY001, D400, N999

from PyQt5 import QtGui, QtWidgets

from .Pipe_Damage_Model_Window import Ui_Pipe_Damage_Model


class Pipe_Damage_Model_Designer(Ui_Pipe_Damage_Model):  # noqa: D101
    def __init__(self, pipe_damage_model):  # noqa: ANN001, ANN204
        self._window = QtWidgets.QDialog()
        self.setupUi(self._window)
        self.pipe_damage_model = pipe_damage_model
        self.material_list.addItems(pipe_damage_model.keys())
        self.alpha_line.setValidator(
            QtGui.QDoubleValidator(
                -1000000,
                1000000,
                20,
                notation=QtGui.QDoubleValidator.StandardNotation,
            )
        )
        self.beta_line.setValidator(
            QtGui.QDoubleValidator(
                -1000000,
                1000000,
                20,
                notation=QtGui.QDoubleValidator.StandardNotation,
            )
        )
        self.gamma_line.setValidator(
            QtGui.QDoubleValidator(
                -1000000,
                1000000,
                20,
                notation=QtGui.QDoubleValidator.StandardNotation,
            )
        )
        self.a_line.setValidator(
            QtGui.QDoubleValidator(
                -1000000,
                1000000,
                20,
                notation=QtGui.QDoubleValidator.StandardNotation,
            )
        )
        self.b_line.setValidator(
            QtGui.QDoubleValidator(
                -1000000,
                1000000,
                20,
                notation=QtGui.QDoubleValidator.StandardNotation,
            )
        )

        self.buttonBox.accepted.connect(self.okButtonPressed)

        self.material_list.currentItemChanged.connect(self.materialChanged)

    def materialChanged(self, current_item, previous_item):  # noqa: ANN001, ANN201, N802, D102
        if previous_item != None:  # noqa: E711
            previous_material = previous_item.text()

            alpha = self.alpha_line.text()
            beta = self.beta_line.text()
            gamma = self.gamma_line.text()
            a = self.a_line.text()
            b = self.b_line.text()

            self.pipe_damage_model[previous_material]['alpha'] = float(alpha)
            self.pipe_damage_model[previous_material]['beta'] = float(beta)
            self.pipe_damage_model[previous_material]['gamma'] = float(gamma)
            self.pipe_damage_model[previous_material]['a'] = float(a)
            self.pipe_damage_model[previous_material]['b'] = float(b)

        current_material = current_item.text()
        alpha = self.pipe_damage_model[current_material]['alpha']
        beta = self.pipe_damage_model[current_material]['beta']
        gamma = self.pipe_damage_model[current_material]['gamma']
        a = self.pipe_damage_model[current_material]['a']
        b = self.pipe_damage_model[current_material]['b']

        self.alpha_line.setText(str(alpha))
        self.beta_line.setText(str(beta))
        self.gamma_line.setText(str(gamma))
        self.a_line.setText(str(a))
        self.b_line.setText(str(b))

    def okButtonPressed(self):  # noqa: ANN201, N802, D102
        current_material = self.material_list.selectedItems()[0].text()

        alpha = self.alpha_line.text()
        beta = self.beta_line.text()
        gamma = self.gamma_line.text()
        a = self.a_line.text()
        b = self.b_line.text()

        self.pipe_damage_model[current_material]['alpha'] = float(alpha)
        self.pipe_damage_model[current_material]['beta'] = float(beta)
        self.pipe_damage_model[current_material]['gamma'] = float(gamma)
        self.pipe_damage_model[current_material]['a'] = float(a)
        self.pipe_damage_model[current_material]['b'] = float(b)
