"""Created on Tue Nov  1 20:36:29 2022

@author: snaeimi
"""

from PyQt5 import QtGui, QtWidgets

from .Node_Damage_Model_Help_Designer import Node_Damage_Model_Help_Designer
from .Node_Damage_Model_Window import Ui_Node_Damage_Model


class Node_Damage_Model_Designer(Ui_Node_Damage_Model):
    def __init__(self, node_damage_model):
        self._window = QtWidgets.QDialog()
        self.setupUi(self._window)
        self.node_damage_model = node_damage_model.copy()

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
        self.c_line.setValidator(
            QtGui.QDoubleValidator(
                -1000000,
                1000000,
                20,
                notation=QtGui.QDoubleValidator.StandardNotation,
            )
        )
        self.d_line.setValidator(
            QtGui.QDoubleValidator(
                -1000000,
                1000000,
                20,
                notation=QtGui.QDoubleValidator.StandardNotation,
            )
        )
        self.e_line.setValidator(
            QtGui.QDoubleValidator(
                -1000000,
                1000000,
                20,
                notation=QtGui.QDoubleValidator.StandardNotation,
            )
        )
        self.f_line.setValidator(
            QtGui.QDoubleValidator(
                -1000000,
                1000000,
                20,
                notation=QtGui.QDoubleValidator.StandardNotation,
            )
        )

        self.aa_line.setValidator(
            QtGui.QDoubleValidator(
                -1000000,
                1000000,
                20,
                notation=QtGui.QDoubleValidator.StandardNotation,
            )
        )
        self.bb_line.setValidator(
            QtGui.QDoubleValidator(
                -1000000,
                1000000,
                20,
                notation=QtGui.QDoubleValidator.StandardNotation,
            )
        )
        self.cc_line.setValidator(
            QtGui.QDoubleValidator(
                -1000000,
                1000000,
                20,
                notation=QtGui.QDoubleValidator.StandardNotation,
            )
        )
        self.dd_line.setValidator(
            QtGui.QDoubleValidator(
                -1000000,
                1000000,
                20,
                notation=QtGui.QDoubleValidator.StandardNotation,
            )
        )
        self.ee1_line.setValidator(
            QtGui.QDoubleValidator(
                -1000000,
                1000000,
                20,
                notation=QtGui.QDoubleValidator.StandardNotation,
            )
        )
        self.ff1_line.setValidator(
            QtGui.QDoubleValidator(
                -1000000,
                1000000,
                20,
                notation=QtGui.QDoubleValidator.StandardNotation,
            )
        )
        self.ee2_line.setValidator(
            QtGui.QDoubleValidator(
                -1000000,
                1000000,
                20,
                notation=QtGui.QDoubleValidator.StandardNotation,
            )
        )
        self.ff2_line.setValidator(
            QtGui.QDoubleValidator(
                -1000000,
                1000000,
                20,
                notation=QtGui.QDoubleValidator.StandardNotation,
            )
        )

        a = self.node_damage_model['a']
        b = self.node_damage_model['b']
        c = self.node_damage_model['c']
        d = self.node_damage_model['d']
        e = self.node_damage_model['e']
        f = self.node_damage_model['f']
        aa = self.node_damage_model['aa']
        bb = self.node_damage_model['bb']
        cc = self.node_damage_model['cc']
        dd = self.node_damage_model['dd']
        ee1 = self.node_damage_model['ee1']
        ff1 = self.node_damage_model['ff1']
        ee2 = self.node_damage_model['ee2']
        ff2 = self.node_damage_model['ff2']

        if self.node_damage_model['damage_node_model'] == 'equal_diameter_emitter':
            self.equal_emitter_button.setChecked(True)
        elif (
            self.node_damage_model['damage_node_model'] == 'equal_diameter_reservoir'
        ):
            self.equal_reservoir_button.setChecked(True)

        self.a_line.setText(str(a))
        self.b_line.setText(str(b))
        self.c_line.setText(str(c))
        self.d_line.setText(str(d))
        self.e_line.setText(str(e))
        self.f_line.setText(str(f))

        self.aa_line.setText(str(aa))
        self.bb_line.setText(str(bb))
        self.cc_line.setText(str(cc))
        self.dd_line.setText(str(dd))
        self.ee1_line.setText(str(ee1))
        self.ff1_line.setText(str(ff1))
        self.ee2_line.setText(str(ee2))
        self.ff2_line.setText(str(ff2))

        self.buttonBox.accepted.connect(self.okButtonPressed)
        self.help_button.clicked.connect(self.showHelpByButton)

    def showHelpByButton(self):
        help_dialog_box = Node_Damage_Model_Help_Designer()
        help_dialog_box._window.exec_()

    def okButtonPressed(self):
        a = self.a_line.text()
        b = self.b_line.text()
        c = self.c_line.text()
        d = self.d_line.text()
        e = self.e_line.text()
        f = self.f_line.text()
        aa = self.aa_line.text()
        bb = self.bb_line.text()
        cc = self.cc_line.text()
        dd = self.dd_line.text()
        ee1 = self.ee1_line.text()
        ff1 = self.ff1_line.text()
        ee2 = self.ee2_line.text()
        ff2 = self.ff2_line.text()

        if_failed = False

        if a == '':
            self.errorMSG('Cannot Save data', 'A Field cannot be left empty')
            if_failed = True
        elif aa == '':
            self.errorMSG('Cannot Save data', 'AA Field cannot be left empty')
            if_failed = True
        elif b == '':
            self.errorMSG('Cannot Save data', 'B Field cannot be left empty')
            if_failed = True
        elif bb == '':
            self.errorMSG('Cannot Save data', 'BB Field cannot be left empty')
            if_failed = True
        elif c == '':
            self.errorMSG('Cannot Save data', 'C Field cannot be left empty')
            if_failed = True
        elif cc == '':
            self.errorMSG('Cannot Save data', 'CC Field cannot be left empty')
            if_failed = True
        elif d == '':
            self.errorMSG('Cannot Save data', 'D Field cannot be left empty')
            if_failed = True
        elif dd == '':
            self.errorMSG('Cannot Save data', 'DD Field cannot be left empty')
            if_failed = True
        elif e == '':
            self.errorMSG('Cannot Save data', 'E Field cannot be left empty')
            if_failed = True
        elif ee1 == '':
            self.errorMSG('Cannot Save data', 'EE1 Field cannot be left empty')
            if_failed = True
        elif ee2 == '':
            self.errorMSG('Cannot Save data', 'EE2 Field cannot be left empty')
            if_failed = True
        elif f == '':
            self.errorMSG('Cannot Save data', 'F Field cannot be left empty')
            if_failed = True
        elif ff1 == '':
            self.errorMSG('Cannot Save data', 'FF1 Field cannot be left empty')
            if_failed = True
        elif ff2 == '':
            self.errorMSG('Cannot Save data', 'FF2 Field cannot be left empty')
            if_failed = True

        if if_failed:
            return

        self.node_damage_model['a'] = float(a)
        self.node_damage_model['b'] = float(b)
        self.node_damage_model['c'] = float(c)
        self.node_damage_model['d'] = float(d)
        self.node_damage_model['e'] = float(e)
        self.node_damage_model['f'] = float(f)
        self.node_damage_model['aa'] = float(aa)
        self.node_damage_model['bb'] = float(bb)
        self.node_damage_model['cc'] = float(cc)
        self.node_damage_model['dd'] = float(dd)
        self.node_damage_model['ee1'] = float(ee1)
        self.node_damage_model['ff1'] = float(ff1)
        self.node_damage_model['ee2'] = float(ee2)
        self.node_damage_model['ff2'] = float(ff2)

        if self.equal_emitter_button.isChecked():
            self.node_damage_model['damage_node_model'] = 'equal_diameter_emitter'
        elif self.equal_reservoir_button.isChecked():
            self.node_damage_model['damage_node_model'] = 'equal_diameter_reservoir'

        self._window.accept()

    def errorMSG(self, error_title, error_msg, error_more_msg=None):
        error_widget = QtWidgets.QMessageBox()
        error_widget.setIcon(QtWidgets.QMessageBox.Critical)
        error_widget.setText(error_msg)
        error_widget.setWindowTitle(error_title)
        error_widget.setStandardButtons(QtWidgets.QMessageBox.Ok)
        if error_more_msg != None:
            error_widget.setInformativeText(error_more_msg)
        error_widget.exec_()
