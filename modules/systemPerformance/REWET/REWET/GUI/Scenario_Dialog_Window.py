# Form implementation generated from reading ui file 'Scenario_Dialog_Window.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtWidgets


class Ui_Scenario_Dialog:
    def setupUi(self, Scenario_Dialog):
        Scenario_Dialog.setObjectName('Scenario_Dialog')
        Scenario_Dialog.resize(351, 241)
        self.buttonBox = QtWidgets.QDialogButtonBox(Scenario_Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(260, 40, 81, 241))
        self.buttonBox.setOrientation(QtCore.Qt.Vertical)
        self.buttonBox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok
        )
        self.buttonBox.setObjectName('buttonBox')
        self.scenario_name_line = QtWidgets.QLineEdit(Scenario_Dialog)
        self.scenario_name_line.setGeometry(QtCore.QRect(110, 40, 113, 20))
        self.scenario_name_line.setObjectName('scenario_name_line')
        self.pipe_damage_line = QtWidgets.QLineEdit(Scenario_Dialog)
        self.pipe_damage_line.setGeometry(QtCore.QRect(110, 70, 113, 20))
        self.pipe_damage_line.setObjectName('pipe_damage_line')
        self.node_damage_line = QtWidgets.QLineEdit(Scenario_Dialog)
        self.node_damage_line.setGeometry(QtCore.QRect(110, 100, 113, 20))
        self.node_damage_line.setObjectName('node_damage_line')
        self.pump_damage_line = QtWidgets.QLineEdit(Scenario_Dialog)
        self.pump_damage_line.setGeometry(QtCore.QRect(110, 130, 113, 20))
        self.pump_damage_line.setObjectName('pump_damage_line')
        self.label = QtWidgets.QLabel(Scenario_Dialog)
        self.label.setGeometry(QtCore.QRect(20, 40, 91, 16))
        self.label.setObjectName('label')
        self.label_2 = QtWidgets.QLabel(Scenario_Dialog)
        self.label_2.setGeometry(QtCore.QRect(20, 70, 71, 16))
        self.label_2.setObjectName('label_2')
        self.label_3 = QtWidgets.QLabel(Scenario_Dialog)
        self.label_3.setGeometry(QtCore.QRect(20, 100, 71, 16))
        self.label_3.setObjectName('label_3')
        self.label_4 = QtWidgets.QLabel(Scenario_Dialog)
        self.label_4.setGeometry(QtCore.QRect(20, 130, 71, 16))
        self.label_4.setObjectName('label_4')
        self.label_5 = QtWidgets.QLabel(Scenario_Dialog)
        self.label_5.setGeometry(QtCore.QRect(20, 160, 81, 16))
        self.label_5.setObjectName('label_5')
        self.label_6 = QtWidgets.QLabel(Scenario_Dialog)
        self.label_6.setGeometry(QtCore.QRect(20, 190, 61, 16))
        self.label_6.setObjectName('label_6')
        self.tank_damage_line = QtWidgets.QLineEdit(Scenario_Dialog)
        self.tank_damage_line.setGeometry(QtCore.QRect(110, 160, 113, 20))
        self.tank_damage_line.setObjectName('tank_damage_line')
        self.probability_line = QtWidgets.QLineEdit(Scenario_Dialog)
        self.probability_line.setGeometry(QtCore.QRect(110, 190, 113, 20))
        self.probability_line.setObjectName('probability_line')

        self.retranslateUi(Scenario_Dialog)
        self.buttonBox.accepted.connect(Scenario_Dialog.accept)
        self.buttonBox.rejected.connect(Scenario_Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Scenario_Dialog)

    def retranslateUi(self, Scenario_Dialog):
        _translate = QtCore.QCoreApplication.translate
        Scenario_Dialog.setWindowTitle(_translate('Scenario_Dialog', 'New Scenario'))
        self.label.setText(_translate('Scenario_Dialog', 'Scenario Name'))
        self.label_2.setText(_translate('Scenario_Dialog', 'Pipe Damage'))
        self.label_3.setText(_translate('Scenario_Dialog', 'Nodal Damage'))
        self.label_4.setText(_translate('Scenario_Dialog', 'Pump Damage'))
        self.label_5.setText(_translate('Scenario_Dialog', 'Tank Damage'))
        self.label_6.setText(_translate('Scenario_Dialog', 'Probability'))
        self.probability_line.setText(_translate('Scenario_Dialog', '1'))


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Scenario_Dialog = QtWidgets.QDialog()
    ui = Ui_Scenario_Dialog()
    ui.setupUi(Scenario_Dialog)
    Scenario_Dialog.show()
    sys.exit(app.exec_())
