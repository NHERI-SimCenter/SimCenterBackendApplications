# Form implementation generated from reading ui file 'Subsitute_Layer_Window.ui'  # noqa: CPY001, D100, N999
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_subsitite_layer_dialoge:  # noqa: D101
    def setupUi(self, subsitite_layer_dialoge):  # noqa: N802, D102
        subsitite_layer_dialoge.setObjectName('subsitite_layer_dialoge')
        subsitite_layer_dialoge.resize(403, 407)
        self.Subsitute_buttonBox = QtWidgets.QDialogButtonBox(
            subsitite_layer_dialoge
        )
        self.Subsitute_buttonBox.setGeometry(QtCore.QRect(110, 360, 261, 32))
        self.Subsitute_buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.Subsitute_buttonBox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Apply
            | QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.Ok
        )
        self.Subsitute_buttonBox.setObjectName('Subsitute_buttonBox')
        self.subsitute_layer_addr_line = QtWidgets.QLineEdit(subsitite_layer_dialoge)
        self.subsitute_layer_addr_line.setGeometry(QtCore.QRect(20, 40, 261, 20))
        self.subsitute_layer_addr_line.setReadOnly(True)
        self.subsitute_layer_addr_line.setObjectName('subsitute_layer_addr_line')
        self.population_browser_button = QtWidgets.QPushButton(
            subsitite_layer_dialoge
        )
        self.population_browser_button.setGeometry(QtCore.QRect(290, 40, 81, 23))
        self.population_browser_button.setObjectName('population_browser_button')
        self.label_27 = QtWidgets.QLabel(subsitite_layer_dialoge)
        self.label_27.setGeometry(QtCore.QRect(20, 20, 121, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.label_27.setFont(font)
        self.label_27.setObjectName('label_27')
        self.groupBox = QtWidgets.QGroupBox(subsitite_layer_dialoge)
        self.groupBox.setGeometry(QtCore.QRect(20, 70, 351, 61))
        self.groupBox.setObjectName('groupBox')
        self.subsitute_layer_projection_name_line = QtWidgets.QLineEdit(
            self.groupBox
        )
        self.subsitute_layer_projection_name_line.setGeometry(
            QtCore.QRect(110, 30, 231, 20)
        )
        self.subsitute_layer_projection_name_line.setReadOnly(True)
        self.subsitute_layer_projection_name_line.setObjectName(
            'subsitute_layer_projection_name_line'
        )
        self.label_28 = QtWidgets.QLabel(self.groupBox)
        self.label_28.setGeometry(QtCore.QRect(10, 30, 121, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.label_28.setFont(font)
        self.label_28.setObjectName('label_28')
        self.iUse_sub_checkbox = QtWidgets.QCheckBox(subsitite_layer_dialoge)
        self.iUse_sub_checkbox.setGeometry(QtCore.QRect(20, 150, 141, 17))
        self.iUse_sub_checkbox.setObjectName('iUse_sub_checkbox')
        self.sub_error_text_edit = QtWidgets.QTextEdit(subsitite_layer_dialoge)
        self.sub_error_text_edit.setGeometry(QtCore.QRect(25, 191, 341, 151))
        self.sub_error_text_edit.setObjectName('sub_error_text_edit')
        self.label_29 = QtWidgets.QLabel(subsitite_layer_dialoge)
        self.label_29.setGeometry(QtCore.QRect(30, 170, 121, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.label_29.setFont(font)
        self.label_29.setObjectName('label_29')

        self.retranslateUi(subsitite_layer_dialoge)
        self.Subsitute_buttonBox.accepted.connect(subsitite_layer_dialoge.accept)
        self.Subsitute_buttonBox.rejected.connect(subsitite_layer_dialoge.reject)
        QtCore.QMetaObject.connectSlotsByName(subsitite_layer_dialoge)

    def retranslateUi(self, subsitite_layer_dialoge):  # noqa: N802, D102
        _translate = QtCore.QCoreApplication.translate
        subsitite_layer_dialoge.setWindowTitle(
            _translate('subsitite_layer_dialoge', 'Dialog')
        )
        self.population_browser_button.setText(
            _translate('subsitite_layer_dialoge', 'Browse')
        )
        self.label_27.setText(
            _translate('subsitite_layer_dialoge', 'Subsitute Layer File')
        )
        self.groupBox.setTitle(
            _translate('subsitite_layer_dialoge', 'Projection System')
        )
        self.label_28.setText(
            _translate('subsitite_layer_dialoge', 'Subsitute Projection')
        )
        self.iUse_sub_checkbox.setText(
            _translate('subsitite_layer_dialoge', 'Use the substitute Layer')
        )
        self.label_29.setText(_translate('subsitite_layer_dialoge', 'Warnings'))


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    subsitite_layer_dialoge = QtWidgets.QDialog()
    ui = Ui_subsitite_layer_dialoge()
    ui.setupUi(subsitite_layer_dialoge)
    subsitite_layer_dialoge.show()
    sys.exit(app.exec_())
