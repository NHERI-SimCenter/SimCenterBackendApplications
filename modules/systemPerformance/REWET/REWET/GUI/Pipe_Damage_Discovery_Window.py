# Form implementation generated from reading ui file 'Pipe_Damage_Discovery_Window.ui'  # noqa: N999, D100
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtWidgets


class Ui_pipe_damage_discovery:  # noqa: N801, D101
    def setupUi(self, pipe_damage_discovery):  # noqa: ANN001, ANN201, N802, D102, PLR0915
        pipe_damage_discovery.setObjectName('pipe_damage_discovery')
        pipe_damage_discovery.resize(450, 400)
        pipe_damage_discovery.setMinimumSize(QtCore.QSize(450, 400))
        pipe_damage_discovery.setMaximumSize(QtCore.QSize(450, 400))
        self.buttonBox = QtWidgets.QDialogButtonBox(pipe_damage_discovery)
        self.buttonBox.setGeometry(QtCore.QRect(350, 20, 81, 61))
        self.buttonBox.setOrientation(QtCore.Qt.Vertical)
        self.buttonBox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok
        )
        self.buttonBox.setObjectName('buttonBox')
        self.groupBox = QtWidgets.QGroupBox(pipe_damage_discovery)
        self.groupBox.setGeometry(QtCore.QRect(19, 19, 311, 351))
        self.groupBox.setObjectName('groupBox')
        self.leak_amount_line = QtWidgets.QLineEdit(self.groupBox)
        self.leak_amount_line.setGeometry(QtCore.QRect(80, 50, 51, 20))
        self.leak_amount_line.setObjectName('leak_amount_line')
        self.leak_anount_label = QtWidgets.QLabel(self.groupBox)
        self.leak_anount_label.setGeometry(QtCore.QRect(10, 50, 71, 16))
        self.leak_anount_label.setObjectName('leak_anount_label')
        self.leak_time_line = QtWidgets.QLineEdit(self.groupBox)
        self.leak_time_line.setGeometry(QtCore.QRect(210, 50, 81, 20))
        self.leak_time_line.setObjectName('leak_time_line')
        self.time_discovery_ratio_table = QtWidgets.QTableWidget(self.groupBox)
        self.time_discovery_ratio_table.setGeometry(QtCore.QRect(10, 141, 211, 191))
        self.time_discovery_ratio_table.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        self.time_discovery_ratio_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )
        self.time_discovery_ratio_table.setRowCount(0)
        self.time_discovery_ratio_table.setObjectName('time_discovery_ratio_table')
        self.time_discovery_ratio_table.setColumnCount(2)
        item = QtWidgets.QTableWidgetItem()
        self.time_discovery_ratio_table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.time_discovery_ratio_table.setHorizontalHeaderItem(1, item)
        self.time_discovery_ratio_table.horizontalHeader().setStretchLastSection(
            True
        )
        self.time_discovery_ratio_table.verticalHeader().setVisible(False)
        self.time_discovery_ratio_table.verticalHeader().setStretchLastSection(False)
        self.leak_time_label = QtWidgets.QLabel(self.groupBox)
        self.leak_time_label.setGeometry(QtCore.QRect(150, 50, 61, 16))
        self.leak_time_label.setObjectName('leak_time_label')
        self.leak_based_radio = QtWidgets.QRadioButton(self.groupBox)
        self.leak_based_radio.setGeometry(QtCore.QRect(10, 20, 111, 17))
        self.leak_based_radio.setObjectName('leak_based_radio')
        self.time_based_radio = QtWidgets.QRadioButton(self.groupBox)
        self.time_based_radio.setGeometry(QtCore.QRect(10, 90, 111, 17))
        self.time_based_radio.setObjectName('time_based_radio')
        self.time_line = QtWidgets.QLineEdit(self.groupBox)
        self.time_line.setGeometry(QtCore.QRect(10, 120, 101, 20))
        self.time_line.setObjectName('time_line')
        self.discovery_ratio_line = QtWidgets.QLineEdit(self.groupBox)
        self.discovery_ratio_line.setGeometry(QtCore.QRect(110, 120, 111, 20))
        self.discovery_ratio_line.setObjectName('discovery_ratio_line')
        self.add_button = QtWidgets.QPushButton(self.groupBox)
        self.add_button.setGeometry(QtCore.QRect(230, 120, 51, 23))
        self.add_button.setObjectName('add_button')
        self.remove_button = QtWidgets.QPushButton(self.groupBox)
        self.remove_button.setGeometry(QtCore.QRect(230, 150, 51, 23))
        self.remove_button.setObjectName('remove_button')

        self.retranslateUi(pipe_damage_discovery)
        self.buttonBox.rejected.connect(pipe_damage_discovery.reject)
        QtCore.QMetaObject.connectSlotsByName(pipe_damage_discovery)

    def retranslateUi(self, pipe_damage_discovery):  # noqa: ANN001, ANN201, N802, D102
        _translate = QtCore.QCoreApplication.translate
        pipe_damage_discovery.setWindowTitle(
            _translate('pipe_damage_discovery', 'Pipe Damaeg Discovery')
        )
        self.groupBox.setTitle(_translate('pipe_damage_discovery', 'Leak Model'))
        self.leak_anount_label.setText(
            _translate('pipe_damage_discovery', 'Leak Amount')
        )
        item = self.time_discovery_ratio_table.horizontalHeaderItem(0)
        item.setText(_translate('pipe_damage_discovery', 'Time'))
        item = self.time_discovery_ratio_table.horizontalHeaderItem(1)
        item.setText(_translate('pipe_damage_discovery', 'Discovery Ratio'))
        self.leak_time_label.setText(
            _translate('pipe_damage_discovery', 'leak time')
        )
        self.leak_based_radio.setText(
            _translate('pipe_damage_discovery', 'Leak Based')
        )
        self.time_based_radio.setText(
            _translate('pipe_damage_discovery', 'Time Based')
        )
        self.add_button.setText(_translate('pipe_damage_discovery', 'add'))
        self.remove_button.setText(_translate('pipe_damage_discovery', 'Remove'))


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    pipe_damage_discovery = QtWidgets.QDialog()
    ui = Ui_pipe_damage_discovery()
    ui.setupUi(pipe_damage_discovery)
    pipe_damage_discovery.show()
    sys.exit(app.exec_())
