# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Pipe_Damage_Model_Window.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Pipe_Damage_Model(object):
    def setupUi(self, Pipe_Damage_Model):
        Pipe_Damage_Model.setObjectName('Pipe_Damage_Model')
        Pipe_Damage_Model.resize(377, 372)
        self.buttonBox = QtWidgets.QDialogButtonBox(Pipe_Damage_Model)
        self.buttonBox.setGeometry(QtCore.QRect(260, 50, 81, 91))
        self.buttonBox.setOrientation(QtCore.Qt.Vertical)
        self.buttonBox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok
        )
        self.buttonBox.setObjectName('buttonBox')
        self.material_list = QtWidgets.QListWidget(Pipe_Damage_Model)
        self.material_list.setGeometry(QtCore.QRect(10, 50, 231, 192))
        self.material_list.setObjectName('material_list')
        self.label = QtWidgets.QLabel(Pipe_Damage_Model)
        self.label.setGeometry(QtCore.QRect(10, 30, 101, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName('label')
        self.alpha_line = QtWidgets.QLineEdit(Pipe_Damage_Model)
        self.alpha_line.setGeometry(QtCore.QRect(60, 300, 50, 20))
        self.alpha_line.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.alpha_line.setText('')
        self.alpha_line.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter
        )
        self.alpha_line.setObjectName('alpha_line')
        self.label_2 = QtWidgets.QLabel(Pipe_Damage_Model)
        self.label_2.setGeometry(QtCore.QRect(115, 302, 16, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setObjectName('label_2')
        self.a_line = QtWidgets.QLineEdit(Pipe_Damage_Model)
        self.a_line.setGeometry(QtCore.QRect(125, 280, 41, 20))
        self.a_line.setText('')
        self.a_line.setObjectName('a_line')
        self.label_3 = QtWidgets.QLabel(Pipe_Damage_Model)
        self.label_3.setGeometry(QtCore.QRect(165, 300, 16, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName('label_3')
        self.label_4 = QtWidgets.QLabel(Pipe_Damage_Model)
        self.label_4.setGeometry(QtCore.QRect(285, 300, 16, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName('label_4')
        self.beta_line = QtWidgets.QLineEdit(Pipe_Damage_Model)
        self.beta_line.setGeometry(QtCore.QRect(180, 300, 50, 20))
        self.beta_line.setText('')
        self.beta_line.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter
        )
        self.beta_line.setObjectName('beta_line')
        self.label_5 = QtWidgets.QLabel(Pipe_Damage_Model)
        self.label_5.setGeometry(QtCore.QRect(235, 300, 16, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setObjectName('label_5')
        self.b_line = QtWidgets.QLineEdit(Pipe_Damage_Model)
        self.b_line.setGeometry(QtCore.QRect(245, 280, 41, 20))
        self.b_line.setText('')
        self.b_line.setObjectName('b_line')
        self.gamma_line = QtWidgets.QLineEdit(Pipe_Damage_Model)
        self.gamma_line.setGeometry(QtCore.QRect(300, 300, 50, 20))
        self.gamma_line.setText('')
        self.gamma_line.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter
        )
        self.gamma_line.setObjectName('gamma_line')
        self.label_6 = QtWidgets.QLabel(Pipe_Damage_Model)
        self.label_6.setGeometry(QtCore.QRect(10, 300, 51, 20))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_6.setFont(font)
        self.label_6.setObjectName('label_6')

        self.retranslateUi(Pipe_Damage_Model)
        self.buttonBox.accepted.connect(Pipe_Damage_Model.accept)
        self.buttonBox.rejected.connect(Pipe_Damage_Model.reject)
        QtCore.QMetaObject.connectSlotsByName(Pipe_Damage_Model)

    def retranslateUi(self, Pipe_Damage_Model):
        _translate = QtCore.QCoreApplication.translate
        Pipe_Damage_Model.setWindowTitle(
            _translate('Pipe_Damage_Model', 'Pipe Damage Model')
        )
        self.label.setText(_translate('Pipe_Damage_Model', 'Pipe Material'))
        self.label_2.setText(_translate('Pipe_Damage_Model', 'D'))
        self.label_3.setText(_translate('Pipe_Damage_Model', '+'))
        self.label_4.setText(_translate('Pipe_Damage_Model', '+'))
        self.label_5.setText(_translate('Pipe_Damage_Model', 'D'))
        self.label_6.setText(_translate('Pipe_Damage_Model', 'opening='))


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Pipe_Damage_Model = QtWidgets.QDialog()
    ui = Ui_Pipe_Damage_Model()
    ui.setupUi(Pipe_Damage_Model)
    Pipe_Damage_Model.show()
    sys.exit(app.exec_())
