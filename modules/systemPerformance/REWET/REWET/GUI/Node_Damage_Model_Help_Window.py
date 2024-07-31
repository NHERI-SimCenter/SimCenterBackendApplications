# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Node_Damage_Model_Help_Window.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Node_Damage_Model_Help(object):
    def setupUi(self, Node_Damage_Model_Help):
        Node_Damage_Model_Help.setObjectName('Node_Damage_Model_Help')
        Node_Damage_Model_Help.resize(340, 130)
        Node_Damage_Model_Help.setMinimumSize(QtCore.QSize(340, 130))
        Node_Damage_Model_Help.setMaximumSize(QtCore.QSize(340, 130))
        self.layoutWidget = QtWidgets.QWidget(Node_Damage_Model_Help)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 20, 291, 101))
        self.layoutWidget.setObjectName('layoutWidget')
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName('verticalLayout_2')
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName('label')
        self.verticalLayout_2.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setObjectName('label_2')
        self.verticalLayout_2.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        self.label_3.setObjectName('label_3')
        self.verticalLayout_2.addWidget(self.label_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName('horizontalLayout_2')
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout_2.addItem(spacerItem)
        self.buttonBox = QtWidgets.QDialogButtonBox(self.layoutWidget)
        self.buttonBox.setOrientation(QtCore.Qt.Vertical)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Close)
        self.buttonBox.setObjectName('buttonBox')
        self.horizontalLayout_2.addWidget(self.buttonBox)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.retranslateUi(Node_Damage_Model_Help)
        QtCore.QMetaObject.connectSlotsByName(Node_Damage_Model_Help)

    def retranslateUi(self, Node_Damage_Model_Help):
        _translate = QtCore.QCoreApplication.translate
        Node_Damage_Model_Help.setWindowTitle(
            _translate('Node_Damage_Model_Help', 'Help')
        )
        self.label.setText(
            _translate('Node_Damage_Model_Help', 'ND = Number of Nodal damage')
        )
        self.label_2.setText(
            _translate('Node_Damage_Model_Help', 'MP = Pressure at the node')
        )
        self.label_3.setText(_translate('Node_Damage_Model_Help', 'RR =Repair Rate'))


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Node_Damage_Model_Help = QtWidgets.QDialog()
    ui = Ui_Node_Damage_Model_Help()
    ui.setupUi(Node_Damage_Model_Help)
    Node_Damage_Model_Help.show()
    sys.exit(app.exec_())
