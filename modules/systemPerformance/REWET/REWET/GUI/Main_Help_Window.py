# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Main_Help_Window.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Main_Help_Window(object):
    def setupUi(self, Main_Help_Window):
        Main_Help_Window.setObjectName("Main_Help_Window")
        Main_Help_Window.resize(680, 320)
        Main_Help_Window.setMinimumSize(QtCore.QSize(680, 320))
        Main_Help_Window.setMaximumSize(QtCore.QSize(680, 320))
        self.layoutWidget = QtWidgets.QWidget(Main_Help_Window)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 20, 641, 281))
        self.layoutWidget.setObjectName("layoutWidget")
        self.main_layout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setObjectName("main_layout")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.main_layout.addWidget(self.label)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout_4.setObjectName("gridLayout_4")
        spacerItem = QtWidgets.QSpacerItem(50, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem, 1, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem1, 0, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap(":/resources/resources/both_logos.jpg"))
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 0, 1, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem2, 0, 2, 1, 1)
        self.main_layout.addLayout(self.gridLayout_4)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.buttonBox = QtWidgets.QDialogButtonBox(self.layoutWidget)
        self.buttonBox.setOrientation(QtCore.Qt.Vertical)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Close)
        self.buttonBox.setObjectName("buttonBox")
        self.horizontalLayout_2.addWidget(self.buttonBox)
        self.main_layout.addLayout(self.horizontalLayout_2)

        self.retranslateUi(Main_Help_Window)
        QtCore.QMetaObject.connectSlotsByName(Main_Help_Window)

    def retranslateUi(self, Main_Help_Window):
        _translate = QtCore.QCoreApplication.translate
        Main_Help_Window.setWindowTitle(_translate("Main_Help_Window", "Help"))
        self.label.setText(_translate("Main_Help_Window", "REstoration of Water after an Event Tool (REWET) is created by Sina Naeimi and Rachel Davidson at University of Delaware. REWET is as it as and developers gurantee neither the usability of the software nor the validity of data it produce in any way."))
from . import REWET_Resource_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Main_Help_Window = QtWidgets.QDialog()
    ui = Ui_Main_Help_Window()
    ui.setupUi(Main_Help_Window)
    Main_Help_Window.show()
    sys.exit(app.exec_())
