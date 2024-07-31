"""Created on Thu Oct 27 18:06:01 2022

@author: snaeimi
"""  # noqa: N999, D400, D415

import os
import pickle
import sys

from Input.Settings import Settings
from Project import Project
from PyQt5 import QtWidgets
from PyQt5.Qt import QDesktopServices, QUrl

from .Damage_Tab_Designer import Damage_Tab_Designer
from .Hydraulic_Tab_Designer import Hydraulic_Tab_Designer
from .Main_Help_Designer import Main_Help_Designer
from .Map_Designer import Map_Designer
from .Opening_Window import Ui_Opening_Window
from .PP_Data_Tab_Designer import PP_Data_Tab
from .Restoration_Tab_Designer import Restoration_Tab_Designer
from .Result_Designer import Result_Designer
from .Run_Tab_Designer import Run_Tab_Designer
from .Simulation_Tab_Designer import Simulation_Tab_Designer


class Opening_Designer(  # noqa: N801, D101
    Ui_Opening_Window,
    Simulation_Tab_Designer,
    Hydraulic_Tab_Designer,
    Damage_Tab_Designer,
    Run_Tab_Designer,
    Restoration_Tab_Designer,
    PP_Data_Tab,
    Result_Designer,
    Map_Designer,
):
    def __init__(self):  # noqa: ANN204, D107
        self.project = None
        self.scenario_list = None
        self.settings = Settings()
        self.settings.initializeScenarioSettings(None)
        self.current_project_directory = os.getcwd()  # noqa: PTH109
        self.project_file_addr = None

        self.asli_app = QtWidgets.QApplication([])
        self.asli_MainWindow = QtWidgets.QMainWindow()

        self.setupUi(self.asli_MainWindow)

        Simulation_Tab_Designer.__init__(self)
        Hydraulic_Tab_Designer.__init__(self)
        Damage_Tab_Designer.__init__(self)
        Restoration_Tab_Designer.__init__(self)
        Run_Tab_Designer.__init__(self)
        PP_Data_Tab.__init__(self, self.project)
        Result_Designer.__init__(self)
        Map_Designer.__init__(self)

        """
        Action Triggers.
        """
        self.action_Open_Project.triggered.connect(self.openProject)
        self.action_Save.triggered.connect(self.saveProject)
        self.action_Save_Project_As.triggered.connect(self.saveProjectAs)
        self.action_REWET_GITHUB.triggered.connect(
            lambda: QDesktopServices.openUrl(
                QUrl('https://github.com/snaeimi/REWET')
            )
        )
        self.action_About.triggered.connect(self.showHelpWindow)
        self.action_Exit.triggered.connect(self.asli_MainWindow.close)

        """
        Native signal overwrite
        """
        self.asli_MainWindow.closeEvent = self.exitApp

    def run(self):  # noqa: ANN201, D102
        self.asli_MainWindow.show()
        sys.exit(self.asli_app.exec_())

    def errorMSG(self, error_title, error_msg, error_more_msg=None):  # noqa: ANN001, ANN201, N802, D102
        error_widget = QtWidgets.QMessageBox()
        error_widget.setIcon(QtWidgets.QMessageBox.Critical)
        error_widget.setText(error_msg)
        error_widget.setWindowTitle(error_title)
        error_widget.setStandardButtons(QtWidgets.QMessageBox.Ok)
        if error_more_msg != None:  # noqa: E711
            error_widget.setInformativeText(error_more_msg)
        error_widget.exec_()

    def questionPrompt(self, title, msg, more_msg=None):  # noqa: ANN001, ANN201, N802, D102
        prompt_widget = QtWidgets.QMessageBox()
        prompt_widget.setIcon(QtWidgets.QMessageBox.Question)
        prompt_widget.setText(msg)
        prompt_widget.setWindowTitle(title)
        prompt_widget.setStandardButtons(
            QtWidgets.QMessageBox.Yes
            | QtWidgets.QMessageBox.No
            | QtWidgets.QMessageBox.Cancel
        )
        if more_msg != None:  # noqa: E711
            prompt_widget.setInformativeText(more_msg)
        return prompt_widget.exec_()

    def openProject(self):  # noqa: ANN201, N802, D102
        file = QtWidgets.QFileDialog.getOpenFileName(
            self.asli_MainWindow,
            'Select project file',
            self.current_project_directory,
            'REWET Project File (*.prj)',
        )
        if file[0] == '':
            return
        split_addr = os.path.split(file[0])
        self.current_project_directory = split_addr

        self.project_file_addr = file[0]
        with open(file[0], 'rb') as f:  # noqa: PTH123
            project = pickle.load(f)  # noqa: S301
        self.project = project
        # sina put a possible check of result version here
        self.setSimulationSettings(project.project_settings)
        self.setHydraulicSettings(project.project_settings)
        self.setDamageSettings(project.project_settings, project.scenario_list)
        self.setRestorationSettings(project.project_settings)
        self.setSimulationUI()
        self.setHydraulicUI()
        self.setDamageUI()
        self.setRestorationUI()

    def saveProject(self, save_as=False):  # noqa: ANN001, ANN201, FBT002, N802, D102
        data_retrived = False
        if self.getSimulationSettings():  # noqa: SIM102
            if self.getHydraulicSettings():  # noqa: SIM102
                if self.getDamageSettings():  # noqa: SIM102
                    if self.getRestorationSettings():
                        data_retrived = True

        if data_retrived == False:  # noqa: E712
            return False

        if save_as == False:  # noqa: E712
            if self.project_file_addr == None:  # noqa: E711
                file_addr = QtWidgets.QFileDialog.getSaveFileName(
                    self.asli_MainWindow,
                    'Save project file',
                    self.project_file_addr,
                    'Project file (*.prj)',
                )
                if file_addr[0] == '':
                    return False
                split_addr = os.path.split(file_addr[0])
                self.current_project_directory = split_addr[0]
                self.project_file_addr = file_addr[0]

            project = Project(self.settings, self.scenario_list)
            self.project = project
            with open(self.project_file_addr, 'wb') as f:  # noqa: PTH123
                pickle.dump(project, f)

        return True

    def saveProjectAs(self):  # noqa: ANN201, N802, D102
        if_saved = self.saveProject(save_as=True)
        if if_saved == False:  # noqa: E712
            return

        file_addr = QtWidgets.QFileDialog.getSaveFileName(
            self.asli_MainWindow,
            'Save project file',
            self.project_file_addr,
            'Project file (*.prj)',
        )
        if file_addr[0] == '':
            return
        split_addr = os.path.split(file_addr[0])
        self.current_project_directory = split_addr[0]
        self.project_file_addr = file_addr[0]

        project = Project(self.settings, self.scenario_list)
        self.project = project
        with open(self.project_file_addr, 'wb') as f:  # noqa: PTH123
            pickle.dump(project, f)

    def showHelpWindow(self):  # noqa: ANN201, N802, D102
        help_window = Main_Help_Designer()
        help_window._window.exec_()  # noqa: SLF001

    def exitApp(self, event):  # noqa: ANN001, ANN201, N802, D102
        return_value = self.questionPrompt(
            'REWET', 'Do you want to save the project before you leave?'
        )

        if return_value == 16384:  # Yes  # noqa: PLR2004
            if_saved = self.saveProject()
            if if_saved:
                event.accept()
            else:
                event.ignore()
        elif return_value == 65536:  # None  # noqa: PLR2004
            event.accept()
        elif return_value == 4194304:  # Cancel  # noqa: PLR2004
            event.ignore()
            return


if __name__ == '__main__':
    opening_designer = Opening_Designer()
    opening_designer.run()
