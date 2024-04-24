# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 18:06:01 2022

@author: snaeimi
"""
import sys
import os
from PyQt5 import QtWidgets
from PyQt5.Qt import QUrl, QDesktopServices
import pickle

from Project                   import Project
from .Opening_Window           import Ui_Opening_Window
from .Simulation_Tab_Designer  import Simulation_Tab_Designer
from .Hydraulic_Tab_Designer   import Hydraulic_Tab_Designer
from .Damage_Tab_Designer      import Damage_Tab_Designer
from .Restoration_Tab_Designer import Restoration_Tab_Designer
from .Run_Tab_Designer         import Run_Tab_Designer
from .Main_Help_Designer       import Main_Help_Designer
from .PP_Data_Tab_Designer     import PP_Data_Tab
from .Result_Designer          import Result_Designer
from .Map_Designer             import Map_Designer

from Input.Settings        import Settings

class Opening_Designer(Ui_Opening_Window, Simulation_Tab_Designer, Hydraulic_Tab_Designer, Damage_Tab_Designer, Run_Tab_Designer, Restoration_Tab_Designer, PP_Data_Tab, Result_Designer, Map_Designer):
    def __init__(self):
        self.project = None
        self.scenario_list = None
        self.settings = Settings()
        self.settings.initializeScenarioSettings(None)
        self.current_project_directory = os.getcwd()
        self.project_file_addr         = None
        
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
        self.action_REWET_GITHUB.triggered.connect(lambda : QDesktopServices.openUrl(QUrl("https://github.com/snaeimi/REWET")))
        self.action_About.triggered.connect(self.showHelpWindow)
        self.action_Exit.triggered.connect(self.asli_MainWindow.close)
        
        """
        Native signal overwrite
        """
        self.asli_MainWindow.closeEvent = self.exitApp
        
    def run(self):
        self.asli_MainWindow.show()
        sys.exit(self.asli_app.exec_())

    def errorMSG(self, error_title, error_msg, error_more_msg=None):
        error_widget = QtWidgets.QMessageBox()
        error_widget.setIcon(QtWidgets.QMessageBox.Critical)
        error_widget.setText(error_msg)
        error_widget.setWindowTitle(error_title)
        error_widget.setStandardButtons(QtWidgets.QMessageBox.Ok)
        if error_more_msg!=None:
            error_widget.setInformativeText(error_more_msg)
        error_widget.exec_()
    
    def questionPrompt(self, title, msg, more_msg=None):
        prompt_widget = QtWidgets.QMessageBox()
        prompt_widget.setIcon(QtWidgets.QMessageBox.Question)
        prompt_widget.setText(msg)
        prompt_widget.setWindowTitle(title)
        prompt_widget.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
        if more_msg!=None:
            prompt_widget.setInformativeText(more_msg)
        return prompt_widget.exec_()
    
    def openProject(self):
        file = QtWidgets.QFileDialog.getOpenFileName(self.asli_MainWindow, 'Select project file', 
         self.current_project_directory, "REWET Project File (*.prj)")
        if file[0] == '':
            return
        split_addr = os.path.split(file[0])
        self.current_project_directory = split_addr
        
        self.project_file_addr = file[0]
        with open(file[0], 'rb') as f:
            project = pickle.load(f)
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
    
    def saveProject(self, save_as=False):
        
        data_retrived = False
        if self.getSimulationSettings():
            if self.getHydraulicSettings():
                if self.getDamageSettings():
                    if self.getRestorationSettings():
                        data_retrived = True
        
        if data_retrived == False:
            return False
        
        if save_as == False:
            if self.project_file_addr == None:
                file_addr = QtWidgets.QFileDialog.getSaveFileName(self.asli_MainWindow, 'Save project file', 
                                                         self.project_file_addr,"Project file (*.prj)")
                if file_addr[0] == '':
                    return False
                split_addr = os.path.split(file_addr[0])
                self.current_project_directory = split_addr[0]
                self.project_file_addr = file_addr[0]
            
            project = Project(self.settings, self.scenario_list)
            self.project = project
            with open(self.project_file_addr, 'wb') as f:
                pickle.dump(project, f)
            
        return True
    
    def saveProjectAs(self):
        if_saved = self.saveProject(save_as=True)
        if if_saved == False:
            return
        
        file_addr = QtWidgets.QFileDialog.getSaveFileName(self.asli_MainWindow, 'Save project file', 
                                                     self.project_file_addr,"Project file (*.prj)")
        if file_addr[0] == '':
            return
        split_addr = os.path.split(file_addr[0])
        self.current_project_directory = split_addr[0]
        self.project_file_addr = file_addr[0]
        
        project = Project(self.settings, self.scenario_list)
        self.project = project
        with open(self.project_file_addr, 'wb') as f:
            pickle.dump(project, f)
    
    def showHelpWindow(self):
        help_window = Main_Help_Designer()
        help_window._window.exec_()
    
    def exitApp(self, event):
        return_value = self.questionPrompt("REWET", "Do you want to save the project before you leave?")

        if return_value == 16384: #Yes
            if_saved = self.saveProject()
            if if_saved:
                event.accept()
            else:
                event.ignore()
        elif return_value == 65536: #None
            event.accept()
        elif return_value == 4194304: #Cancel
            event.ignore()
            return
        
        
    
if __name__ == "__main__":
    opening_designer = Opening_Designer()
    opening_designer.run()