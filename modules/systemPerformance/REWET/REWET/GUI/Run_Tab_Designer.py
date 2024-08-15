"""Created on Wed Nov  2 14:40:45 2022

@author: snaeimi
"""

import subprocess
import threading

from PyQt5.QtCore import QObject, pyqtSignal


class Custom_Object(QObject):
    outSignal = pyqtSignal(bytes)


class Run_Tab_Designer:
    def __init__(self):
        self.run_button.clicked.connect(self.runREWET)
        self.stop_button.clicked.connect(self.stopRun)
        self.cobject = Custom_Object()
        self.cobject.outSignal.connect(self.updateRunOuput)
        self.rewet_sub_process = None
        self.if_run_in_progress = False

    def runREWET(self):
        if self.if_run_in_progress == True:
            return False
        if_saved = self.saveProject()

        if if_saved == False:
            return False
        self.ouput_textedit.clear()
        # start = Starter()
        if self.project_file_addr == None:
            self.errorMSG(
                'REWET',
                'File address is empty. Please report it as a bug to the developer.',
            )
        self.if_run_in_progress = True
        self.setAllTabsEnabled(False)
        threading.Thread(target=self._RunREWETHelper, args=(), daemon=True).start()

    def _RunREWETHelper(self):
        self.rewet_sub_process = subprocess.Popen(
            ['python', 'initial.py', self.project_file_addr],
            stdout=subprocess.PIPE,
            bufsize=0,
        )

        for line in iter(self.rewet_sub_process.stdout.readline, b''):
            # sys.stdout.flush()
            self.cobject.outSignal.emit(line)
        self.rewet_sub_process.stdout.close()

    def setAllTabsEnabled(self, enabled):
        # self.ouput_textedit.setEnabled(enabled)
        self.main_tab.setTabEnabled(1, enabled)
        self.main_process1.setTabEnabled(0, enabled)
        self.main_process1.setTabEnabled(1, enabled)
        self.main_process1.setTabEnabled(2, enabled)
        self.main_process1.setTabEnabled(3, enabled)
        self.run_button.setEnabled(enabled)
        # self.results_tabs_widget.setEnabled(enabled)
        # self.stop_button.setEnabled(True)

    # @pyqtSlot(bytes)
    def updateRunOuput(self, string):
        string = string.decode()

        if 'Time of Single run is' in string:
            self.endSimulation()
        elif 'Error' in string:
            self.errorInSimulation()

        self.ouput_textedit.appendPlainText(string)

        # running code for the project

    def endSimulation(self):
        end_message = (
            '\n-------------------\nSIMULATION FINISHED\n-------------------\n'
        )
        self.setAllTabsEnabled(True)
        self.if_run_in_progress = False
        self.ouput_textedit.appendPlainText(end_message)

    def errorInSimulation(self):
        end_message = '\n-------------\nERROR OCCURRED\n-------------\n'
        self.setAllTabsEnabled(True)
        self.if_run_in_progress = False
        self.errorMSG(
            'REWET',
            'An error happened during the simulation. Please look at the log for further information.',
        )
        self.ouput_textedit.appendPlainText(end_message)

    def stopRun(self):
        if self.if_run_in_progress == False:
            return
        if type(self.rewet_sub_process) != type(None):
            self.rewet_sub_process.terminate()
            termination_message = '\n-------------\nRUN CANCELLED\n-------------\n'
            self.setAllTabsEnabled(True)
            self.if_run_in_progress = False
            self.ouput_textedit.appendPlainText(termination_message)
