"""Created on Wed Nov  2 14:40:45 2022

@author: snaeimi
"""  # noqa: N999, D400, D415

import subprocess
import threading

from PyQt5.QtCore import QObject, pyqtSignal


class Custom_Object(QObject):  # noqa: N801, D101
    outSignal = pyqtSignal(bytes)  # noqa: N815


class Run_Tab_Designer:  # noqa: N801, D101
    def __init__(self):  # noqa: ANN204, D107
        self.run_button.clicked.connect(self.runREWET)
        self.stop_button.clicked.connect(self.stopRun)
        self.cobject = Custom_Object()
        self.cobject.outSignal.connect(self.updateRunOuput)
        self.rewet_sub_process = None
        self.if_run_in_progress = False

    def runREWET(self):  # noqa: ANN201, N802, D102
        if self.if_run_in_progress == True:  # noqa: E712
            return False
        if_saved = self.saveProject()

        if if_saved == False:  # noqa: E712
            return False
        self.ouput_textedit.clear()
        # start = Starter()
        if self.project_file_addr == None:  # noqa: E711
            self.errorMSG(
                'REWET',
                'File address is empty. Please report it as a bug to the developer.',
            )
        self.if_run_in_progress = True
        self.setAllTabsEnabled(False)
        threading.Thread(target=self._RunREWETHelper, args=(), daemon=True).start()  # noqa: RET503

    def _RunREWETHelper(self):  # noqa: ANN202, N802
        self.rewet_sub_process = subprocess.Popen(  # noqa: S603
            ['python', 'initial.py', self.project_file_addr],  # noqa: S607
            stdout=subprocess.PIPE,
            bufsize=0,
        )

        for line in iter(self.rewet_sub_process.stdout.readline, b''):
            # sys.stdout.flush()
            self.cobject.outSignal.emit(line)
        self.rewet_sub_process.stdout.close()

    def setAllTabsEnabled(self, enabled):  # noqa: ANN001, ANN201, N802, D102
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
    def updateRunOuput(self, string):  # noqa: ANN001, ANN201, N802, D102
        string = string.decode()

        if 'Time of Single run is' in string:
            self.endSimulation()
        elif 'Error' in string:
            self.errorInSimulation()

        self.ouput_textedit.appendPlainText(string)

        # running code for the project

    def endSimulation(self):  # noqa: ANN201, N802, D102
        end_message = (
            '\n-------------------\nSIMULATION FINISHED\n-------------------\n'
        )
        self.setAllTabsEnabled(True)
        self.if_run_in_progress = False
        self.ouput_textedit.appendPlainText(end_message)

    def errorInSimulation(self):  # noqa: ANN201, N802, D102
        end_message = '\n-------------\nERROR OCCURRED\n-------------\n'
        self.setAllTabsEnabled(True)
        self.if_run_in_progress = False
        self.errorMSG(
            'REWET',
            'An error happened during the simulation. Please look at the log for further information.',
        )
        self.ouput_textedit.appendPlainText(end_message)

    def stopRun(self):  # noqa: ANN201, N802, D102
        if self.if_run_in_progress == False:  # noqa: E712
            return
        if type(self.rewet_sub_process) != type(None):  # noqa: E721
            self.rewet_sub_process.terminate()
            termination_message = '\n-------------\nRUN CANCELLED\n-------------\n'
            self.setAllTabsEnabled(True)
            self.if_run_in_progress = False
            self.ouput_textedit.appendPlainText(termination_message)
