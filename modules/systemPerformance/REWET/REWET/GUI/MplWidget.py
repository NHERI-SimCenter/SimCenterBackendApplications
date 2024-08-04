"""Created on Thu Nov 10 18:26:02 2022

@author: snaeimi
"""  # noqa: CPY001, D400, N999

# Imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PyQt5 import QtWidgets


class MplCanvas(Canvas):  # noqa: D101
    def __init__(self):  # noqa: ANN204
        self.fig = Figure(figsize=(100, 40), dpi=100, tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(
            self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        Canvas.updateGeometry(self)


# Matplotlib widget
class MplWidget(QtWidgets.QWidget):  # noqa: D101
    def __init__(self, parent=None):  # noqa: ANN001, ANN204
        QtWidgets.QWidget.__init__(self, parent)  # Inherit from QWidget
        self.canvas = MplCanvas()  # Create canvas object
        toolbar = NavigationToolbar(self.canvas, self)
        self.vbl = QtWidgets.QVBoxLayout()  # Set box for plotting
        self.vbl.addWidget(toolbar)
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)
