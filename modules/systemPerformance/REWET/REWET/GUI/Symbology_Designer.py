# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 00:08:01 2023

@author: snaeimi
"""
import sys
from PyQt5 import QtCore, QtWidgets
from GUI.Symbology_Window import Ui_Symbology_Dialog
import pandas      as pd
import geopandas   as gpd
import mapclassify
import matplotlib.pylab as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar



class Symbology_Designer(Ui_Symbology_Dialog):
    def __init__(self, sym, data, val_column):
        super().__init__()
        self._window = QtWidgets.QDialog()
        self.setupUi(self._window)
        self.sym = sym
        self.val_column = val_column
        self.plotted_map = data
        self.data = data[val_column]
        self.current_item_value = None
        self.fig, self.ax1 = plt.subplots()
        self.legend_widget = FigureCanvas(self.fig)
        lay = QtWidgets.QVBoxLayout(self.sample_legend_widget)  
        lay.setContentsMargins(0, 0, 0, 0)      
        lay.addWidget(self.legend_widget)
        self.initializeForm()
        
        self.method_combo.currentTextChanged.connect(self.methodChanged)
        self.range_table.currentItemChanged.connect(self.currentItemChanged)
        self.range_table.itemChanged.connect(self.tableDataChanged)
        #self.range_table.currentItemChanged.connect(self.currentItemChanged)
        self.remove_button.clicked.connect(self.removeButtonClicked)
        self.color_combo.currentTextChanged.connect(self.colorChanged)
        self.no_clases_line.editingFinished.connect(self.numberOfClassEditingFinished)
        self.add_up_button.clicked.connect(lambda : self.addByButton("UP"))
        self.add_below_button.clicked.connect(lambda : self.addByButton("DOWN"))
        
        self.sample_legend_widget

    def initializeForm(self):
        self.method_combo.setCurrentText(self.sym["Method"])
        if self.sym["Method"] == "FisherJenks" or self.sym["Method"] == "EqualInterval":
            self.no_clases_line.setText(str(self.sym["kw"]["k"]))
        else:
            self.no_clases_line.clear()
            self.no_clases_line.setEnabled(False)
        self.updateTable()
        #self.updateLegendSample()
    
    def addByButton(self, add_location):
        to_be_added_row = None
        selected_item_list = self.range_table.selectedItems()
        if len(selected_item_list) == 0:
            return
        else:
            selected_row = selected_item_list[0].row()
        
        if add_location == "UP":
            old_value = float(self.range_table.item(selected_row, 0).text() )
            if selected_row == 0:
                other_side_value = float(self.range_table.item(selected_row, 1).text() )
            else:
                other_side_value = float(self.range_table.item(selected_row-1, 0).text() )
        elif add_location == "DOWN":
            old_value = float(self.range_table.item(selected_row, 1).text() )
            if selected_row == (self.range_table.rowCount()-1):
                other_side_value = float(self.range_table.item(selected_row, 1).text() )
            else:
                other_side_value = float(self.range_table.item(selected_row+1, 1).text() )
        
        bins = self.class_data.bins.tolist()
        #bins[bins==old_value] = new_item_value
        bins.append((old_value+other_side_value)/2)
        bins.sort()
        kw = {"bins":bins}
        self.bins = bins
        self.sym["kw"] = kw
        
        if self.sym["Method"] != "UserDefined":
            self.method_combo.blockSignals(True)
            self.sym["Method"] = "UserDefined"
            self.no_clases_line.setEnabled(False)
            self.method_combo.setCurrentText("User Defined")
            self.method_combo.blockSignals(False)
        self.updateTable()
        
    def numberOfClassEditingFinished(self):
        k = float(self.no_clases_line.text())
        k = int(k)
        kw = {'k':k}
        self.sym["kw"] = kw
        self.updateTable()
    
    def colorChanged(self, text):
        self.sym["Color"] = text
        self.updateLegendSample()
     
    def updateLegendSample(self):
        fig, ax = plt.subplots()
        self.plotted_map.plot(ax=ax, cax=self.ax1,  column=self.val_column, cmap=self.sym["Color"], legend=True)
        self.legend_widget.draw() 
        #self.mpl_map.canvas.fig.tight_layout()
    
    def updateTable(self):
        self.range_table.blockSignals(True)
        self.clearRangeTable()
        if self.sym["Method"] == "FisherJenks":
            self.class_data = mapclassify.FisherJenks(self.data, self.sym["kw"]["k"])
        elif self.sym["Method"] == "EqualInterval":
            self.class_data = mapclassify.EqualInterval(self.data, self.sym["kw"]["k"])
        elif self.sym["Method"] == "UserDefined":
            self.class_data = mapclassify.UserDefined(self.data, self.sym["kw"]["bins"])
        else:
            raise ValueError("Unknown symbology method: "+repr(self.sym["Method"]) )
        min_val = self.data.min()
        max_val = self.data.max()
        
        bins = [min_val]
        bins.extend(self.class_data.bins.tolist())
        bins.append(max_val)
        
        bins = pd.Series(bins)
        bins = bins.unique()
        bins = bins.tolist()
        
        for i in range(len(bins)-1 ):
            number_of_rows = self.range_table.rowCount()
            self.range_table.insertRow(number_of_rows)
            beg_item   = QtWidgets.QTableWidgetItem(str(bins[i])              )
            end_item   = QtWidgets.QTableWidgetItem(str(bins[i+1])            )
            count_item = QtWidgets.QTableWidgetItem(str(self.class_data.counts[i]) )
            
            
            if i == 0:
                beg_item.setFlags(QtCore.Qt.NoItemFlags)
            
            if i == len(bins)-2:
                end_item.setFlags(QtCore.Qt.NoItemFlags)
            
            count_item.setFlags(QtCore.Qt.NoItemFlags)
            
            self.range_table.setItem(number_of_rows, 0, beg_item              )
            self.range_table.setItem(number_of_rows, 1, end_item              )
            self.range_table.setItem(number_of_rows, 2, count_item            )
        
        self.range_table.blockSignals(False)
        self.updateLegendSample()
    
    def clearRangeTable(self):
        for i in range(self.range_table.rowCount()):
            self.range_table.removeRow(0)
    
    def methodChanged(self, text):
        print(text)
        if text == "FisherJenks":
            self.sym["Method"] = "FisherJenks"
        elif text == "Equal Interval":
            self.sym["Method"] = "EqualInterval"
        elif text == "User Defined":
            self.sym["Method"] = "UserDefined"
        
        if text == "FisherJenks" or text == "Equal Interval":
            k = float(self.no_clases_line.text())
            k = int(k)
            kw = {'k':k}
        elif text == "User Defined":
            self.no_clases_line.setEnabled(False)
            #bins = self.getUserDefinedBins()
            try:
                kw = {'bins':self.bins}
            except:
                kw = {'bins':self.class_data}
        else:
            raise
        
        self.sym["kw"] = kw
        self.updateTable()
        
    def currentItemChanged(self, current, previous):
        if current != None:
            self.current_item_value = float(current.text())
        print("cur "+repr(self.current_item_value) )
        
    def tableDataChanged(self, item):
        #row = item.row()
        #col = item.column()
        
        #item_text = self.range_table.item(row, col).text()
        previous_item_value = float(self.current_item_value)
        try:
            new_item_value      = float(item.text() )
            if new_item_value < self.data.min() or new_item_value > self.data.max():
                raise
        except:
            self.range_table.item(item.row(),item.column() ).setText(str(previous_item_value) )
            return
            
        
        
        bins = self.class_data.bins
        bins[bins==previous_item_value] = new_item_value
        bins.sort()
        kw = {"bins":bins}
        self.bins = bins
        self.sym["kw"] = kw
        
        if self.sym["Method"] != "UserDefined":
            self.method_combo.blockSignals(True)
            self.sym["Method"] = "UserDefined"
            self.no_clases_line.setEnabled(False)
            self.method_combo.setCurrentText("User Defined")
            self.method_combo.blockSignals(False)
        self.updateTable()
        
        return
        
        
    def findBeginingRowFor(self, value):
        if self.range_table.rowCount() == 0:
            raise
            
        for i in range(self.range_table.rowCount() -1):
            current_item_value = float(self.range_table.item(i, 0).text() )
            next_item_value    = float(self.range_table.item(i+1, 0).text() )
            if value >= current_item_value and next_item_value < current_item_value:
                return i
        return self.range_table.rowCount()-1
    
    def findEndingRowFor(self, value):
        if self.range_table.rowCount() == 0:
            raise
            
        for i in range(self.range_table.rowCount() -1):
            current_item_value = float(self.range_table.item(i, 1).text() )
            next_item_value    = float(self.range_table.item(i+1, 1).text() )
            if value > current_item_value and next_item_value >= current_item_value:
                return i+1
        return self.range_table.rowCount()-1
        
    def removeButtonClicked(self):
        selected_item_list = self.range_table.selectedItems()
        if len(selected_item_list) == 0:
            return
        selected_row = selected_item_list[0].row()
        self.removeRow(selected_row)
        

    def removeRow(self, row):
        if row == 0 and self.range_table.rowCount() >= 2:
            item_text = self.range_table.item(row, 0).text()
            self.range_table.removeRow(0)
            self.range_table.item(0,0).setText(item_text)
        elif row == self.range_table.rowCount()-1 and self.range_table.rowCount() >= 2:
            item_text = self.range_table.item(row, 1).text()
            self.range_table.removeRow(row)
            self.range_table.item(row-1,1).setText(item_text)
        elif self.range_table.rowCount() == 1:
            self.range_table.removeRow(0)
        else:
            beg_text = self.range_table.item(row, 0).text()
            end_text = self.range_table.item(row, 1).text()
            self.range_table.removeRow(row)
            self.range_table.item(row-1,1).setText(beg_text)
            self.range_table.item(row,0).setText(end_text)
        
    
        
if __name__ == "__main__":
    symbology             = {"Method":"FisherJenks", "kw":{"k":5}}
    s  = gpd.read_file("ss2.shp" )
    print(s.columns)
    app = QtWidgets.QApplication(sys.argv)
    ss = Symbology_Designer(symbology, s["restoratio"])
    ss._window.show()
    sys.exit(app.exec_())
