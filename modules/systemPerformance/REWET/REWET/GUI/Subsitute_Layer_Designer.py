# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 16:31:32 2023

@author: snaeimi
"""

import shapely
import os
import pandas as pd
import geopandas as gpd
from PyQt5 import QtWidgets
from GUI.Subsitute_Layer_Window import Ui_subsitite_layer_dialoge


class Subsitute_Layer_Designer(Ui_subsitite_layer_dialoge):
    def __init__(self, subsitute_layer_addr, subsitute_layer,iUse_substitute_layer,  demand_node_layers):
        super().__init__()
        self._window = QtWidgets.QDialog()
        self.setupUi(self._window)
        
        self.current_substitute_directory = ''
        self.subsitute_layer       = subsitute_layer
        self.old_subsitute_layer   = subsitute_layer
        self.subsitute_layer_addr  = subsitute_layer_addr
        self.demand_node_layers    = demand_node_layers
        self.iUse_substitute_layer = iUse_substitute_layer
        self.demand_node_layers.to_file("Northridge\demand_node_layer.shp")
        self.subsitute_layer_addr_line.setText(self.subsitute_layer_addr )
        if type(self.subsitute_layer) != type(None):
            self.subsitute_layer_projection_name_line.setText(self.subsitute_layer.crs.name )
        
        apply_button = self.Subsitute_buttonBox.button(QtWidgets.QDialogButtonBox.Apply)
        apply_button.clicked.connect(self.applyNewSubsituteLayer)
        ok_button = self.Subsitute_buttonBox.button(QtWidgets.QDialogButtonBox.Ok)
        ok_button.clicked.connect(self.applyNewSubsituteLayer)
        
        self.population_browser_button.clicked.connect(self.substituteLayerBrowseButton)
        
        self.iUse_sub_checkbox.setChecked(self.iUse_substitute_layer)
        self.iUse_sub_checkbox.stateChanged.connect(self.iUseSubstituteCheckBoxStateChanged)
    
    def iUseSubstituteCheckBoxStateChanged(self, state):
        if state == 0:
            self.iUse_substitute_layer = False
        elif state == 2:
            self.iUse_substitute_layer = True
            
    def applyNewSubsituteLayer(self):
        #demand_node_layers = self.createGeopandasPointDataFrameForNodes(self, self.wn, self.demand_node_name)
        if type(self.subsitute_layer) == type(None):
            return
    
    def substituteLayerBrowseButton(self):
        file = QtWidgets.QFileDialog.getOpenFileName(self._window, 'Open file', 
         self.current_substitute_directory,"Shapefile file (*.shp)")
        
        if file[0] == '':
            return
        split_addr                        = os.path.split(file[0])
        self.current_substitute_directory = split_addr[0]
        
        self.subsitute_layer_addr_line.setText(file[0])

        self.subsitute_layer      = gpd.read_file(file[0])
        self.subsitute_layer_addr = file[0]
        self.subsitute_layer_addr_line.setText(file[0])
        self.subsitute_layer_projection_name_line.setText(self.subsitute_layer.crs.name)
        
        self.sub_error_text_edit.clear()
        self.demand_node_layers = self.demand_node_layers.set_crs(crs=self.subsitute_layer.crs)
        joined_map = gpd.sjoin(self.subsitute_layer, self.demand_node_layers)
        
        number_list = pd.Series(index=self.demand_node_layers.index, data=0)
        for ind, val in joined_map["index_right"].iteritems():
            number_list.loc[val] = number_list.loc[val] + 1
        
        number_list = number_list[number_list > 1]
        number_list = number_list.sort_values(ascending=False)
        
        text = ""
        if len(number_list) > 0:
            text += "The following nodes is joined with more than 1 substitute layer feature\n"
        
        for ind, num in number_list.iteritems():
            text+=repr(ind) + " : " + repr(num) + "\n"
        
        text += "\n\n"
        
        index_number_list = pd.Series(index=self.subsitute_layer.index.unique(), data=0)
        for ind in joined_map.index.to_list():
            index_number_list.loc[ind] = index_number_list.loc[ind] + 1
        
        index_number_list = index_number_list[index_number_list > 1]
        index_number_list = index_number_list.sort_values(ascending=False)
        
        if len(index_number_list) > 0:
            text += "The following substitute layer feature have multiple nodes\n"
        i=1
        for ind, num in index_number_list.iteritems():
            st = self.subsitute_layer.loc[ind]
            st = st.drop("geometry")
            text += repr(st) + " : "+repr(num) + "\n"
            text += "---------- "+ repr(i)+" ----------"
            i+=1
        self.sub_error_text_edit.setText(text)
        
        
        
        
    