# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 18:29:50 2022

@author: snaeimi
"""


from PyQt5 import QtWidgets, QtGui
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from GUI.Subsitute_Layer_Designer import Subsitute_Layer_Designer
from GUI.Symbology_Designer       import Symbology_Designer


single_scenario_map_options = ['', 'Quantity Return', 'Delivery Return', ]
multi_scenario_map_options = ['','Quantity Outage vs. Exceedance','Delivery Outage vs. Exceedance', 'Quantity Exceedance vs. Time', 'Delivery Exceedance vs. Time']
map_settings = {  'Quantity Return':[{"Label":"Time", "Type":"Time", "Default":"seconds"}, {"Label":"LDN leak", "Type":"Yes-No_Combo", "Default":"No"}, {"Label":"leak Criteria", "Type":"Float Line", "Default":"0.75"}, {"Label":"Time Window", "Type":"Int Line", "Default":"7200"}],
                  'Delivery Return':[{"Label":"Time", "Type":"Time", "Default":"seconds"}, {"Label":"LDN leak", "Type":"Yes-No_Combo", "Default":"No"}, {"Label":"leak Criteria", "Type":"Float Line", "Default":"1.25"}, {"Label":"Time Window", "Type":"Int Line", "Default":"7200"}],
                  'Quantity Outage vs. Exceedance':[{"Label":"Time", "Type":"Time", "Default":"seconds"}, {"Label":"LDN leak", "Type":"Yes-No_Combo", "Default":"No"}, {"Label":"leak Criteria", "Type":"Float Line", "Default":"1.25"}, {"Label":"Time Window", "Type":"Int Line", "Default":"7200"}, {"Label":"Ex. Prob.", "Type":"Float Line", "Default":"0.09", "Validator":{"Min":0, "Max":1}}],
                  'Delivery Outage vs. Exceedance':[{"Label":"Time", "Type":"Time", "Default":"seconds"}, {"Label":"LDN leak", "Type":"Yes-No_Combo", "Default":"No"}, {"Label":"leak Criteria", "Type":"Float Line", "Default":"1.25"}, {"Label":"Time Window", "Type":"Int Line", "Default":"7200"}, {"Label":"Ex. Prob.", "Type":"Int Line", "Default":str(24*3600), "Validator":{"Min":0, "Max":1000*24*3600}}],
                  'Quantity Exceedance vs. Time':[{"Label":"Time", "Type":"Time", "Default":"seconds"}, {"Label":"LDN leak", "Type":"Yes-No_Combo", "Default":"No"}, {"Label":"leak Criteria", "Type":"Float Line", "Default":"1.25"}, {"Label":"Time Window", "Type":"Int Line", "Default":"7200"}, {"Label":"Outage Time", "Type":"Float Line", "Default":"0.09", "Validator":{"Min":0, "Max":1}}],
                  'Delivery Exceedance vs. Time':[{"Label":"Time", "Type":"Time", "Default":"seconds"}, {"Label":"LDN leak", "Type":"Yes-No_Combo", "Default":"No"}, {"Label":"leak Criteria", "Type":"Float Line", "Default":"1.25"}, {"Label":"Time Window", "Type":"Int Line", "Default":"7200"}, {"Label":"Outage Time", "Type":"Int Line", "Default":str(24*3600), "Validator":{"Min":0, "Max":1000*24*3600}}]}
norm = plt.Normalize(1,4)
cmap = plt.cm.RdYlGn

class Time_Unit_Combo(QtWidgets.QComboBox):
    def __init__(self):
        super().__init__()
        time_units = ["second", "hour", "day"]

        self.addItems(time_units)
    
    def changeMapTimeUnit(self, raw_time_map, value_columns_name):
        
        time_justified_map = raw_time_map.copy()
        
        time_unit = self.currentText()
        data      = time_justified_map[value_columns_name]
        
        #time_justified_map = time_justified_map.reset_index()
        
        
        if time_unit == "second":
           return raw_time_map.copy()
        elif time_unit == "hour":
            data = data/3600
        elif time_unit == "day":
            data = data/3600/24
        else:
            raise ValueError("Unknown unit time: "+repr(time_unit) )
        
        for ind in data.index.to_list():
            time_justified_map.loc[ind, value_columns_name] = data.loc[ind]
        return time_justified_map

class Yes_No_Combo(QtWidgets.QComboBox):
    def __init__(self):
        super().__init__()
        self.addItems(["No", "Yes"])

class Map_Designer():
    def __init__(self):

        self.current_raw_map       = None
        self.current_map           = None
        self.annotation_map        = None
        self.plotted_map           = None   
        self.subsitute_layer_addr  = None
        self.subsitute_layer       = None
        self.iUse_substitute_layer = False
        self.map_settings_widgets  = {}
        self.symbology             = {"Method":"FisherJenks", "kw":{"k":5}, "Color":"Blues"}
        
        self.main_tab.currentChanged.connect(self.tabChangedMap)
        self.map_all_scenarios_checkbox.stateChanged.connect(self.mapAllScenarioCheckboxChanged)
        self.save_map_button.clicked.connect(self.saveCurrentMapByButton)
        self.map_scenario_combo.currentTextChanged.connect(self.resultScenarioChanged)
        self.map_type_combo.currentTextChanged.connect(self.mapTypeChanegd)
        self.annotation_checkbox.stateChanged.connect(self.AnnotationCheckboxChanged)
        self.annotation_event_combo.currentTextChanged.connect(self.getAnnotationtype)
        self.mpl_map.canvas.fig.canvas.mpl_connect("motion_notify_event", self.mouseHovered)
        self.mpl_map.canvas.fig.canvas.mpl_connect("button_press_event", self.mouseClicked)
                
        
        
        """
        Signals
        """
        self.annotation_radius_line.editingFinished.connect(self.annotationRadiusChanegd)
        self.spatial_join_button.clicked.connect(self.openSubsituteLayerWindow)
        self.major_tick_size_line.editingFinished.connect(self.majorTickSet)
        self.symbology_button.clicked.connect(self.symbologyByButton)
        
        """
        Validators
        """
        self.annotation_radius_line.setValidator(QtGui.QDoubleValidator(0, 1000000, 20, notation=QtGui.QDoubleValidator.StandardNotation) )
        self.major_tick_size_line.setValidator(QtGui.QIntValidator(0, 64) )
        
        self.map_value_columns_name = None
        self.anottation_type = "None"
        self.annotation_column = None
        
        self.initializeMap()
    
    def initializeMap(self):
        self.setMapAllScenarios(True)
        self.map_all_scenarios_checkbox.setChecked(True)
        self.map_scenario_combo.clear()
        self.map_scenario_combo.addItems(self.result_scenarios)
        #self.current_map_data = None
    
    def symbologyByButton(self):
        sym = Symbology_Designer(self.symbology, self.plotted_map, self.map_value_columns_name)
        val = sym._window.exec_()
        
        if val == 1:
            self.symbology = sym.sym
            self.plotMap(self.map_value_columns_name)
    
    def majorTickSet(self):
        major_tick_fond_size = self.major_tick_size_line.text()
        major_tick_fond_size = float(major_tick_fond_size)
               
        self.mpl_map.canvas.ax.tick_params(axis='both', which='major', labelsize=major_tick_fond_size)
        self.mpl_map.canvas.fig.canvas.draw_idle()
    
    def openSubsituteLayerWindow(self):
        demand_node_temporary_layer = self.project_result.createGeopandasPointDataFrameForNodes()
        sub_layer = Subsitute_Layer_Designer(self.subsitute_layer_addr, self.subsitute_layer, self.iUse_substitute_layer, demand_node_temporary_layer)
        val = sub_layer._window.exec_()

        if val == 1:
            self.subsitute_layer_addr  = sub_layer.subsitute_layer_addr
            self.subsitute_layer       = sub_layer.subsitute_layer
            self.iUse_substitute_layer = sub_layer.iUse_substitute_layer
            self.plotMap(self.map_value_columns_name)
    
    def annotationRadiusChanegd(self):
        annotation_radius = self.annotation_radius_line.text()
        self.annotation_map = self.plotted_map.copy(deep=True)
        if annotation_radius=="":
            annotation_radius = 0
            self.annotation_radius_line.settext("0")
        annotation_radius = float(annotation_radius)
        for ind, val in self.current_map.geometry.iteritems():
            self.annotation_map.geometry.loc[ind] = val.buffer(annotation_radius)
    
    def AnnotationCheckboxChanged(self, state):
        if state == 0:
            self.annotation_event_combo.setEnabled(False)
            self.annotation_radius_line.setEnabled(False)
            self.anottation_type = "None"
            self.annot.set_visible(False)
        elif state == 2:
            self.annotation_event_combo.setEnabled(True)
            self.annotation_radius_line.setEnabled(True)
            self.getAnnotationtype()
    
    def mapAllScenarioCheckboxChanged(self, state):
        if state == 0:
            self.setMapAllScenarios(False)
        elif state == 2:
            self.setMapAllScenarios(True)
    
    def getAnnotationtype(self, text=None):
        combo_value = self.annotation_event_combo.currentText()
        if combo_value == "Mouse hover":
            self.anottation_type = combo_value
        elif combo_value == "Mouse click":
            self.anottation_type = combo_value
        else:
            raise ValueError("unknown annotation type: "+repr(combo_value))
            
    def mouseHovered(self, event):
        if self.anottation_type != "Mouse hover":
            return
        
        if type(self.current_map) == type(None):
            return
        self.putAnnotation(event)
    
    def mouseClicked(self, event):
        if self.anottation_type != "Mouse click":
            return
        
        if type(self.current_map) == type(None):
            return
        
        if event.button != 1:
            return
        
        self.putAnnotation(event)
    
    def putAnnotation(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.mpl_map.canvas.ax:
            #print((event.xdata, event.ydata) )
            mouse_point = Point(event.xdata, event.ydata)

            s = self.annotation_map.geometry.contains(mouse_point)
            s_index_list = s[s==True].index
            
            if len(s_index_list) >= 1:
                cont = True
                s_index = s_index_list[0]
            elif len(s_index_list) == 0:
                cont = False
            
            if cont:
                #print(len(s_index_list))
                data = self.annotation_map.loc[s_index, self.map_value_columns_name]
                if type(data) == pd.core.series.Series:
                    data = data.iloc[0]
                text = repr(data)
                self.update_annot(text, event)
                self.annot.set_visible(True)
                self.mpl_map.canvas.fig.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.mpl_map.canvas.fig.canvas.draw_idle()
                    
    
    def update_annot(self, text, event):
        self.annot.xy = (event.xdata, event.ydata)

        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_facecolor(cmap(norm(1)))
        self.annot.get_bbox_patch().set_alpha(0.4)
    
    def clearMapPlot(self):
        self.mpl_map.canvas.ax.cla()
        
    def plotMap(self, value_columns_name):
        self.clearMapPlot()
        self.mpl_map.canvas.ax.clear()
        #for ind, val in self.current_map.geometry.iteritems():
            #self.current_map.geometry.loc[ind] = val.buffer(2000)
        #self.mpl_map.canvas.ax.clear()
        data = self.current_map
        #print(data.head() )
        
        self.annot = self.mpl_map.canvas.ax.annotate("", xy=(0, 0), xytext=(20, 20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)

        if self.iUse_substitute_layer == True:
            data = data.set_crs(crs=self.subsitute_layer.crs)
            joined_map = gpd.sjoin(self.subsitute_layer, data)
            #joined_map.plot(ax=self.mpl_map.canvas.ax, column=value_columns_name, cmap="Blues", legend=True)
            data = joined_map
        else:
            pass
        self.annotation_map = data.copy(deep=True)
        #data.to_file("Northridge/ss2.shp")
        self.plotted_map = self.prepareForLegend(data, value_columns_name)
        self.plotted_map.plot(ax=self.mpl_map.canvas.ax, column=value_columns_name, cmap=self.symbology["Color"], categorical=True, legend="True", scheme=self.symbology["Method"], classification_kwds=self.symbology["kw"])
        self.mpl_map.canvas.ax.ticklabel_format(axis='both', style='plain')
        #self.majorTickSet()
        
        #labels = self.mpl_map.canvas.ax.get_xticks()
        #self.mpl_map.canvas.ax.set_xticklabels(labels, rotation=45, ha='right')
        #self.mpl_map.canvas.ax.plot(self.current_map.index, self.current_map.to_list())  
        
        self.mpl_map.canvas.draw() 
        self.mpl_map.canvas.fig.tight_layout()
        
    def prepareForLegend(self, data, value_columns_name):
        return data.copy(deep=True)
        data = data.copy(deep=True)
        min_value = data[value_columns_name].min()
        max_value = data[value_columns_name].max()
        step = (max_value - min_value)/5

        
        step_array = np.arange(min_value, max_value, step)
        step_array = step_array.tolist()
        step_array.append(max_value)
        
        for i in range(len(step_array) - 1):
            step_max = step_array[i+1]
            step_min = step_array[i]
            index_list = data[(data[value_columns_name] < step_max) & (data[value_columns_name] > step_min)].index
            #print(index_list)
            for ind in index_list:
                data.loc[ind, value_columns_name]= step_max
        
        return data
        
        
    def setMapAllScenarios(self, flag):
        if flag == True:
            self.map_all_scenarios_checkbox.setChecked(True)
            self.map_scenario_combo.setEnabled(False)
            self.map_type_combo.clear()
            self.map_type_combo.addItems(multi_scenario_map_options)
            self.clearMapPlot()
        elif flag == False:
            self.map_all_scenarios_checkbox.setChecked(False)
            self.map_scenario_combo.setEnabled(True)
            self.map_type_combo.clear()
            self.map_type_combo.addItems(single_scenario_map_options)
            self.clearMapPlot()
        else:
            raise ValueError("Unknown flag: " + repr(flag))
    
    def resultScenarioChanged(self, text):
        self.map_result_current_scenario = text #self.map_scenario_combo.getText()
        
    def mapTypeChanegd(self, text):
        if self.project_result == None:
            return
        self.current_map_type = text 
        self.setMapSettingBox(text)
        self.calculateCurrentMap()
    
    def calculateCurrentMap(self):
        
        map_type = self.current_map_type
        if map_type == 'Quantity Outage vs. Exceedance':
           
           iConsider_leak          = self.map_settings_widgets["LDN leak"].currentText()
           leak_ratio              = self.map_settings_widgets["leak Criteria"].text()
           time_window             = self.map_settings_widgets["Time Window"].text()
           exeedance_probability   = self.map_settings_widgets["Ex. Prob."].text()
           
           if iConsider_leak == "Yes":
               iConsider_leak = True
           else:
               iConsider_leak = False
           
           leak_ratio            = float(leak_ratio) 
           time_window           = int(float(time_window) )
           exeedance_probability = float(exeedance_probability)
           
           self.map_value_columns_name = "res"
           map_data = self.project_result.AS_getOutage_4(LOS='QN', iConsider_leak=iConsider_leak, leak_ratio=leak_ratio, consistency_time_window=time_window)
           #print(map_data)
           self.current_raw_map  = self.project_result.getDLQNExceedenceProbabilityMap(map_data, ihour=True , param=exeedance_probability)
           #self.current_map      = self.current_raw_map.copy()
           self.current_map      = self.time_combo.changeMapTimeUnit(self.current_raw_map, self.map_value_columns_name)
           
           #print(exeedance_probability)
           self.plotMap(self.map_value_columns_name)
       
        elif map_type == 'Delivery Outage vs. Exceedance':
            
           iConsider_leak          = self.map_settings_widgets["LDN leak"].currentText()
           leak_ratio              = self.map_settings_widgets["leak Criteria"].text()
           time_window             = self.map_settings_widgets["Time Window"].text()
           exeedance_probability   = self.map_settings_widgets["Ex. Prob."].text()
           
           if iConsider_leak == "Yes":
               iConsider_leak = True
           else:
               iConsider_leak = False
           
           leak_ratio            = float(leak_ratio) 
           time_window           = int(float(time_window) )
           exeedance_probability = float(exeedance_probability)
           
           self.map_value_columns_name = "res"
           map_data = self.project_result.AS_getOutage_4(LOS='DL', iConsider_leak=iConsider_leak, leak_ratio=leak_ratio, consistency_time_window=time_window)
           #print(map_data)
           self.current_raw_map  = self.project_result.getDLQNExceedenceProbabilityMap(map_data, ihour=True , param=exeedance_probability)
           #self.current_map      = self.current_raw_map.copy()
           self.current_map      = self.time_combo.changeMapTimeUnit(self.current_raw_map, self.map_value_columns_name)
           
           #print(exeedance_probability)
           self.plotMap(self.map_value_columns_name)
        
        elif map_type == 'Quantity Exceedance vs. Time':
           
           iConsider_leak          = self.map_settings_widgets["LDN leak"].currentText()
           leak_ratio              = self.map_settings_widgets["leak Criteria"].text()
           time_window             = self.map_settings_widgets["Time Window"].text()
           outage_time             = self.map_settings_widgets["Outage Time"].text()
           
           if iConsider_leak == "Yes":
               iConsider_leak = True
           else:
               iConsider_leak = False
           
           leak_ratio            = float(leak_ratio) 
           time_window           = int(float(time_window) )
           outage_time           = int(float(outage_time) )
           
           self.map_value_columns_name = "res"
           map_data = self.project_result.AS_getOutage_4(LOS='QN', iConsider_leak=iConsider_leak, leak_ratio=leak_ratio, consistency_time_window=time_window)
           #print(map_data)
           self.current_raw_map  = self.project_result.getDLQNExceedenceProbabilityMap(map_data, ihour=False , param=outage_time)
           #self.current_map      = self.current_raw_map.copy()
           self.current_map      = self.time_combo.changeMapTimeUnit(self.current_raw_map, self.map_value_columns_name)
           
           #print(exeedance_probability)
           self.plotMap(self.map_value_columns_name)
       
        elif map_type == 'Delivery Exceedance vs. Time':
            
           iConsider_leak          = self.map_settings_widgets["LDN leak"].currentText()
           leak_ratio              = self.map_settings_widgets["leak Criteria"].text()
           time_window             = self.map_settings_widgets["Time Window"].text()
           outage_time             = self.map_settings_widgets["Outage Time"].text()
           
           if iConsider_leak == "Yes":
               iConsider_leak = True
           else:
               iConsider_leak = False
           
           leak_ratio            = float(leak_ratio) 
           time_window           = int(float(time_window) )
           outage_time           = int(float(outage_time) )
           
           self.map_value_columns_name = "res"
           map_data = self.project_result.AS_getOutage_4(LOS='DL', iConsider_leak=iConsider_leak, leak_ratio=leak_ratio, consistency_time_window=time_window)
           #print(map_data)
           self.current_raw_map  = self.project_result.getDLQNExceedenceProbabilityMap(map_data, ihour=False , param=outage_time)
           #self.current_map      = self.current_raw_map.copy()
           self.current_map      = self.time_combo.changeMapTimeUnit(self.current_raw_map, self.map_value_columns_name)
           
           #print(exeedance_probability)
           self.plotMap(self.map_value_columns_name)
        
        elif map_type == 'Quantity Return':
            iConsider_leak          = self.map_settings_widgets["LDN leak"].currentText()
            leak_ratio              = self.map_settings_widgets["leak Criteria"].text()
            time_window             = self.map_settings_widgets["Time Window"].text()
            
            if iConsider_leak == "Yes":
                iConsider_leak = True
            else:
                iConsider_leak = False
            
            leak_ratio  = float(leak_ratio) 
            time_window = int(float(time_window) )
            
            scn_name                = self.map_scenario_combo.currentText()
            self.current_raw_map    = self.project_result.getOutageTimeGeoPandas_4(scn_name, LOS='QN', iConsider_leak=iConsider_leak, leak_ratio=leak_ratio, consistency_time_window=time_window)
            value_column_label      = "restoration_time"
            self.current_map        = self.time_combo.changeMapTimeUnit(self.current_raw_map, value_column_label)
            self.plotMap(value_column_label)
            
            self.map_value_columns_name = value_column_label
            
        elif map_type == 'Delivery Return':
            iConsider_leak          = self.map_settings_widgets["LDN leak"].currentText()
            leak_ratio              = self.map_settings_widgets["leak Criteria"].text()
            time_window             = self.map_settings_widgets["Time Window"].text()
            
            if iConsider_leak == "Yes":
                iConsider_leak = True
            else:
                iConsider_leak = False
            
            leak_ratio  = float(leak_ratio) 
            time_window = int(float(time_window) )
            
            scn_name                = self.map_scenario_combo.currentText()
            self.current_raw_map    = self.project_result.getOutageTimeGeoPandas_4(scn_name, LOS='DL', iConsider_leak=iConsider_leak, leak_ratio=leak_ratio, consistency_time_window=time_window)
            value_column_label      = "restoration_time"
            self.current_map        = self.time_combo.changeMapTimeUnit(self.current_raw_map, value_column_label)
            self.plotMap(value_column_label)
            
            self.map_value_columns_name = value_column_label
            
        elif map_type == 'SSI':
            return
            #self.current_map_data = (map_type, pd.DataFrame())
            iPopulation             = self.map_settings_widgets["Population"].currentText()
            scn_name                = self.map_scenario_combo.currentText()
            self.current_raw_map  = self.project_result.getSystemServiceabilityIndexMap(scn_name, iPopulation=iPopulation)
            self.current_map      = self.time_combo.changeMapTimeUnit(self.current_raw_map)
            self.plotMap("SSI", "Time")
        elif map_type == '':
            return
        else:
            raise
        
        #self.annotation_map = self.current_raw_map.copy()
        self.annotationRadiusChanegd()
            
        
    def setMapSettingBox(self, map_type):
        for i in range(self.map_settings_table.rowCount()):
            self.map_settings_table.removeRow(0)
        
        if map_type in map_settings:
            self.populateMapSettingsTable(map_settings[map_type] )
        else:
            pass
            #raise ValueError("Unknown Map type: "+repr(map_type))
    
    def populateMapSettingsTable(self, settings_content):
        self.map_settings_widgets.clear()
        vertical_header = []
        cell_type_list  = []
        default_list    = []
        content_list    = []
        validator_list  = []
        for row in settings_content:
            for k in row:
                if k == "Label":
                    vertical_header.append(row[k])
                elif k == "Type":
                    cell_type_list.append(row[k])
                elif k == "Default":
                    default_list.append(row[k])
                
            if "Content" in row:
                content_list.append(row["Content" ])
            else:
                content_list.append(None)
            
            if "Validator" in row:
                validator_list.append(row["Validator" ])
            else:
                validator_list.append(None)
    
        self.map_settings_table.setColumnCount(1 )
        self.map_settings_table.setRowCount(len(settings_content))
        self.map_settings_table.setVerticalHeaderLabels(vertical_header)
        
        i = 0
        for cell_type in cell_type_list:
            if cell_type=="Time":
                self.time_combo = Time_Unit_Combo()
                self.map_settings_table.setCellWidget(i,0, self.time_combo)
                self.time_combo.currentTextChanged.connect(self.mapTimeSettingsChanged )
            
            elif cell_type=="Yes-No_Combo":
                current_widget = Yes_No_Combo()
                self.map_settings_table.setCellWidget(i,0, current_widget)
                current_widget.currentTextChanged.connect(self.mapSettingChanged )
                
                default_value = default_list[i]
                current_widget.setCurrentText(default_value)
                
                self.map_settings_widgets[vertical_header[i]] = current_widget
            
            elif cell_type=="Custom_Combo":
                current_widget = QtWidgets.QComboBox()
                contents       = content_list[i]
                current_widget.addItems(contents)
                self.map_settings_table.setCellWidget(i,0, current_widget)
                current_widget.currentTextChanged.connect(self.mapSettingChanged )
                
                default_value = default_list[i]
                current_widget.setCurrentText(default_value)
                
                self.map_settings_widgets[vertical_header[i]] = current_widget
            
            elif cell_type=="Float Line":
                current_widget = QtWidgets.QLineEdit()
                self.map_settings_table.setCellWidget(i,0, current_widget)
                current_widget.editingFinished.connect(self.mapSettingChanged )
                if validator_list[i] == None:
                    current_widget.setValidator(QtGui.QDoubleValidator(0, 1000000, 20, notation=QtGui.QDoubleValidator.StandardNotation) )
                else:
                    current_widget.setValidator(QtGui.QDoubleValidator(validator_list[i]["Min"], validator_list[i]["Max"], 20, notation=QtGui.QDoubleValidator.StandardNotation) )
                    
                default_value = default_list[i]
                current_widget.setText(default_value)
                self.map_settings_widgets[vertical_header[i]] = current_widget
            
            elif cell_type=="Int Line":
                current_widget = QtWidgets.QLineEdit()
                self.map_settings_table.setCellWidget(i,0, current_widget)
                current_widget.editingFinished.connect(self.mapSettingChanged )
                
                if validator_list[i] == None:
                    current_widget.setValidator(QtGui.QIntValidator(0, 3600*24*1000) )
                else:
                    current_widget.setValidator(QtGui.QIntValidator(validator_list[i]["Min"], validator_list[i]["Max"]) )
                    
                default_value = default_list[i]
                current_widget.setText(default_value)
                self.map_settings_widgets[vertical_header[i]] = current_widget
            else:
                raise ValueError(repr(cell_type) )
                
            i += 1
        #for label in settings_content:
    
    def mapTimeSettingsChanged(self, x):
        self.current_map = self.time_combo.changeMapTimeUnit(self.current_raw_map, self.map_value_columns_name)
        self.plotMap(self.map_value_columns_name)
    
    def mapSettingChanged(self):
        if "Population" in self.map_settings_widgets:
            new_population_setting = self.map_settings_widgets["Population"].currentText()
            if new_population_setting == "Yes" and type(self.project_result._population_data) == type(None):
                self.errorMSG("Error", "Population data is not loaded")
                self.map_settings_widgets["Population"].setCurrentText("No")
                return
        self.calculateCurrentMap()
    def tabChangedMap(self, index):
        if index == 1:
            self.initializeMap()
    
    def saveCurrentMapByButton(self):
        #if self.current_map_data == None:
        if type(self.current_map) == type(None):
            self.errorMSG("REWET", 'No map is ploted')
            return
        
        file_addr = QtWidgets.QFileDialog.getSaveFileName(self.asli_MainWindow, 'Save File', 
                                                     self.project_file_addr,"Shapefile (*.shp)")
        if file_addr[0] == '':
            return
        
        #self.current_map_data[1].to_excel(file_addr[0])
        self.current_map.to_file(file_addr[0])
        