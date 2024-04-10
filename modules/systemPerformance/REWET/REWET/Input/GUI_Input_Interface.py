# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:04:40 2022

This is the input module, an interface between all inputs from GUI, TEXT-based
inputs and the mail code.

@author: snaeimi
"""

class input():
    def __init__(self, settings, registry):
        pass
    
    def convertShiftFromDictToPandasTable(self, dict_data):
        shift_name_list      = list(shift_data )
        shift_begining_list  = [shift_data[i][0] for i in shift_name_list]
        shift_end_list       = [shift_data[i][1] for i in shift_name_list]
