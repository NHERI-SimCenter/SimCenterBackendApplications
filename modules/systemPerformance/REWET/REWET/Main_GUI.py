# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 21:46:04 2022

@author: snaeimi
"""
import os
import sys
if __name__ == "__main__":
    from GUI.Opening_Designer import Project
    from GUI.Opening_Designer import Opening_Designer
    opening_designer = Opening_Designer()
    print(os.getpid())
    sys.exit(opening_designer.run() )