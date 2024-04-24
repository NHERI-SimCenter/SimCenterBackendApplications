# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:23:09 2024

@author: snaeimi

This is the main final to run REWET. This file superscedes intital.py to run
REWET. In order to keep the backward compatibility, the initial.py is kepy,
so one can run initial.py to run REWET. currently, REWET's GUI still works with
initial.py. Main.py is going to be the most developed tool.

"""

import sys
import os
import argparse
from initial import Starter

if __name__ == "__main__":
    argParser = argparse.ArgumentParser(prog="REWET V0.2",
        description="REstoration tool for Restoration of Water after Event Tool is a package for modeling damages and restoration in water network. You can specify settings in with providing a JSON. An exampel JSON file is provided in example folder. Modify the exampel folder and provide its path as an input. If not provided, the default settings valeus from the input/settings.py will be ran. thus, you can alterbatively modify values in settings for a single run."
        )
    
    argParser.add_argument("--json", "-j",  default=None, 
                           help="json settings file")
    
    argParser.add_argument("-p", default=None, 
                           help="REWET project file")
    
    parse_namespace = argParser.parse_args()
    
    starter = Starter()
    # No file is pecified, so the default values in settinsg file is going to
    # be ran.
    
    if parse_namespace.json == None and parse_namespace.p == None: 
        starter.run()
        sys.exit(0)
    elif parse_namespace.json != None and parse_namespace.p == None:
        if parse_namespace.json.split(".")[-1].upper() != "JSON":
            print("ERROR in json file name: ", parse_namespace.json,
                  "The json file must have json extention")
            sys.exit(0)
        elif not os.path.exists(parse_namespace.json):
            print("ERROR in json file: ", parse_namespace.json,
                  "does not exist")
        else:
            starter.run(parse_namespace.json)
            
    elif parse_namespace.json == None and parse_namespace.p != None:
        if parse_namespace.p.split(".")[-1].upper() != "PRJ":
            print("ERROR in project file name: ", parse_namespace.json,
                  "The project file must have PRJ extention")
            sys.exit(0)
        elif not os.path.exists(parse_namespace.p):
            print("ERROR in project file: ", parse_namespace.p,
                  "does not exist")
        else:
            starter.run(parse_namespace.json)
    
    else:
        print("ERROR in arguments\n",
              "Either of the json or project file arguments must be used")
   
else:
    print("Main File has been ran with not being the main module (i.e.,\
          __name__ is not \"__main__\"")