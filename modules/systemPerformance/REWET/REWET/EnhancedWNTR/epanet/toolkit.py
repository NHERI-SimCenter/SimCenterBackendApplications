# -*- coding: utf-8 -*-
"""
Created on Wed May 26 16:11:36 2021

@author: snaeimi
"""
import wntr.epanet.toolkit
import numpy as np
import ctypes
import os, sys
from pkg_resources import resource_filename
import platform

import logging
logger = logging.getLogger(__name__)

class EpanetException(Exception):
    pass

class ENepanet(wntr.epanet.toolkit.ENepanet):
    def __init__(self, inpfile='', rptfile='', binfile='', changed_epanet=False, version=2.2):
        if changed_epanet==False or changed_epanet==True:
            self.changed_epanet=changed_epanet
        else:
            raise ValueError("changed_epanet must be a boolean value")
            
        if changed_epanet==False:
            super().__init__(inpfile, rptfile, binfile, version=version)
        else:
            try:
                super().__init__(inpfile, rptfile, binfile, version=version)
            except:
                pass # to add robustness for the time when for the WNTR
                     #cannot load the umodified DLLs for any reason
                 
            if float(version) != 2.2:
                raise ValueError("EPANET version must be 2.2 when using tegh changed version")
        
            elif float(version) == 2.2:
                libnames = ["epanet22_mod", "epanet22_win32_mod"]
                if "64" in platform.machine():
                    libnames.insert(0, "epanet22_amd64_mod")
            for lib in libnames:
                try:
                    if os.name in ["nt", "dos"]:
                        libepanet = resource_filename(
                            __name__, "Windows/%s.dll" % lib
                        )
                        self.ENlib = ctypes.windll.LoadLibrary(libepanet)
                    elif sys.platform in ["darwin"]:
                        libepanet = resource_filename(
                            __name__, "Darwin/lib%s.dylib" % lib
                        )
                        self.ENlib = ctypes.cdll.LoadLibrary(libepanet)
                    else:
                        libepanet = resource_filename(
                            __name__, "Linux/lib%s.so" % lib
                        )
                        self.ENlib = ctypes.cdll.LoadLibrary(libepanet)
                    return
                except Exception as E1:
                    if lib == libnames[-1]:
                        raise E1
                    pass
                finally:
                    if version >= 2.2 and '32' not in lib:
                        self._project = ctypes.c_uint64()
                    elif version >= 2.2:
                        self._project = ctypes.c_uint32()
                    else:
                        self._project = None                

    
    def ENSetIgnoreFlag(self, ignore_flag=0):
        if abs(ignore_flag - np.round(ignore_flag))>0.00001 or ignore_flag<0:
            logger.error('ignore_flag must be int value and bigger than zero'+str(ignore_flag))
        flag=ctypes.c_int(int(ignore_flag))
        #print('++++++++++++++++++++++')
        #self.ENlib.ENEXTENDEDsetignoreflag(flag)