# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:37:52 2024

@author: naeim
"""

import os
pid = os.fork()
if pid > 0:
     
    print("\nIn parent process")
    info = os.waitpid(pid, 0)
    if os.WIFEXITED(info[1]) :
        code = os.WEXITSTATUS(info[1])
        print("Child's exit code:", code)
     
else :
    print("In child process")
    print("Process ID:", os.getpid())
    print("Hello ! Geeks")
    print("Child exiting..")
        
    os._exit(os.EX_OK)