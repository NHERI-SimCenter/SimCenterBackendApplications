simcenter_add_python_script(SCRIPT SimCenterUQ.py)
simcenter_add_python_script(SCRIPT surrogateBuild.py)
simcenter_add_python_script(SCRIPT UQengine.py)
simcenter_add_python_script(SCRIPT runPLoM.py)
add_subdirectory(PLoM)
add_subdirectory(nataf_gsa)

