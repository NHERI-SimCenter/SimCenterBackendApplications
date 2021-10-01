simcenter_add_executable(NAME SiteResponse
                         FILES SiteResponse.cpp)

simcenter_add_python_script(SCRIPT SiteResponse.py)

simcenter_add_python_script(SCRIPT calibration.py)

simcenter_add_python_script(SCRIPT Gauss1D.py)

simcenter_add_python_script(SCRIPT RegionalSiteResponse.py)

simcenter_add_python_script(SCRIPT postProcess.py)

simcenter_add_python_script(SCRIPT createGM4BIM.py)

simcenter_add_python_script(SCRIPT postProcessRegional.py)
