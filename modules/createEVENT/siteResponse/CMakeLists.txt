simcenter_add_executable(NAME SiteResponse
                         FILES SiteResponse.cpp)

simcenter_add_python_script(SCRIPT SiteResponse.py)

simcenter_add_python_script(SCRIPT RegionalSiteResponse.py)

simcenter_add_python_script(SCRIPT postProcess.py)
