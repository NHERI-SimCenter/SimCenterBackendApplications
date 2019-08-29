simcenter_add_executable(NAME DEDM_HRP
                         FILES DEDM_HRP.cpp dedmCall.cpp
                         DEPENDS CONAN_PKG::jansson CONAN_PKG::libcurl common)

simcenter_add_python_script(SCRIPT DEDM_HRP.py)
