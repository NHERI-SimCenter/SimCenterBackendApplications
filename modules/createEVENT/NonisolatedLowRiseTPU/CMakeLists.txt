simcenter_add_executable(NAME NonIsolatedLowRiseTPU
                         FILES NonIsolatedLowRiseTPU.cpp callNonIsolatedLowRiseTPU.cpp
                         DEPENDS CONAN_PKG::jansson CONAN_PKG::libcurl common)

simcenter_add_python_script(SCRIPT NonIsolatedLowRiseTPU.py)
