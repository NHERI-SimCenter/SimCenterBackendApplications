simcenter_add_executable(NAME LowRiseTPU
                         FILES LowRiseTPU.cpp callTPU.cpp
                         DEPENDS CONAN_PKG::jansson CONAN_PKG::libcurl common)

simcenter_add_python_script(SCRIPT LowRiseTPU.py)
