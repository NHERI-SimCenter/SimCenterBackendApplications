simcenter_add_executable(NAME HighRiseTPU
                         FILES HighRiseTPU.cpp callHighRiseTPU.cpp
                         DEPENDS CONAN_PKG::jansson CONAN_PKG::libcurl common)

simcenter_add_python_script(SCRIPT HighRiseTPU.py)
