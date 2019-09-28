simcenter_add_executable(NAME ExtractPGA
                         FILES ExtractPGA.cpp
                         DEPENDS CONAN_PKG::jansson common)
