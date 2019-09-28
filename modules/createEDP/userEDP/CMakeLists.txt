simcenter_add_executable(NAME UserDefinedEDP
                         FILES UserDefinedEDP.cpp
                         DEPENDS CONAN_PKG::jansson)
