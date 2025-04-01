simcenter_add_executable(NAME parseSGMDtoCSV
                         FILES parseSGMDtoCSV.cpp
                         DEPENDS CONAN_PKG::jansson)


simcenter_add_executable(NAME parseSGMDtoJSON
                         FILES parseSGMDtoJSON.cpp
                         DEPENDS CONAN_PKG::jansson)

simcenter_add_executable(NAME parseSGMDtoJSON_PGA
                         FILES parseSGMDtoCSV_PGA.cpp
                         DEPENDS CONAN_PKG::jansson)

set_property(TARGET parseSGMDtoCSV PROPERTY CXX_STANDARD 17)
set_property(TARGET parseSGMDtoJSON PROPERTY CXX_STANDARD 17)
