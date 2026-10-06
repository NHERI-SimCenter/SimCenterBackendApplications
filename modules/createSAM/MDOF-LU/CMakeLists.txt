add_subdirectory(data)
simcenter_add_executable(NAME MDOF-LU
                         FILES MDOF-LU.cpp InterstoryParam.cpp HazusSAM_Generator.cpp FloorParam.cpp Building.cpp whereami.c
                         DEPENDS CONAN_PKG::jansson common)
# std::filesystem requires C++17 (overrides the C++14 set by simcenter_add_executable)
set_property(TARGET MDOF-LU PROPERTY CXX_STANDARD 17)
