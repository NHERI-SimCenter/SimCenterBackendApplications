simcenter_add_executable(NAME MDOF_BldgMdl
                         FILES MDOF_BuildingModelToSAM.cpp
                         DEPENDS CONAN_PKG::jansson common)
