simcenter_add_executable(NAME MDOF_BuildingModel
                         FILES MDOF_BuildingModelToSAM.cpp
                         DEPENDS CONAN_PKG::jansson common)
