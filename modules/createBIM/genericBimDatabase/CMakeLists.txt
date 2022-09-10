simcenter_add_executable(NAME GenericBimDatabase
                         FILES GenericBimDatabase.cpp BIMConfig.cpp BIM.cpp csvparser.c
                         DEPENDS CONAN_PKG::jansson)
