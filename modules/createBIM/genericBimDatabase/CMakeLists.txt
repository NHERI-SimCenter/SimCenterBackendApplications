simcenter_add_executable(NAME GenericBimDatabase
                         FILES GenericBimDatabase.cpp BimConfig.cpp BIM.cpp csvparser.c
                         DEPENDS CONAN_PKG::jansson)
