simcenter_add_executable(NAME extractEDP
                         FILES extractEDP.cpp
                         DEPENDS CONAN_PKG::jansson)
simcenter_add_executable(NAME postprocessDakota
                         FILES postprocessDakota.cpp
                         DEPENDS CONAN_PKG::jansson)
