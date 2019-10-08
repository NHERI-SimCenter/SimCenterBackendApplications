simcenter_add_executable(NAME WindTunnelExperiment
                         FILES WindTunnelExperiment.cpp
                         DEPENDS CONAN_PKG::jansson common)
