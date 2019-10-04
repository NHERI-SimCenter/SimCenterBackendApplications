simcenter_add_executable(NAME WindTunnelExp
                         FILES WindTunnelExperiment.cpp
                         DEPENDS CONAN_PKG::jansson common)
