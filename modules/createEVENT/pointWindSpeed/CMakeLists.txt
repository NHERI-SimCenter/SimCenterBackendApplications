simcenter_add_executable(NAME pointWindSpeed
                         FILES WindSpeedFromCloudData.cpp
                         DEPENDS CONAN_PKG::jansson CONAN_PKG::nanoflann)

simcenter_add_python_script(SCRIPT parseHurricaneScenario.py)
