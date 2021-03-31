simcenter_add_executable(NAME WindFieldSimulation
                         FILES WindFieldModel.cpp
                         DEPENDS CONAN_PKG::jansson CONAN_PKG::Eigen3)
simcenter_add_python_script(SCRIPT ComputeIntensityMeasure.py)
simcenter_add_python_script(SCRIPT CreateScenario.py)
simcenter_add_python_script(SCRIPT CreateStation.py)
simcenter_add_python_script(SCRIPT HurricaneSimulation.py)
simcenter_add_python_script(SCRIPT WindFieldSimulation.py)
add_subdirectory(database)
