simcenter_add_executable(NAME WindFieldSimulation
			 DEPENDS CONAN_PKG::jansson
			 DEPENDS CONAN_PKG::eigen			 
                         FILES WindFieldModel.cpp)
simcenter_add_python_script(SCRIPT ComputeIntensityMeasure.py)
simcenter_add_python_script(SCRIPT CreateScenario.py)
simcenter_add_python_script(SCRIPT CreateStation.py)
simcenter_add_python_script(SCRIPT HurricaneSimulation.py)
simcenter_add_python_script(SCRIPT WindFieldSimulation.py)
add_subdirectory(database)
