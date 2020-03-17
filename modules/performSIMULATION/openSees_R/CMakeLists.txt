simcenter_add_executable(NAME mainPostprocessor
                         FILES mainPostprocessor.cpp OpenSeesPostprocessor.cpp
                         DEPENDS CONAN_PKG::jansson)
simcenter_add_executable(NAME mainPreprocessor
                         FILES mainPreprocessor.cpp OpenSeesPreprocessor.cpp OpenSeesConcreteShearWalls.cpp
                         DEPENDS CONAN_PKG::jansson)
simcenter_add_executable(NAME getUncertainty
                         FILES getUncertainty.cpp OpenSeesPreprocessor.cpp OpenSeesConcreteShearWalls.cpp
                         DEPENDS CONAN_PKG::jansson)
simcenter_add_python_script(SCRIPT OpenSeesSimulation.py)
