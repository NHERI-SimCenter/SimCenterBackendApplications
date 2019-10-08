simcenter_add_executable(NAME OpenSeesPostprocessor
                         FILES OpenSeesPostprocessor.cpp
                         DEPENDS CONAN_PKG::jansson)
simcenter_add_executable(NAME OpenSeesPreprocessor
                         FILES OpenSeesPreprocessor.cpp
                         DEPENDS CONAN_PKG::jansson common)
simcenter_add_python_script(SCRIPT OpenSeesSimulation.py)
