simcenter_add_executable(NAME OpenSeesPostprocessor
                         FILES OpenSeesPostprocessor.cpp
                         DEPENDS CONAN_PKG::jansson common)
simcenter_add_executable(NAME OpenSeesPreprocessor
                         FILES OpenSeesPreprocessor.cpp
                         DEPENDS CONAN_PKG::jansson common)
# std::filesystem requires C++17 (simcenter_add_executable defaults to 14)
set_property(TARGET OpenSeesPreprocessor PROPERTY CXX_STANDARD 17)
simcenter_add_python_script(SCRIPT OpenSeesSimulation.py)
