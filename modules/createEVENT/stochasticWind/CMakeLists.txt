#simcenter_add_executable(NAME StochasticWind
# DEPENDS CONAN_PKG::kissfft smelt common)

add_executable(StochasticWind command_parser.cpp floor_forces.cpp wind_generator.cpp main.cpp)

include_directories(../common/smelt)
target_link_libraries (StochasticWind CONAN_PKG::kissfft smelt common)

set_property(TARGET StochasticWind PROPERTY CXX_STANDARD 17)


install(TARGETS StochasticWind DESTINATION ${PROJECT_SOURCE_DIR}/applications/createEVENT/stochasticWind)