#simcenter_add_executable(StochasticGM
#                         command_parser.cpp eq_generator.cpp main.cpp)

add_executable(StochasticGM command_parser.cpp eq_generator.cpp main.cpp)

include_directories(../common/smelt)
target_link_libraries (StochasticGM CONAN_PKG::kissfft smelt)

set_property(TARGET StochasticGM PROPERTY CXX_STANDARD 17)

install(TARGETS StochasticGM DESTINATION ${PROJECT_SOURCE_DIR}/applications/createEVENT/stochasticGroundMotion)