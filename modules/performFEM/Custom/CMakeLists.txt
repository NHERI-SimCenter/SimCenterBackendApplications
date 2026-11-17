add_executable(createCustomDriver createCustomDriver.cpp)
target_include_directories(createCustomDriver PRIVATE ${CMAKE_SOURCE_DIR}/modules)
file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS createCustomDriver DESTINATION ${_rel})
