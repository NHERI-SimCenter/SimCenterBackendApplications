add_executable(simCenterSub simCenterSub.cpp)
target_include_directories(simCenterSub PRIVATE ${CMAKE_SOURCE_DIR}/modules)
file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS simCenterSub DESTINATION ${_rel})
