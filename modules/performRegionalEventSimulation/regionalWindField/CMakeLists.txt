add_executable(WindFieldSimulation WindFieldModel.cpp)
target_include_directories(WindFieldSimulation PRIVATE ${CMAKE_SOURCE_DIR}/modules)
target_link_libraries(WindFieldSimulation PRIVATE jansson::jansson Eigen3::Eigen)
file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS WindFieldSimulation DESTINATION ${_rel})
install(FILES
    ComputeIntensityMeasure.py CreateScenario.py CreateStation.py
    HurricaneSimulation.py WindFieldSimulation.py
    DESTINATION ${_rel})

add_subdirectory(database)
