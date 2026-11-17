add_executable(mainPostprocessor mainPostprocessor.cpp OpenSeesPostprocessor.cpp)
target_include_directories(mainPostprocessor PRIVATE ${CMAKE_SOURCE_DIR}/modules)
target_link_libraries(mainPostprocessor PRIVATE jansson::jansson)
file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS mainPostprocessor DESTINATION ${_rel})

add_executable(mainPreprocessor mainPreprocessor.cpp OpenSeesPreprocessor.cpp OpenSeesConcreteShearWalls.cpp)
target_include_directories(mainPreprocessor PRIVATE ${CMAKE_SOURCE_DIR}/modules)
target_link_libraries(mainPreprocessor PRIVATE jansson::jansson)
install(TARGETS mainPreprocessor DESTINATION ${_rel})

add_executable(getUncertainty getUncertainty.cpp OpenSeesPreprocessor.cpp OpenSeesConcreteShearWalls.cpp)
target_include_directories(getUncertainty PRIVATE ${CMAKE_SOURCE_DIR}/modules)
target_link_libraries(getUncertainty PRIVATE jansson::jansson)
install(TARGETS getUncertainty DESTINATION ${_rel})

install(FILES OpenSeesSimulation.py DESTINATION ${_rel})
