add_executable(OpenSeesPostprocessor OpenSeesPostprocessor.cpp)
target_include_directories(OpenSeesPostprocessor PRIVATE ${CMAKE_SOURCE_DIR}/modules)
target_link_libraries(OpenSeesPostprocessor PRIVATE jansson::jansson common)
file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS OpenSeesPostprocessor DESTINATION ${_rel})

add_executable(OpenSeesPreprocessor OpenSeesPreprocessor.cpp)
target_include_directories(OpenSeesPreprocessor PRIVATE ${CMAKE_SOURCE_DIR}/modules)
target_link_libraries(OpenSeesPreprocessor PRIVATE jansson::jansson common)
install(TARGETS OpenSeesPreprocessor DESTINATION ${_rel})

install(FILES OpenSeesSimulation.py DESTINATION ${_rel})
