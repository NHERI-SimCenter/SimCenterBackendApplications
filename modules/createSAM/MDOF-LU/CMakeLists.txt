add_subdirectory(data)

add_executable(MDOF-LU MDOF-LU.cpp InterstoryParam.cpp HazusSAM_Generator.cpp FloorParam.cpp Building.cpp whereami.c)
target_include_directories(MDOF-LU PRIVATE ${CMAKE_SOURCE_DIR}/modules)
target_link_libraries(MDOF-LU PRIVATE jansson::jansson common)
file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS MDOF-LU DESTINATION ${_rel})
