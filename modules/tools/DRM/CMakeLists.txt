add_subdirectory(MeshGenerator)
add_subdirectory(StructresFiles)

file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES modelCreator.py DESTINATION ${_rel})
