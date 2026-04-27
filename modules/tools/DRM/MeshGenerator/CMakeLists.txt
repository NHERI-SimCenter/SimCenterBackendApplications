add_subdirectory(models)

file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES Infowriter.py MeshGenrator.py Structure.py DESTINATION ${_rel})
