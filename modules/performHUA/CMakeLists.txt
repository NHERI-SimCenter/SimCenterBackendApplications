# performHUA has no C++ source files; it only installs Python scripts
add_subdirectory(pyincore_data)

file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES INCORECensusUtil.py DESTINATION ${_rel})
