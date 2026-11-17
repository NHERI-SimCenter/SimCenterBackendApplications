file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES preprocessUQpy.py UQpyEngine.py DESTINATION ${_rel})

add_subdirectory(src)
