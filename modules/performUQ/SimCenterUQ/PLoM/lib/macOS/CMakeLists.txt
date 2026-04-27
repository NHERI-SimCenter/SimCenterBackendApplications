file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES PLoM_C_library.o PLoM_C_library.so DESTINATION ${_rel})
