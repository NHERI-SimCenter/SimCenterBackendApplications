file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES __init__.py general.py PLoM.py PLoM_library.py DESTINATION ${_rel})

add_subdirectory(lib)
