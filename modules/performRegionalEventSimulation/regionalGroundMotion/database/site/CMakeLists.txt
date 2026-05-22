file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})

install(FILES
	global_vs30_4km.tar.gz
	global_zTR_4km.tar.gz
        DESTINATION ${_rel})
