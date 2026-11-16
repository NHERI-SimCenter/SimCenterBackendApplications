file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES censusutil.py globals.py DESTINATION ${_rel})
install(FILES logging.ini DESTINATION ${_rel})
