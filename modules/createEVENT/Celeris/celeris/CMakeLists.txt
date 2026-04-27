file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES domain.py runner.py solver.py utils.py DESTINATION ${_rel})
