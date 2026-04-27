file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES MPM.py post_process_sensors.py setup_case.py DESTINATION ${_rel})
