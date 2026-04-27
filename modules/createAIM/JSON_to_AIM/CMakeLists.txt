file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES JSON_to_AIM_transport.py GeoJSON_to_AIM_transport.py DESTINATION ${_rel})
