file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES opensha-all.jar USGS_HazardCurveConfig.json DESTINATION ${_rel})
