file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES DRM_Sanramon.json Single_Sanramon.json DESTINATION ${_rel})
