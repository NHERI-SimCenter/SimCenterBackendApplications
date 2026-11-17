file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES damage_convertor.py preprocessorIO.py REWET_Wrapper.py DESTINATION ${_rel})
