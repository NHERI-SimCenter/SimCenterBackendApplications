file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES pelicun3_wrapper.py HDF_to_CSV.py DL_visuals.py DESTINATION ${_rel})
