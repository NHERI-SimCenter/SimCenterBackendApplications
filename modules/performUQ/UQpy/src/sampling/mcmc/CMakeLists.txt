file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES StretchDto.py ModifiedMetropolisHastingsDto.py DESTINATION ${_rel})
