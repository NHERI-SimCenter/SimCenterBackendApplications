add_subdirectory(data)

file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES __init__.py CorrelationModel.py SignificantDurationModel.py openSHAGMPE.py DESTINATION ${_rel})
