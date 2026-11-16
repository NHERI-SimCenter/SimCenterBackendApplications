file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES ERADist.py ERANataf.py ERACond.py ERARosen.py DESTINATION ${_rel})
