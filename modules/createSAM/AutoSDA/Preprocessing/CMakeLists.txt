file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES BeamDatabase1.csv BeamDatabase2.csv CleanBeamSectionDatabase.ipynb BeamDatabase3.csv DESTINATION ${_rel})
