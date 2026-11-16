file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES
    EigenValueAnalysis.tcl Database.csv Model.tcl
    DefineFunctionsAndProcedures.tcl PerformGravityAnalysis.tcl DefineVariables.tcl
    DESTINATION ${_rel})
