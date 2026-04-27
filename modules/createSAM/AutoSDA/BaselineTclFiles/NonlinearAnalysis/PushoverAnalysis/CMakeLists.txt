file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES
    DefineVariables.tcl Database.csv Model.tcl DefineAllRecorders.tcl
    PerformGravityAnalysis.tcl DefineFunctionsAndProcedures.tcl RunStaticPushover.tcl
    DESTINATION ${_rel})
