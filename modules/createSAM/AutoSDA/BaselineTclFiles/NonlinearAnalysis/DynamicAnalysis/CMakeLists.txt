file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES
    DynamicAnalysisCollapseSolver.tcl Database.csv MaxDriftTester.tcl
    DefineAllRecorders.tcl Model.tcl DefineFunctionsAndProcedures.tcl
    PerformGravityAnalysis.tcl DefineTimeHistory.tcl RunIDA2DModel.tcl DefineVariables.tcl
    DESTINATION ${_rel})
