file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES
    UQpyRunner.py prepareUQ.py runWorkflowDriver.py configureAndRunUQ.py
    processUQpyOutput.py uqRunner.py createTemplate.py runOtherUQ.py HeirBayesRunner.py
    DESTINATION ${_rel})
