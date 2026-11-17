add_subdirectory(whale)

file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES
    femUQ.py changeJSON.py sWHALE.py qWHALE.py rWHALE.py
    createResponseCSV.py siteResponseWHALE.py createGM4BIM.py
    computeResponseSpectrum.py RegionalEarthquakeSimulation.py WorkflowUtils.py
    MultiModelApplication.py MultiModelDriver.py
    SiteResponse_workflow.py R2DTool_workflow.py CreateWorkflowJobs.py
    DESTINATION ${_rel})

install(FILES WorkflowApplications.json DESTINATION ${_rel})

add_executable(runMultiModelDriver runMultiModelDriver.cpp)
target_include_directories(runMultiModelDriver PRIVATE ${CMAKE_SOURCE_DIR}/modules)
install(TARGETS runMultiModelDriver DESTINATION ${_rel})
