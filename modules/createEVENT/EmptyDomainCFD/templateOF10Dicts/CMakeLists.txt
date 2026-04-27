file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES
    UFileTemplate kFileTemplate nutFileTemplate pFileTemplate epsilonFileTemplate
    physicalPropertiesTemplate transportPropertiesTemplate momentumTransportTemplate DFSRTurbDictTemplate
    controlDictTemplate blockMeshDictTemplate snappyHexMeshDictTemplate decomposeParDictTemplate
    fvSchemesTemplateLES fvSchemesTemplateDES fvSchemesTemplateRANS fvSolutionTemplate
    probeTemplate vtkPlaneTemplate
    DESTINATION ${_rel})
