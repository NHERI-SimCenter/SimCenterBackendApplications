file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES
    UFileTemplate kFileTemplate nutFileTemplate pFileTemplate epsilonFileTemplate
    physicalPropertiesTemplate transportPropertiesTemplate momentumTransportTemplate DFSRTurbDictTemplate
    controlDictTemplate blockMeshDictTemplate snappyHexMeshDictTemplate decomposeParDictTemplate
    surfaceFeaturesDictTemplate
    fvSchemesTemplateLES fvSchemesTemplateDES fvSchemesTemplateRANS fvSolutionTemplate
    probeTemplate storyForcesTemplate baseForcesTemplate vtkPlaneTemplate
    DESTINATION ${_rel})
