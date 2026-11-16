file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES
    getBRAILSAttributes.py getBRAILSBaselineInv.py getBRAILSFootprints.py
    getBRAILSLocationBoundary.py runBrails.py runBrailsTransp.py
    DESTINATION ${_rel})
