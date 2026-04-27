file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES
    UCSD_UQ.py runFEM.py parseData.py pdfs.py runTMCMC.py mainscript.py
    tmcmcFunctions.py defaultLogLikeScript.py processInputs.py calibration_utilities.py
    mainscript_tmcmc.py mainscript_hierarchical_bayesian.py
    preprocess_hierarchical_bayesian.py mwg_sampler.py loglike_script.py
    DESTINATION ${_rel})
