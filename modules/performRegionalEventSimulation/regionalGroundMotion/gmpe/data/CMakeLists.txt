file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES
    loth_baker_correlation_2013_B1.csv loth_baker_correlation_2013_B2.csv loth_baker_correlation_2013_B3.csv
    markhvida_ceferino_baker_correlation_2017_model_coeff.csv
    markhvida_ceferino_baker_correlation_2017_pca_coeff.csv
    markhvida_ceferino_baker_correlation_2017_var_scale.csv
    du_ning_correlation_2021_model_coeff.csv du_ning_correlation_2021_pca_coeff.csv
    du_ning_correlation_2021_var_scale.csv
    CY14.csv ASK14.csv CB14.csv BSSA14.csv
    DESTINATION ${_rel})
