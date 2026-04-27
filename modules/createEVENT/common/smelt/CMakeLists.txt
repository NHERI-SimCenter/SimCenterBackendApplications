add_library(smelt STATIC
  numeric_utils.cc
  normal_multivar.cc
  normal_dist.cc
  lognormal_dist.cc
  beta_dist.cc
  inv_gauss_dist.cc
  students_t_dist.cc
  json_object.cc
  vlachos_et_al.cc
  configure.cc
  wittig_sinha.cc
  filter.cc
  wind_profile.cc
  uniform_dist.cc
  dabaghi_der_kiureghian.cc
  nelder_mead.cc)

target_include_directories(smelt PUBLIC ${CMAKE_SOURCE_DIR}/modules)

target_link_libraries(smelt PUBLIC
    jansson::jansson
    nlohmann_json::nlohmann_json
    kissfft::kissfft
    Boost::boost
    NLopt::nlopt
    Eigen3::Eigen)
