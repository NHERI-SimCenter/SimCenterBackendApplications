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

# Include directories
target_include_directories(smelt PUBLIC ${CONAN_INCLUDE_DIRS})
#target_include_directories(smelt PUBLIC ${CONAN_INCLUDE_DIRS_JANSSON})

