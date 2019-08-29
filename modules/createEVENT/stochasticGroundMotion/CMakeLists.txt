simcenter_add_executable(NAME StochasticGroundMotion
                         DEPENDS CONAN_PKG::mkl-static CONAN_PKG::ipp-static
                         CONAN_PKG::smelt CONAN_PKG::eigen CONAN_PKG::clara CONAN_PKG::jsonformoderncpp)
