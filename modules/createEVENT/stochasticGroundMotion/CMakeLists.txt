simcenter_add_executable(NAME StochasticGM
                         DEPENDS CONAN_PKG::mkl-static CONAN_PKG::ipp-static
                         CONAN_PKG::smelt CONAN_PKG::clara CONAN_PKG::jsonformoderncpp)
