simcenter_add_executable(NAME StochasticWind
                         DEPENDS CONAN_PKG::smelt CONAN_PKG::ipp-static
                         CONAN_PKG::mkl-static common)
