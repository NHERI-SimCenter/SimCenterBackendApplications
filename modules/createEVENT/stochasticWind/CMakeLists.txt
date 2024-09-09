simcenter_add_executable(NAME StochasticWind
                         DEPENDS CONAN_PKG::smelt CONAN_PKG::kissfft common)

set_property(TARGET StochasticWind PROPERTY CXX_STANDARD 17)
