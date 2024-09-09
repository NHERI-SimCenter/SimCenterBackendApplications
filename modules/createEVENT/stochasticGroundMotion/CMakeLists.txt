simcenter_add_executable(NAME StochasticGM
                         DEPENDS CONAN_PKG::smelt CONAN_PKG::kissfft) 

set_property(TARGET StochasticGM PROPERTY CXX_STANDARD 17)
