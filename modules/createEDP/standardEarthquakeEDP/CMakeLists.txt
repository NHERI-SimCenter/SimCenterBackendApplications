simcenter_add_executable(NAME StandardEarthquakeEDP
                         FILES StandardEarthquakeEDP.cpp
                         DEPENDS CONAN_PKG::jansson)
