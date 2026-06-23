######################################################################################################################
# SimCenterConanCompat.cmake
#
# Provides backward compatibility for module CMakeLists.txt written against Conan 1,
# which used CONAN_PKG::<name> target names and the CONAN_INCLUDE_DIRS variable.
#
# Conan 2 / CMakeDeps generates <Package>::<Component> targets whose names vary by
# recipe version.  This file creates ALIAS targets (CONAN_PKG::<name>) pointing at
# whatever real target was actually generated, trying multiple candidate names so
# the mapping stays correct even if conan-center-index recipes change.
#
# Must be included AFTER all find_package() calls in the top-level CMakeLists.txt.
######################################################################################################################

# _simcenter_conan_compat(<conan1_name> <candidate1> [<candidate2> ...])
#
# Finds the first existing candidate target and creates:
#   CONAN_PKG::<conan1_name>  ALIAS pointing at it
#   Appends its include dirs to the CONAN_INCLUDE_DIRS cache variable.
#
function(_simcenter_conan_compat conan1_name)
    if(TARGET CONAN_PKG::${conan1_name})
        return()
    endif()

    foreach(_candidate ${ARGN})
        if(TARGET ${_candidate})
            add_library(CONAN_PKG::${conan1_name} ALIAS ${_candidate})

            get_target_property(_incs ${_candidate} INTERFACE_INCLUDE_DIRECTORIES)
            if(_incs)
                # Global CONAN_INCLUDE_DIRS (all packages combined)
                set(_merged ${CONAN_INCLUDE_DIRS} ${_incs})
                list(REMOVE_DUPLICATES _merged)
                set(CONAN_INCLUDE_DIRS ${_merged} CACHE INTERNAL
                    "Aggregated Conan include dirs (Conan 1 compat)")

                # Per-package CONAN_INCLUDE_DIRS_<NAME> (uppercase)
                string(TOUPPER "${conan1_name}" _upper)
                set(CONAN_INCLUDE_DIRS_${_upper} ${_incs} CACHE INTERNAL
                    "Conan 1 compat include dirs for ${conan1_name}")
            endif()
            return()
        endif()
    endforeach()

    message(WARNING
        "SimCenterConanCompat: no target found for CONAN_PKG::${conan1_name}. "
        "Tried: ${ARGN}")
endfunction()

# Candidate lists are ordered most-likely-first; the first match wins.
# Multiple candidates handle recipe changes across conan-center-index versions.
_simcenter_conan_compat(jansson          jansson::jansson)
_simcenter_conan_compat(zlib             ZLIB::ZLIB)
_simcenter_conan_compat(libcurl          CURL::libcurl   CURL::CURL)
_simcenter_conan_compat(eigen            Eigen3::Eigen   Eigen3::Eigen3  eigen::eigen)
_simcenter_conan_compat(clara            clara::clara)
_simcenter_conan_compat(nanoflann        nanoflann::nanoflann)
_simcenter_conan_compat(nlopt            NLopt::nlopt    nlopt::nlopt)
_simcenter_conan_compat(kissfft          kissfft::kissfft)
_simcenter_conan_compat(boost            Boost::boost    Boost::headers)
_simcenter_conan_compat(jsonformoderncpp nlohmann_json::nlohmann_json)
