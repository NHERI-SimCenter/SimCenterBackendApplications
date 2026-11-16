file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES
    flume.py GeoClaw.py GeoClawBathy.py GetOpenFOAMEvent.py hydroUtils.py
    of7Alpboundary.py of7Building.py of7Dakota.py of7Decomp.py of7Geometry.py
    of7Initial.py of7Materials.py of7Meshing.py of7Others.py of7Prboundary.py
    of7Process.py of7PtDboundary.py of7Solve.py of7Turbulence.py of7Uboundary.py
    openfoam7.py osuFlume.py Processor.py GeoClawOpenFOAM.py userFlume.py
    DESTINATION ${_rel})
