file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES
    TaichiEvent.py taichi_script.py mpm_lagrangian_forces.py mpm128.py mpm3d.py pbf2d.py
    karman_vortex_street.py vortex_rings.py mass_spring_game.py fem99.py fem128.py
    euler.py eulerfluid2d.py stable_fluid.py waterwave.py mandelbrot_zoom.py
    DESTINATION ${_rel})
