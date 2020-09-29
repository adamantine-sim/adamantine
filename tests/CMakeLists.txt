include(${CMAKE_SOURCE_DIR}/cmake/Testing.cmake)

add_definitions(-DBOOST_TEST_DYN_LINK)

include_directories(${CMAKE_SOURCE_DIR}/source)

set(UNIT_TESTS "")
list(APPEND
     UNIT_TESTS
     test_geometry
     test_heat_source
     test_implicit_operator
     test_integration_2d
     test_material_property
     test_newton_solver
     test_post_processor
     test_scan_path
     test_thermal_operator
     test_thermal_physics
     test_timer
     test_utils
     )

set(CUDA_UNIT_TESTS "")
list(APPEND
     CUDA_UNIT_TESTS
     test_integration_2d_device
     test_thermal_operator_device
     test_thermal_physics_device
     )

foreach(TEST_NAME ${UNIT_TESTS})
  adamantine_ADD_BOOST_TEST(${TEST_NAME})
endforeach()

if (ADAMANTINE_ENABLE_CUDA)
  foreach(TEST_NAME ${CUDA_UNIT_TESTS})
    adamantine_ADD_BOOST_CUDA_TEST(${TEST_NAME})
  endforeach()
endif()

adamantine_COPY_INPUT_FILE(beam_position.csv tests/data)
adamantine_COPY_INPUT_FILE(integration_2d.info tests/data)
adamantine_COPY_INPUT_FILE(integration_2d_gold.txt tests/data)
adamantine_COPY_INPUT_FILE(scan_path.txt tests/data)
adamantine_COPY_INPUT_FILE(scan_path_test_thermal_physics.txt tests/data)
adamantine_COPY_INPUT_FILE(extruded_cube.msh tests/data)
