include(${CMAKE_SOURCE_DIR}/cmake/Testing.cmake)

add_definitions(-DBOOST_TEST_DYN_LINK)

include_directories(${CMAKE_SOURCE_DIR}/source)

set(UNIT_TESTS "")
list(APPEND
     UNIT_TESTS
     test_data_assimilator
     test_geometry
     test_heat_source
     test_implicit_operator
     test_material_property
     test_arraymd
     test_newton_solver
     test_post_processor
     test_scan_path
     test_thermal_operator
     test_timer
     test_utils
     )

set(MPI_UNIT_TESTS "")
list(APPEND
     MPI_UNIT_TESTS
     test_experimental_data
     test_integration_2d
     test_material_deposition
     test_thermal_physics
    )

set(CUDA_UNIT_TESTS "")
list(APPEND
     CUDA_UNIT_TESTS
     test_integration_2d_device
     test_arraymd_device
     test_thermal_operator_device
     test_thermal_physics_device
     )

foreach(TEST_NAME ${UNIT_TESTS})
  adamantine_ADD_BOOST_TEST(${TEST_NAME})
endforeach()

foreach(TEST_NAME ${MPI_UNIT_TESTS})
  adamantine_ADD_BOOST_TEST(${TEST_NAME} 1 2)
endforeach()

if (ADAMANTINE_ENABLE_CUDA)
  foreach(TEST_NAME ${CUDA_UNIT_TESTS})
    adamantine_ADD_BOOST_CUDA_TEST(${TEST_NAME})
  endforeach()
endif()

adamantine_COPY_INPUT_FILE(demo_316_short.info tests/data)
adamantine_COPY_INPUT_FILE(demo_316_short_scan_path.txt tests/data)
adamantine_COPY_INPUT_FILE(experimental_data_0_0.csv tests/data)
adamantine_COPY_INPUT_FILE(extruded_cube.msh tests/data)
adamantine_COPY_INPUT_FILE(integration_2d.info tests/data)
adamantine_COPY_INPUT_FILE(integration_2d_gold.txt tests/data)
adamantine_COPY_INPUT_FILE(material_deposition_2d.txt tests/data)
adamantine_COPY_INPUT_FILE(material_deposition_3d.txt tests/data)
adamantine_COPY_INPUT_FILE(material_path_test_material_deposition.txt tests/data)
adamantine_COPY_INPUT_FILE(raytracing_experimental_data_0_0.csv tests/data)
adamantine_COPY_INPUT_FILE(scan_path.txt tests/data)
adamantine_COPY_INPUT_FILE(scan_path_event_series.inp tests/data)
adamantine_COPY_INPUT_FILE(scan_path_test_thermal_physics.txt tests/data)
adamantine_COPY_INPUT_FILE(scan_path_L.txt tests/data)
adamantine_COPY_INPUT_FILE(scan_path_layers.txt tests/data)
