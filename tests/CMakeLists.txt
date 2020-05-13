include(${CMAKE_SOURCE_DIR}/cmake/Testing.cmake)

add_definitions(-DBOOST_TEST_DYN_LINK)

include_directories(${CMAKE_SOURCE_DIR}/source)

set(UNIT_TESTS "")
list(APPEND
     UNIT_TESTS
     test_electron_beam
     test_geometry
     test_implicit_operator
     test_material_property
     test_newton_solver
     test_post_processor
     test_thermal_operator
     test_thermal_physics
     test_timer
     test_utils
     )

set(CUDA_UNIT_TESTS "")
list(APPEND
     CUDA_UNIT_TESTS
     test_thermal_operator_device
     )

foreach(TEST_NAME ${UNIT_TESTS})
  adamantine_ADD_BOOST_TEST(${TEST_NAME})
endforeach()

if (ADAMANTINE_ENABLE_CUDA)
  foreach(TEST_NAME ${CUDA_UNIT_TESTS})
    adamantine_ADD_BOOST_CUDA_TEST(${TEST_NAME})
  endforeach()
endif()

# For now integration tests only run adamantine but they don't check the 
# results
adamantine_ADD_INTEGRATION_TEST(test_integration_2d 1)

adamantine_COPY_INPUT_FILE(beam_position.csv tests/data)
