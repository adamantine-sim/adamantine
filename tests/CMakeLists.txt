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
     test_utils
     )

foreach(TEST_NAME ${UNIT_TESTS})
  adamantine_ADD_BOOST_TEST(${TEST_NAME})
endforeach()

# For now integration tests only run adamantine but they don't check the 
# results
set(INTEGRATION_TESTS "")
list(APPEND
     INTEGRATION_TESTS
     test_integration_2d
     )

foreach(TEST_NAME ${INTEGRATION_TESTS})
  adamantine_ADD_INTEGRATION_TEST(${TEST_NAME})
endforeach()
