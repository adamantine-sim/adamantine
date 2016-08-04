include(${CMAKE_SOURCE_DIR}/cmake/UnitTesting.cmake)

add_definitions(-DBOOST_TEST_DYN_LINK)

include_directories(${CMAKE_SOURCE_DIR}/source)

set(TESTS "")
list(APPEND
     TESTS
     test_electron_beam
     test_geometry
     test_material_property
     test_post_processor
     test_thermal_operator
     test_thermal_physics
     )

foreach(TEST_NAME ${TESTS})
  adamantine_ADD_BOOST_TEST(${TEST_NAME})
endforeach()
