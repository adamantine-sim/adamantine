include(${CMAKE_SOURCE_DIR}/cmake/UnitTesting.cmake)

add_definitions(-DBOOST_TEST_DYN_LINK)

include_directories(${CMAKE_SOURCE_DIR}/source)

set(TESTS "")
list(APPEND
     TESTS
     test_geometry
     )

set(tests_SOURCES
    ${CMAKE_SOURCE_DIR}/source/Geometry.cc
    ${CMAKE_SOURCE_DIR}/source/ThermalOperator.cc
  )

foreach(TEST_NAME ${TESTS})
  adamantine_ADD_BOOST_TEST(${TEST_NAME})
endforeach()
