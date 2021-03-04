message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  add_definitions(-DADAMANTINE_DEBUG)
endif()
