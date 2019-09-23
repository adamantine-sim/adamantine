message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE MATCHES "Release")
  add_definitions(-DBOOST_DISABLE_ASSERTS)
elseif(CMAKE_BUILD_TYPE MATCHES "Debug")
  add_definitions(-DADAMANTINE_DEBUG)
else()
  message(SEND_ERROR
        "Possible values for CMAKE_BUILD_TYPE are Debug and Release"
    )
endif()
