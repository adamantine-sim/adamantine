set(adamantine_VERSION_MAJOR 0)
set(adamantine_VERSION_MINOR 1)
set(adamantine_VERSION_PATCH 0)
set(adamantine_VERSION
    ${adamantine_VERSION_MAJOR}.${adamantine_VERSION_MINOR}.${adamantine_VERSION_PATCH})
message("adamatine version: ${adamantine_VERSION}")
    
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

message("Build type: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE MATCHES "Release")
    add_definitions(-DBOOST_DISABLE_ASSERTS)
elseif(CMAKE_BUILD_TYPE MATCHES "Debug")
  add_definitions(-DADAMANTINE_DEBUG)
else()
    message(FATAL_ERROR
        "Possible values for CMAKE_BUILD_TYPE are Debug and Release"
    )
endif()
