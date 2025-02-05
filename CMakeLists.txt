cmake_minimum_required(VERSION 3.15)

if(POLICY CMP0167)
  # Policy introduced in CMake 3.30. Use the Boost CMake Config installed with Boost instead of
  # the FindBoost.cmake module installed with CMake.
  cmake_policy(SET CMP0167 NEW)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# This needs to be called before project is called and before we setup the other
# TPL and adamantine
include(SetupDealII)

project(Adamantine LANGUAGES CXX VERSION 1.0.9)

include(SetupTPLs)
include(SetupAdamantine)

add_subdirectory(application)
add_subdirectory(source)

option(BUILD_SHARED_LIBS "Build the adamantine library as a shared object" ON)

option(ADAMANTINE_ENABLE_COVERAGE "Measure coverage" OFF)
if (ADAMANTINE_ENABLE_COVERAGE)
  include(CodeCoverage)
endif()

option(ADAMANTINE_ENABLE_TESTS "Build tests" OFF)
if (ADAMANTINE_ENABLE_TESTS)
  include(Testing)
  enable_testing()
  include(CTest)
  add_subdirectory(tests)
  add_test(NAME indent_code
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMAND ./indent
  )
endif()

# Provide "indent" target for indenting all the header and the source files.
add_custom_target(indent
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND ./indent
)

if (ADAMANTINE_ENABLE_DOCUMENTATION)
    add_subdirectory(doc)
endif()
