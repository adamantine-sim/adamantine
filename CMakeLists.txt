cmake_minimum_required(VERSION 3.13)

project(adamantine LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(Setupadamantine)
include(SetupTPLs)


add_subdirectory(application)
add_subdirectory(source)

option(ADAMANTINE_ENABLE_COVERAGE "Measure coverage" OFF)
if(ADAMANTINE_ENABLE_COVERAGE)
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

if(ADAMANTINE_ENABLE_DOCUMENTATION)
    add_subdirectory(doc)
endif()
