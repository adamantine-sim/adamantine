cmake_minimum_required(VERSION 3.1)

project(adamantine LANGUAGES CXX VERSION ${adamantine_VERSION})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(SetupTPLs)
include(Setupadamantine)
include(UnitTesting)

enable_testing()
include(CTest)

add_subdirectory(source)
add_subdirectory(tests)

# Provide "indent" target for indenting all the header and the source files.
add_custom_target(indent
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND ./indent
  )
