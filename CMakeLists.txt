cmake_minimum_required(VERSION 3.1)

project(adamantine LANGUAGES CXX VERSION ${adamantine_VERSION})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(SetupTPLs)
include(Setupadamantine)

enable_testing()
include(CTest)

add_subdirectory(source)
