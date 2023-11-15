message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  add_compile_definitions(ADAMANTINE_DEBUG)
endif()

# deal.II flags override any other flags so we need to trick deal.II by
# appending user's flag to deal.II's flags
string(APPEND DEAL_II_CXX_FLAGS " $ENV{CXXFLAGS} ${CMAKE_CXX_FLAGS}")
string(APPEND DEAL_II_CXX_FLAGS_DEBUG " ${CMAKE_CXX_FLAGS_DEBUG}")
string(APPEND DEAL_II_CXX_FLAGS_RELEASE " ${CMAKE_CXX_FLAGS_RELEASE}")
string(APPEND DEAL_II_LINKER_FLAGS " $ENV{LDFLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_DEBUG "")
set(CMAKE_EXE_LINKER_FLAGS "")
