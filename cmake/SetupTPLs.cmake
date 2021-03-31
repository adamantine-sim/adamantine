#### Message Passing Interface (MPI) #########################################
find_package(MPI REQUIRED)

#### Boost ###################################################################
if (DEFINED BOOST_DIR)
    set(BOOST_ROOT ${BOOST_DIR})
endif()
set(Boost_COMPONENTS
    chrono
    filesystem
    program_options
    timer
    unit_test_framework
)
find_package(Boost 1.70.0 REQUIRED COMPONENTS ${Boost_COMPONENTS})

#### Caliper #################################################################
if (ADAMANTINE_ENABLE_CALIPER)
  find_package(caliper REQUIRED PATHS ${CALIPER_DIR})
  add_compile_definitions(ADAMANTINE_WITH_CALIPER)
  message(STATUS "Found Caliper: ${caliper_INSTALL_PREFIX}")
endif()
