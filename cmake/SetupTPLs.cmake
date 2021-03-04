#### Message Passing Interface (MPI) #########################################
find_package(MPI REQUIRED)

#### Boost ###################################################################
if(DEFINED BOOST_DIR)
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
