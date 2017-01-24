#### Message Passing Interface (MPI) #########################################
find_package(MPI REQUIRED)

#### Boost ###################################################################
if(DEFINED BOOST_DIR)
    set(BOOST_ROOT ${BOOST_DIR})
endif()
set(Boost_COMPONENTS
    chrono
    filesystem
    mpi
    program_options
    timer
    unit_test_framework
)
find_package(Boost 1.61.0 REQUIRED COMPONENTS ${Boost_COMPONENTS})

#### deal.II #################################################################
find_package(deal.II 8.5 REQUIRED PATHS ${DEAL_II_DIR})
