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

#### deal.II #################################################################
find_package(deal.II 9.3 REQUIRED PATHS ${DEAL_II_DIR})

# If deal.II was configured in DebugRelease mode, then if adamantine was configured
# in Debug mode, we link against the Debug version of deal.II. If adamatine was
# configured in Release mode, we link against the Release version of eal.II
string(FIND "${DEAL_II_LIBRARIES}" "general" SINGLE_DEAL_II)
if (${SINGLE_DEAL_II} EQUAL -1)
    if(CMAKE_BUILD_TYPE MATCHES "Release")
        set(DEAL_II_LIBRARIES ${DEAL_II_LIBRARIES_RELEASE})
    else()
        set(DEAL_II_LIBRARIES ${DEAL_II_LIBRARIES_DEBUG})
  endif()
endif()
