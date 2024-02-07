#### deal.II #################################################################
find_package(deal.II 9.5 REQUIRED PATHS ${DEAL_II_DIR})

deal_ii_initialize_cached_variables()

if(NOT DEAL_II_WITH_CXX17)
  message(FATAL_ERROR "deal.II needs to be configured with C++17 support.")
endif()

if(NOT DEAL_II_WITH_MPI)
  message(FATAL_ERROR "deal.II needs to be configured with MPI support.")
endif()

if(NOT DEAL_II_WITH_ARBORX)
  message(FATAL_ERROR "deal.II needs to be configured with ArborX support.")
endif()

if(NOT DEAL_II_ARBORX_WITH_MPI)
  message(FATAL_ERROR "ArborX needs to be configured with MPI support.")
endif()

if(NOT DEAL_II_WITH_P4EST)
  message(FATAL_ERROR "deal.II needs to be configured with P4EST support.")
endif()

if(NOT DEAL_II_WITH_TRILINOS)
  message(FATAL_ERROR "deal.II needs to be configured with Trilinos support.")
endif()
