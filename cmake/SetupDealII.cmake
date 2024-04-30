#### deal.II #################################################################
find_package(deal.II 9.5 REQUIRED PATHS ${DEAL_II_DIR})

deal_ii_initialize_cached_variables()

set(DEAL_II_HAS_ALL_DEPENDENCIES TRUE)
if(NOT DEAL_II_WITH_CXX17)
  message(SEND_ERROR "deal.II needs to be configured with C++17 support.")
  set(DEAL_II_HAS_ALL_DEPENDENCIES FALSE)
endif()

if(NOT DEAL_II_WITH_MPI)
  message(SEND_ERROR "deal.II needs to be configured with MPI support.")
  set(DEAL_II_HAS_ALL_DEPENDENCIES FALSE)
endif()

if(NOT DEAL_II_WITH_ARBORX)
  message(SEND_ERROR "deal.II needs to be configured with ArborX support.")
  set(DEAL_II_HAS_ALL_DEPENDENCIES FALSE)
endif()

if(NOT DEAL_II_ARBORX_WITH_MPI)
  message(SEND_ERROR "ArborX needs to be configured with MPI support.")
  set(DEAL_II_HAS_ALL_DEPENDENCIES FALSE)
endif()

if(NOT DEAL_II_WITH_P4EST)
  message(SEND_ERROR "deal.II needs to be configured with P4EST support.")
  set(DEAL_II_HAS_ALL_DEPENDENCIES FALSE)
endif()

if(NOT DEAL_II_WITH_TRILINOS)
  message(SEND_ERROR "deal.II needs to be configured with Trilinos support.")
  set(DEAL_II_HAS_ALL_DEPENDENCIES FALSE)
endif()

if(NOT DEAL_II_HAS_ALL_DEPENDENCIES)
  message(FATAL_ERROR "deal.II wasn't configured with all required dependencies.")
endif()
