#### deal.II #################################################################
find_package(deal.II 9.6 REQUIRED PATHS ${DEAL_II_DIR})

deal_ii_initialize_cached_variables()

set(DEAL_II_REQUIRED_FEATURES ARBORX CXX17 MPI P4EST TRILINOS)

foreach(FEATURE ${DEAL_II_REQUIRED_FEATURES})
  if(NOT DEAL_II_WITH_${FEATURE})
    list(APPEND DEAL_II_MISSING_FEATURES ${FEATURE})
  endif()
endforeach()

if(DEAL_II_MISSING_FEATURES)
  string(REPLACE ";"  ", " DEAL_II_MISSING_FEATURES "${DEAL_II_MISSING_FEATURES}")
  message(FATAL_ERROR "deal.II wasn't configured with all required dependencies. The missing dependencies are ${DEAL_II_MISSING_FEATURES}.")
endif()

if(NOT DEAL_II_ARBORX_WITH_MPI)
  message(FATAL_ERROR "ArborX needs to be configured with MPI support.")
endif()
