#### deal.II #################################################################
find_package(deal.II 9.5 REQUIRED PATHS ${DEAL_II_DIR})

# If deal.II was configured in DebugRelease mode, then if adamantine was configured
# in Debug mode, we link against the Debug version of deal.II. If adamantine was
# configured in Release mode, we link against the Release version of eal.II
string(FIND "${DEAL_II_LIBRARIES}" "general" SINGLE_DEAL_II)
if (${SINGLE_DEAL_II} EQUAL -1)
    if(CMAKE_BUILD_TYPE MATCHES "Release")
        set(DEAL_II_LIBRARIES ${DEAL_II_LIBRARIES_RELEASE})
    else()
        set(DEAL_II_LIBRARIES ${DEAL_II_LIBRARIES_DEBUG})
  endif()
endif()

