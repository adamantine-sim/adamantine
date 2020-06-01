# Create adamantine executable and link against the static library.
if (ADAMANTINE_ENABLE_CUDA)
  add_executable(adamantine adamantine.cu)
else()
  add_executable(adamantine adamantine.cc)
endif()
DEAL_II_SETUP_TARGET(adamantine)
target_link_libraries(adamantine Adamantine)
file(COPY input.info DESTINATION ${CMAKE_BINARY_DIR}/bin)
