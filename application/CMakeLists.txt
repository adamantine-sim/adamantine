# Create adamantine executable and link against the static library.
if (ADAMANTINE_ENABLE_CUDA)
  add_executable(adamantine adamantine.cu)
  target_compile_definitions(adamantine PRIVATE ADAMANTINE_HAVE_CUDA)
  set_target_properties(adamantine PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
  )
else()
  add_executable(adamantine adamantine.cc)
  set_target_properties(adamantine PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
  )
endif()

DEAL_II_SETUP_TARGET(adamantine)
target_link_libraries(adamantine Adamantine)
file(COPY input.info DESTINATION ${CMAKE_BINARY_DIR}/bin)
file(COPY input_scan_path.txt DESTINATION ${CMAKE_BINARY_DIR}/bin)
