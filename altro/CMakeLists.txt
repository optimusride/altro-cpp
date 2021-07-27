if (RUN_CLANG_TIDY AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  message("Running clang-tidy on source directory.")
  set(CMAKE_CXX_CLANG_TIDY
      clang-tidy-8;
      -export-fixes=tidy-fixes.yaml;
      -header-filter=.;
      -checks="bugprone-*, clang-*, google-*, readability-*, modernize-*, performance-*, portability-*";)
endif()

add_subdirectory(common)
add_subdirectory(ilqr)
add_subdirectory(problem)
add_subdirectory(utils)
add_subdirectory(constraints)
add_subdirectory(augmented_lagrangian)
