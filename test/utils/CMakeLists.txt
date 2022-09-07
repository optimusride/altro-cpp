add_executable(benchmarking_test
  benchmarking_test.cpp
)
target_link_libraries(benchmarking_test
  PRIVATE
  gtest_main
  util
  fmt::fmt
)
gtest_discover_tests(benchmarking_test)


# Derivative Checker
add_executable(derivative_checker_test
  derivative_checker_test.cpp
)
target_link_libraries(derivative_checker_test
  PRIVATE
  Eigen3::Eigen
  gtest_main
  util
)
gtest_discover_tests(derivative_checker_test)
