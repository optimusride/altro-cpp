
# Derivative Checker
add_executable(derivative_checker_test
  derivative_checker_test.cpp
)
target_link_libraries(derivative_checker_test
  gtest_main
  util
)
gtest_discover_tests(derivative_checker_test)
