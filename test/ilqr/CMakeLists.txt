# Cost Expansion tests
add_executable(cost_expansion_test
  cost_expansion_test.cpp
)
target_link_libraries(cost_expansion_test
  gtest_main
  ilqr
  cost_function_examples
)
target_include_directories(cost_expansion_test
  PRIVATE
  ${PROJECT_SOURCE_DIR}/examples
)
gtest_discover_tests(cost_expansion_test)