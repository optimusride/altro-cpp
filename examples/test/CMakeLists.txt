add_executable(quadratic_cost_test
  quadratic_cost_test.cpp
)
target_link_libraries(quadratic_cost_test
  gtest_main
  cost_function_examples
)
gtest_discover_tests(quadratic_cost_test)

add_executable(triple_integrator_test
  triple_integrator_test.cpp
)
target_link_libraries(triple_integrator_test
  gtest_main
  dynamics_examples
  utils
)
gtest_discover_tests(triple_integrator_test)