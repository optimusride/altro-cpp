
# Quadratic Cost 
add_executable(quadratic_cost_test
  quadratic_cost_test.cpp
)
target_link_libraries(quadratic_cost_test
  gtest_main
  cost_function_examples
)
gtest_discover_tests(quadratic_cost_test)

# Triple integrator
add_executable(triple_integrator_test
  triple_integrator_test.cpp
)
target_link_libraries(triple_integrator_test
  gtest_main
  dynamics_examples
  utils
)
gtest_discover_tests(triple_integrator_test)

# Unicycle
add_executable(unicycle_test
  unicycle_test.cpp
)
target_link_libraries(unicycle_test
  gtest_main
  dynamics_examples
  utils
)
gtest_discover_tests(unicycle_test)

# Problem
add_executable(problem_test
  problem_test.cpp
)
target_link_libraries(problem_test
  constraints
	constraint_examples
  gtest_main
  problem
  dynamics_examples
  cost_function_examples
  utils
)
gtest_discover_tests(problem_test)