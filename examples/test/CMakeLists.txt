add_executable(quadraticcost_test
  quadraticcost_test.cpp
)
target_link_libraries(quadraticcost_test
  gtest_main
  cost_function_examples
)
gtest_discover_tests(quadraticcost_test)

add_executable(triple_integrator_test
  triple_integrator_test.cpp
)
target_link_libraries(triple_integrator_test
  gtest_main
  dynamics_examples
)
gtest_discover_tests(triple_integrator_test)