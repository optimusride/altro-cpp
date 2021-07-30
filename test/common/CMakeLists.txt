# Trajectory
add_executable(trajectory_test
  trajectory_test.cpp
)
target_link_libraries(trajectory_test
  gtest_main
  common
)
gtest_discover_tests(trajectory_test)

# Knotpoint
add_executable(knotpoint_test
  knotpoint_test.cpp
)
target_link_libraries(knotpoint_test
  gtest_main
  common
)
gtest_discover_tests(knotpoint_test)

# Solver options
add_executable(solver_options_test
  solver_options_test.cpp
)
target_link_libraries(solver_options_test
  gtest_main
  common
)
gtest_discover_tests(solver_options_test)

add_executable(solver_logging_test
  solver_logging_test.cpp
)
target_link_libraries(solver_logging_test
  gtest_main
  common
)
gtest_discover_tests(solver_logging_test)

# Timer
add_executable(timer_test
  timer_test.cpp
)
target_link_libraries(timer_test
  gtest_main
  common
)
gtest_discover_tests(timer_test)
target_compile_definitions(timer_test PRIVATE LOGDIR="${CMAKE_CURRENT_BINARY_DIR}")
