# Trajectory
add_executable(trajectory_test
  trajectory_test.cpp
)
target_link_libraries(trajectory_test
  PRIVATE
  gtest_main
  common
  utils
)
gtest_discover_tests(trajectory_test)

# Knotpoint
add_executable(knotpoint_test
  knotpoint_test.cpp
)
target_link_libraries(knotpoint_test
  PRIVATE
  gtest_main
  common
  utils
)
gtest_discover_tests(knotpoint_test)

# Solver options
add_executable(solver_options_test
  solver_options_test.cpp
)
target_link_libraries(solver_options_test
  PRIVATE
  gtest_main
  common
  utils
)
gtest_discover_tests(solver_options_test)

add_executable(solver_logging_test
  solver_logging_test.cpp
)
target_link_libraries(solver_logging_test
  PRIVATE
  gtest_main
  common
  utils
)
gtest_discover_tests(solver_logging_test)

# Timer
add_executable(timer_test
  timer_test.cpp
)
target_link_libraries(timer_test
  PRIVATE
  gtest_main
  common
)
gtest_discover_tests(timer_test)
target_compile_definitions(timer_test PRIVATE LOGDIR="${CMAKE_CURRENT_BINARY_DIR}")

add_executable(functionbase_test
  functionbase_test.cpp
)
target_link_libraries(functionbase_test
  PRIVATE
  gtest_main
  common
  utils
)
gtest_discover_tests(functionbase_test)

add_executable(threadpool_test
  threadpool_test.cpp
)
target_link_libraries(threadpool_test
  PRIVATE
  gtest_main
  common
  threadpool
  utils
)
gtest_discover_tests(threadpool_test TEST_LIST THREADTESTS)
set_tests_properties(${THREADTESTS} PROPERTIES RUN_SERIAL True)