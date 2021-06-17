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