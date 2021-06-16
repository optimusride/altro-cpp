add_executable(trajectory_test
  trajectory_test.cpp
)
target_link_libraries(trajectory_test
  gtest_main
  trajectory
)
gtest_discover_tests(trajectory_test)

add_executable(costexpansion_test
  costexpansion_test.cpp
)
target_link_libraries(costexpansion_test
  gtest_main
  altro 
)
gtest_discover_tests(costexpansion_test)


add_executable(knotpoint_test
  knotpoint_test.cpp
)
target_link_libraries(knotpoint_test
  gtest_main
  altro
)
gtest_discover_tests(knotpoint_test)