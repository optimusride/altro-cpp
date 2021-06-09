add_executable(quadraticcost_test
  quadraticcost_test.cpp
)
target_link_libraries(quadraticcost_test
  gtest_main
  costfunctions
)
gtest_discover_tests(quadraticcost_test)