set(LOCAL_LOG_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(benchmark_triple_integrator
  benchmark_triple_integrator.cpp
)
target_link_libraries(benchmark_triple_integrator
  example_problems
)
target_compile_definitions(benchmark_triple_integrator
  PRIVATE
  LOCAL_LOG_DIR="${LOCAL_LOG_DIR}"
)

add_executable(benchmark_unicycle
  benchmark_unicycle.cpp
)
target_link_libraries(benchmark_unicycle
  example_problems
)
target_compile_definitions(benchmark_unicycle 
  PRIVATE
  LOCAL_LOG_DIR="${LOCAL_LOG_DIR}"
)

add_executable(benchmark_expansions
  benchmark_expansions.cpp
)
target_link_libraries(benchmark_expansions
  PRIVATE
  example_problems
  common
)
target_compile_definitions(benchmark_expansions
  PRIVATE
  LOCAL_LOG_DIR="${LOCAL_LOG_DIR}"
)

add_executable(benchmark_threadpool
  benchmark_threadpool.cpp
)
target_link_libraries(benchmark_threadpool
  PRIVATE
  example_problems
  common
)
target_compile_definitions(benchmark_threadpool
  PRIVATE
  LOCAL_LOG_DIR="${LOCAL_LOG_DIR}"
)