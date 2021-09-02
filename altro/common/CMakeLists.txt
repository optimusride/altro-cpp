add_library(common
  knotpoint.cpp
  knotpoint.hpp
  state_control_sized.hpp
  trajectory.cpp
  trajectory.hpp
  solver_options.hpp
  solver_options.cpp
  solver_stats.hpp
  solver_stats.cpp
  log_entry.hpp
  log_entry.cpp
  solver_logger.hpp
  solver_logger.cpp
  timer.hpp
  timer.cpp
  profile_entry.hpp
  profile_entry.cpp
  functionbase.hpp
  functionbase.cpp
)
target_link_libraries(common 
  PUBLIC
  Eigen3::Eigen # needs to be public since the interface uses Eigen
  fmt::fmt      # needs to be public since solver_logger uses it in the interface

  PRIVATE
  utils
)
target_compile_definitions(common 
  PRIVATE
  LOGDIR="${LOGDIR}"
)
export_library(common AltroCppTargets)

add_library(threadpool
  threadsafe_queue.hpp
  threadsafe_queue.cpp
  threadpool.hpp
  threadpool.cpp
)
target_link_libraries(threadpool
  PUBLIC
  Threads::Threads
  PRIVATE
  fmt::fmt
  utils
)
export_library(threadpool AltroCppTargets)
