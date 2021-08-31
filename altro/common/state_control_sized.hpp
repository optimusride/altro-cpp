// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <eigen3/Eigen/Dense>

#include "altro/utils/assert.hpp"

namespace altro {

/**
 * @brief Add the state and control dimension together
 * If either is dynamically sized (Eigen::Dynamic) then their addition
 * will be as well. Useful for template metaprogramming.
 * 
 * It provides `StateDimension()` and `ControlDimension()` for checking the 
 * runtime (actual) sizes, and `StateMemorySize()` and `ControlMemorySize()` for
 * checking the compile-time size.
 *
 * @param n state dimension at compile time
 * @param m control dimension at compile time
 * @return n+m if both sizes are known at compile time, Eigen::Dynamic otherwise
 */
constexpr int AddSizes(int n, int m) {
  if (n == Eigen::Dynamic || m == Eigen::Dynamic) {
    return Eigen::Dynamic;
  }
  return n + m;
}

/**
 * @brief A super class that stores the state and control dimension
 * 
 * With an option to store them at compile time for statically-sized data 
 * structures.
 * 
 * Here's an example of basic inheritance
 * @code {.cpp}
 * template <int n, int m>
 * class Derived : public StateControlSized<n,m> {
 * 	public:
 *   Derived(int state_dim, int control_dim) 
 * 	     : StateControlSized<n,m>(state_dim, control_dim) {}
 * }
 * @endcode
 * 
 * @param n state dimension at compile time
 * @param m control dimension at compile time
 */
template <int n, int m>
class StateControlSized {
 public:
  StateControlSized(int state_dim, int control_dim)
      : n_(state_dim), m_(control_dim) {
    if (n > 0) {
      ALTRO_ASSERT(n == n_, "State sizes must be consistent.");
    }
    if (m > 0) {
      ALTRO_ASSERT(m == m_, "Control sizes must be consistent.");
    }
  }
  StateControlSized() : n_(n), m_(m) {
    ALTRO_ASSERT(n > 0, "State dimension must be greater than zero.");
    ALTRO_ASSERT(m > 0, "Control dimension must be greater than zero.");
  }

  int StateDimension() const { return n_; }
  int ControlDimension() const { return m_; }
  static constexpr int StateMemorySize() { return n; }
  static constexpr int ControlMemorySize() { return m; }

 protected:
  int n_;
  int m_;
};

}  // namespace altro