#pragma once

#include <iostream>
#include <vector>

#include "altro/common/knotpoint.hpp"
#include "altro/eigentypes.hpp"
#include "altro/utils/assert.hpp"

namespace altro {

/**
 * @brief Represents a state and control trajectory
 *
 * If the number of states or controls across the trajectory is not constant
 * the associated type parameter can be set to Eigen::Dynamic.
 *
 * @tparam n size of the state vector. Can be Eigen::Dynamic.
 * @tparam m size of the control vector. Can be Eigen::Dynamic.
 * @tparam T floating precision of state and control vectors
 */
template <int n, int m, class T = double>
class Trajectory {
  using StateVector = VectorN<n, T>;
  using ControlVector = VectorN<m, T>;

 public:
  /**
   * @brief Construct a new Trajectory object of size N
   *
   * @param N the number of segments in the trajectory. This means there are
   * N+1 state vectors and N control vectors.
   */
  explicit Trajectory(int N) : traj_(N) {}
  explicit Trajectory(int _n, int _m, int N)
      : traj_(N + 1, KnotPoint<n, m, T>(_n, _m)) {}
  explicit Trajectory(std::vector<KnotPoint<n, m, T>> zs) : traj_(zs) {}

  /**
   * @brief Construct a new Trajectory object from state, control and times
   *
   * @param X (N+1,) vector of states
   * @param U (N,) vector of controls
   * @param times (N+1,) vector of times
   */
  Trajectory(std::vector<VectorN<n, T>> X, std::vector<VectorN<m, T>> U,
             std::vector<float> times) {
    ALTRO_ASSERT(X.size() == U.size() + 1,
                 "Length of control vector must be one less than the length of "
                 "the state trajectory.");
    ALTRO_ASSERT(X.size() == times.size(),
                 "Length of times vector must be equal to the length of the "
                 "state trajectory.");
    int N = U.size();
    traj_.reserve(N + 1);
    for (int k = 0; k < N; ++k) {
      float h = times[k + 1] - times[k];
      traj_.emplace_back(X[k], U[k], times[k], h);
    }
    traj_.emplace_back(X[N], 0 * U[N-1], times[N], 0.0);
  }


  /***************************** Copying **************************************/
  Trajectory(const Trajectory& Z) : traj_(Z.traj_) {}
  Trajectory& operator=(const Trajectory& Z) {
    traj_ = Z.traj_;
    return *this;
  }

  /***************************** Moving ***************************************/
  Trajectory(Trajectory&& Z) : traj_(std::move(Z.traj_)) {}
  Trajectory& operator=(Trajectory&& Z) {
    traj_ = std::move(Z.traj_);
    return *this;
  }

  /*************************** Iteration **************************************/
  typedef typename std::vector<KnotPoint<n,m>>::iterator iterator;
  typedef typename std::vector<KnotPoint<n,m>>::const_iterator const_iterator;
  iterator begin() { return traj_.begin(); }
  const_iterator begin() const { return traj_.begin(); }
  iterator end() { return traj_.end(); }
  const_iterator end() const { return traj_.end(); }

  /*************************** Getters ****************************************/
  int NumSegments() const { return traj_.size() - 1; }
  StateVector& State(int k) { return traj_[k].State(); }
  ControlVector& Control(int k) { return traj_[k].Control(); }

  const KnotPoint<n, m, T>& GetKnotPoint(int k) const { return traj_[k]; }
  const StateVector& State(int k) const { return traj_[k].State(); }
  const ControlVector& Control(int k) const { return traj_[k].Control(); }

  KnotPoint<n, m, T>& GetKnotPoint(int k) { return traj_[k]; }
  KnotPoint<n, m, T>& operator[](int k) { return GetKnotPoint(k); }

  int StateDimension(int k) const { return traj_[k].StateDimension(); }
  int ControlDimension(int k) const { return traj_[k].ControlDimension(); }

  T GetTime(int k) const { return traj_[k].GetTime(); }
  float GetStep(int k) const { return traj_[k].GetStep(); }

  /*************************** Setters ****************************************/

  /**
   * @brief Set the states and controls to zero 
   * 
   */
  void SetZero() {
    for (iterator z_ptr = begin(); z_ptr != end(); ++z_ptr) {
      z_ptr->State().setZero();
      z_ptr->Control().setZero();
    }
  }

  void SetTime(int k, float t) { traj_[k].SetTime(t); }
  void SetStep(int k, float h) { traj_[k].SetStep(h); }

  void SetUniformStep(float h) {
    int N = NumSegments();
    for (int k = 0; k < N; ++k) {
      traj_[k].SetStep(h);
      traj_[k].SetTime(k * h);
    }
    traj_[N].SetStep(0.0);
    traj_[N].SetTime(h * N);
  }

  /**
   * @brief Check if the times and time steps are consistent
   *
   * @param eps tolerance check for float comparison
   * @return true if t[k+1] - t[k] == h[k] for all k
   */
  bool CheckTimeConsistency(const double eps = 1e-6,
                            const bool verbose = false) {
    for (int k = 0; k < NumSegments(); ++k) {
      float h_calc = GetTime(k + 1) - GetTime(k);
      float h_stored = GetStep(k);
      if (std::abs(h_stored - h_calc) > eps) {
        if (verbose) {
          std::cout << "k=" << k << "\t h=" << h_stored << std::endl;
          std::cout << "t-=" << GetTime(k) << "\t t+=" << GetTime(k + 1)
                    << "\t dt=" << h_calc << std::endl;
        }
        return false;
      }
    }
    return true;
  }

 private:
  std::vector<KnotPoint<n, m, T>> traj_;
};

using TrajectoryXXd = Trajectory<Eigen::Dynamic, Eigen::Dynamic, double>;

}  // namespace altro