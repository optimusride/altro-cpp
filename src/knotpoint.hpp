#pragma once

// #include <iostream>
#include "eigentypes.hpp"

namespace altro {

/**
 * @brief Stores the state, control, time, and time step for a single knot point
 *
 * The state and control vectors can live on either the stack or the heap,
 * leveraging Eigen's Matrix class. If @tparam n or @tparam m are equal to
 * Eigen::Dynamic, the vector will allocated on the heap.
 *
 * Use `StateDimension` and `ControlDimension` to query the actual state or
 * control dimesnion. Use `StateSize` and `ControlSize` to get the type
 * parameters.
 *
 * @tparam n size of state vector. Can be Eigen::Dynamic.
 * @tparam m size of state vector. Can be Eigen::Dynamic.
 * @tparam T precision of state and control variables
 */
template <int n, int m, class T = double>
class KnotPoint {
  using StateVector = Vector<n, T>;
  using ControlVector = Vector<m, T>;

 public:
  KnotPoint()
      : x_(StateVector::Zero()),
        u_(ControlVector::Zero()),
        t_(0.0f),
        h_(0.0),
        n_(n),
        m_(m) {}
  KnotPoint(const StateVector& x, const ControlVector& u, float t = 0.0,
            float h = 0.0)
      : x_(x), u_(u), t_(t), h_(h), n_(x.size()), m_(u.size()) {}
  KnotPoint(int _n, int _m)
      : x_(StateVector::Zero(_n)),
        u_(ControlVector::Zero(_m)),
        t_(0.0f),
        h_(0.0),
        n_(_n),
        m_(_m) {}

  KnotPoint(const KnotPoint& z)
      : x_(z.x_), u_(z.u_), t_(z.t_), h_(z.h_), n_(z.n_), m_(z.m_) {}
  KnotPoint& operator=(const KnotPoint& z) {
    x_ = z.x_;
    u_ = z.u_;
    t_ = z.t_;
    h_ = z.h_;
    n_ = z.n_;
    m_ = z.m_;
    return *this;
  }

  KnotPoint(KnotPoint&& z)
      : x_(std::move(z.x_)), u_(std::move(z.u_)), t_(z.t_), h_(z.h_) {}
  KnotPoint& operator=(KnotPoint&& z) {
    x_ = std::move(z.x_);
    u_ = std::move(z.u_);
    t_ = z.t_;
    h_ = z.h_;
    return *this;
  }

  int StateDimension() const { return n_; }
  int ControlDimension() const { return m_; }
  static int StateSize() { return n; }
  static int ControlSize() { return m; }
  StateVector& State() { return x_; }
  ControlVector& Control() { return u_; }
  const StateVector& State() const { return x_; }
  const ControlVector& Control() const { return u_; }
  Vector<n + m, T> GetStateControl() const {
    Vector<n + m, T> z;
    z << x_, u_;
    return z;
  }
  float GetTime() const { return t_; }
  float GetStep() const { return h_; }
	bool IsTerminal() const { return h_ == 0; }

	friend std::ostream& operator<<(std::ostream &os, const KnotPoint<n,m,T>& z) {
		return os << "x: [" << z.State().transpose() 
		          << "], u: [" << z.Control().transpose()
							<< "], t=" << z.GetTime()
							<< ", h=" << z.GetStep();
}

 private:
  StateVector x_;
  ControlVector u_;
  float t_;  // time
  float h_;  // time step
  int n_;    // state dimension
  int m_;    // control dimension
};



}  // namespace altro
