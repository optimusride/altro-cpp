#pragma once

#include "altro/common/state_control_sized.hpp"
#include "altro/eigentypes.hpp"
#include "altro/utils/assert.hpp"

namespace altro {

/**
 * @brief Stores the state, control, time, and time step for a single knot point
 *
 * The state and control vectors can live on either the stack or the heap,
 * leveraging Eigen's Matrix class. If @tparam n or @tparam m are equal to
 * Eigen::Dynamic, the vector will allocated on the heap.
 *
 * Use `StateDimension` and `ControlDimension` to query the actual state or
 * control dimension. Use `StateMemorySize` and `ControlMemorySize` to get the
 * type
 * parameters.
 *
 * @tparam n size of state vector. Can be Eigen::Dynamic.
 * @tparam m size of state vector. Can be Eigen::Dynamic.
 * @tparam T precision of state and control variables
 */
template <int n, int m, class T = double>
class KnotPoint : public StateControlSized<n, m> {
  using StateVector = Vector<n, T>;
  using ControlVector = Vector<m, T>;

 public:
  KnotPoint()
      : StateControlSized<n, m>(n, m),
        x_(StateVector::Zero()),
        u_(ControlVector::Zero()),
        t_(0.0f),
        h_(0.0) {}
  KnotPoint(const StateVector& x, const ControlVector& u, const float t = 0.0,
            const float h = 0.0)
      : StateControlSized<n, m>(x.size(), u.size()),
        x_(x),
        u_(u),
        t_(t),
        h_(h) {}
  KnotPoint(int _n, int _m)
      : StateControlSized<n, m>(_n, _m),
        x_(StateVector::Zero(_n)),
        u_(ControlVector::Zero(_m)),
        t_(0.0f),
        h_(0.0) {}

  // Copy from a knot point of different memory location but same size
  template <int n2, int m2>
  KnotPoint(const KnotPoint<n2, m2>& z2)
      : StateControlSized<n, m>(z2.StateDimension(), z2.ControlDimension()),
        x_(z2.State()),
        u_(z2.Control()),
        t_(z2.GetTime()),
        h_(z2.GetStep()) {}

  // Copy operations
  KnotPoint(const KnotPoint& z)
      : StateControlSized<n, m>(z.n_, z.m_),
        x_(z.x_),
        u_(z.u_),
        t_(z.t_),
        h_(z.h_) {}
  KnotPoint& operator=(const KnotPoint& z) {
    x_ = z.x_;
    u_ = z.u_;
    t_ = z.t_;
    h_ = z.h_;
    this->n_ = z.n_;
    this->m_ = z.m_;
    return *this;
  }

  // Move operations
  KnotPoint(KnotPoint&& z)
      : StateControlSized<n, m>(z.n_, z.m_),
        x_(std::move(z.x_)),
        u_(std::move(z.u_)),
        t_(z.t_),
        h_(z.h_) {}
  KnotPoint& operator=(KnotPoint&& z) {
    x_ = std::move(z.x_);
    u_ = std::move(z.u_);
    t_ = z.t_;
    h_ = z.h_;
    this->n_ = z.n_;
    this->m_ = z.m_;
    return *this;
  }

  static KnotPoint Random() {
    ALTRO_ASSERT(n > 0 && m > 0,
                 "Must pass in size if state or control dimension is unknown "
                 "at compile time.");
    Vector<n> x = Vector<n>::Random();
    Vector<m> u = Vector<m>::Random();
    double t = static_cast<double>(rand() % 100) / 10.0;   // 0 to 10
    double h = static_cast<double>(rand() % 100) / 100.0;  // 0 to 1
    return KnotPoint(x, u, t, h);
  }

  static KnotPoint Random(int state_dim, int control_dim) {
    VectorXd x = VectorXd::Random(state_dim);
    VectorXd u = VectorXd::Random(control_dim);
    double t = static_cast<double>(rand() % 100) / 10.0;   // 0 to 10
    double h = static_cast<double>(rand() % 100) / 100.0;  // 0 to 1
    return KnotPoint(x, u, t, h);
  }

  StateVector& State() { return x_; }
  ControlVector& Control() { return u_; }
  const StateVector& State() const { return x_; }
  const ControlVector& Control() const { return u_; }
  Vector<AddSizes(n,m), T> GetStateControl() const {
    Vector<AddSizes(n,m), T> z;
    z << x_, u_;
    return z;
  }
  float GetTime() const { return t_; }
  float GetStep() const { return h_; }

  /**
   * @brief Check if the knot point is the last point in the trajectory, which
   * has no length and only stores a state vector.
   *
   * @return true if the knot point is a terminal knot point
   */
  bool IsTerminal() const { return h_ == 0; }

  /**
   * @brief Set the knot point to be a terminal knot point, or the last
   * knot point in the trajectory.
   *
   */
  void SetTerminal() {
    h_ = 0;
    u_.setZero();
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const KnotPoint<n, m, T>& z) {
    return os << "x: [" << z.State().transpose() << "], u: ["
              << z.Control().transpose() << "], t=" << z.GetTime()
              << ", h=" << z.GetStep();
  }

 private:
  StateVector x_;
  ControlVector u_;
  float t_;  // time
  float h_;  // time step
};

}  // namespace altro
