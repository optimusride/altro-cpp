#pragma once

#include <string>

#include <fmt/format.h>

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
  using StateVector = VectorN<n, T>;
  using ControlVector = VectorN<m, T>;

 public:
  KnotPoint()
      : StateControlSized<n, m>(n, m),
        x_(StateVector::Zero()),
        u_(ControlVector::Zero()) {}
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
        u_(ControlVector::Zero(_m)) {}

  // Copy from a knot point of different memory location but same size
  template <int n2, int m2>
  KnotPoint(const KnotPoint<n2, m2>& z2)  // NOLINT(google-explicit-constructor)
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
  KnotPoint(KnotPoint&& z) noexcept
      : StateControlSized<n, m>(z.n_, z.m_),
        x_(std::move(z.x_)),
        u_(std::move(z.u_)),
        t_(z.t_),
        h_(z.h_) {}
  KnotPoint& operator=(KnotPoint&& z) noexcept {
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
    return Random(n, m);
  }

  static KnotPoint Random(int state_dim, int control_dim) {
    VectorN<n> x = VectorN<n>::Random(state_dim);
    VectorN<m> u = VectorN<m>::Random(control_dim);
    const double max_time = 10.0;
    const double max_h = 1.0;
    const int resolution = 100;
    double t = UniformRandom(max_time, resolution);
    double h = UniformRandom(max_h, resolution);
    return KnotPoint(x, u, t, h);
  }

  StateVector& State() { return x_; }
  ControlVector& Control() { return u_; }
  const StateVector& State() const { return x_; }
  const ControlVector& Control() const { return u_; }
  VectorN<AddSizes(n,m), T> GetStateControl() const {
    VectorN<AddSizes(n,m), T> z;
    z << x_, u_;
    return z;
  }
  float GetTime() const { return t_; }
  float GetStep() const { return h_; }
  void SetTime(float t) { t_ = t; }
  void SetStep(float h) { h_ = h; }

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
    return os << z.ToString();
  }

  /**
   * @brief Create a string containing a print out of all the states and controls
   * in a single line.
   * 
   * @param width Controls the width of each numerical field
   * @return std::string 
   */
  std::string ToString(int width = 9) const {
    std::string out;
    out += fmt::format("x: [");
    for (int i = 0; i < this->n_; ++i) {
      out += fmt::format("{1: > {0}.3g} ", width, State()(i));
    }
    out += fmt::format("] u: [");
    for (int i = 0; i < this->m_; ++i) {
      out += fmt::format("{1: > {0}.3g} ", width, Control()(i));
    }
    out += fmt::format("] t={:4.2}, h={:4.2}", GetTime(), GetStep());
    return out;
  }

 private:
  static double UniformRandom(double upper, int resolution) { 
    return upper * static_cast<double>(rand() % resolution) / static_cast<double>(resolution); 
  }

  StateVector x_;
  ControlVector u_;
  float t_ = 0.0;  // time
  float h_ = 0.0;  // time step
};

}  // namespace altro
