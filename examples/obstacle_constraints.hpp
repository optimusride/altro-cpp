#pragma once

#include <cmath>
#include <vector>

#include "altro/constraints/constraint.hpp"
#include "altro/eigentypes.hpp"
#include "altro/utils/utils.hpp"

namespace altro {
namespace examples {

struct Circle {
  explicit Circle(double px, double py, double radius) : x(px), y(py), r(radius) {}
  double x;
  double y;
  double r;

  /**
   * @brief Compute the squared distance function.
   * 
   * ## Mathematical Formulation
   * The squared distance function is defined to be:
   * 
   * \f[
   * || p_x - c_x ||_2^2 - r^2 ||
   * \f]
   * where \f$ p_x \f$ is the position of the robot, \f$ c_x \f$ is the center 
   * of the circle, and \f$ f \f$ is the radius of the circle.
   * 
   * @tparam T 
   * @param px X-coordinate of the point
   * @param py Y-coordinate of the point
   * @return T Squared distance between the point and the circle.
   */
  template <class T>
  T Distance2(const T px, const T py) const {
    return std::pow(px - x, 2) + std::pow(py - y, 2) - std::pow(r, 2);
  }

  template <class T>
  T Distance(const T px, const T py) const {
    return std::sqrt(std::pow(px - x, 2) + std::pow(py - y, 2)) - r;
  }

  /**
   * @brief Gradient of the squared distance function
   * 
   * @tparam T 
   * @param[in] px X-coordinate of point
   * @param[in] py Y-corredinate of the point
   * @param[out] grad Gradient of the signed distance function
   */
  template <class T>
  void Distance2Gradient(const T px, const T py, T* grad) const {
    grad[0] = 2 * (px - x);
    grad[1] = 2 * (py - y);
  }
};

/**
 * @brief Constraint that keeps the position away from a list of different 
 * circular regions. Assumes that the robot is a point (i.e. you need to add a 
 * collision buffer yourself) 
 * 
 */
class CircleConstraint : public constraints::Constraint<constraints::NegativeOrthant> {
 public:

  /**
   * @brief Add a circular obstacle to avoid. Arguments are passed to the constructor 
   * of the Circle object.
   */
  template <class... Args>
  void AddObstacle(Args&& ...args) { 
    obstacles_.emplace_back(std::forward<Args>(args)...);
  }

  /**
   * @brief Set which state indices correspond to the x and y positions.
   * 
   * By default, the x-coordinate is assume to be the first state, and the 
   * y-coordinate to be the second.
   * 
   * @param x_index Index of the x-coordinate (0 <= x_index < state_dimension).
   * @param y_index Index of the y-coordinate (0 <= y_index < state_dimension).
   */
  void SetXYIndices(int x_index, int y_index) { 
    ALTRO_ASSERT(x_index >= 0, "X index must be non-negative");
    ALTRO_ASSERT(y_index >= 0, "Y index must be non-negative");
    x_index_ = x_index; 
    y_index_ = y_index; 
  }

  int OutputDimension() const override { return obstacles_.size(); }

  void Evaluate(const VectorXdRef& x, const VectorXdRef& /*u*/,
                Eigen::Ref<VectorXd> c) override {
    const double px = x(x_index_);
    const double py = x(y_index_);
    for (size_t i = 0; i < obstacles_.size(); ++i) {
      // store negative since it must be in the negative orthant
      c(i) = -obstacles_[i].Distance2(px, py);
    }
  }

  void Jacobian(const VectorXdRef& x, const VectorXdRef& /*u*/,
                JacobianRef jac) override {
    const double px = x(x_index_);
    const double py = x(y_index_);
    for (size_t i = 0; i < obstacles_.size(); ++i) {
      // TODO(bjackson): Store Jacobians in row-major format so that the 
      // raw columns can be passed to each obstacle to compute the gradient
      const double cx = obstacles_[i].x;
      const double cy = obstacles_[i].y;
      jac(i, 0) = 2 * (cx - px);
      jac(i, 1) = 2 * (cy - py);
    }
  }

 private:
  int x_index_ = 0;
  int y_index_ = 1;
  std::vector<Circle> obstacles_;
};

}  // namespace examples
}  // namespace altro