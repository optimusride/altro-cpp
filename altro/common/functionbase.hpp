#pragma once

#include <type_traits>

#include "altro/eigentypes.hpp"
#include "altro/utils/utils.hpp"

namespace altro {

// clang-format off
/**
 * @brief Represents a generic vector-valued function of the form 
 * \f[
 *   out = f(x, u)`
 * \f]
 *
 * At a minimum, the function must have a well-defined Jacobian. Second-order
 * information is provided by defining the Jacobian of the
 * Jacobian-transpose-vector product.
 *
 * The implemented derivatives can be checked using finite differencing using
 * `CheckJacobian` and `CheckHessian`. These functions can provided sample inputs,
 * or else will generate random inputs if none are provided.
 *
 * # Interface
 * To implement this interface, the user must specify the following:
 * - `int StateDimension() const` - number of states (length of x)
 * - `int ControlDimension() const` - number of controls (length of u)
 * - `int OutputDimension() const` - size of output (length of out).
 * - `void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<Eigen::VectorXd> out)`
 * - `void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> out)`
 * - `void Hessian(const VectorXdRef& x, const VectorXdRef& u, const VectorXdRef& b,
 * Eigen::Ref<Eigen::MatrixXd> hess)` - optional
 * - `bool HasHessian() const` - Specify if the Hessian is implemented
 *
 * Where we use the following Eigen type alias:
 * 
 *      using VectorXdRef = Eigen::Ref<const Eigen::VectorXd>
 *
 * The user also has the option of defining the static constants:
 * 
 *      static constexpr int NStates
 *      static constexpr int NControls
 *      static constexpr int NOutputs
 *
 * which can be used to provide compile-time size information. These can value
 * can be queried on run-time types using the `StateMemorySize`, `ControlMemorySize`,
 * and `OutputMemorySize` functions.
 */
// clang-format off
class FunctionBase {
 public:
  virtual ~FunctionBase() = default;

  static constexpr int NStates = Eigen::Dynamic;
  static constexpr int NControls = Eigen::Dynamic;
  static constexpr int NOutputs = Eigen::Dynamic;

  virtual int StateDimension() const { return 0; }
  virtual int ControlDimension() const { return 0; }
  virtual int OutputDimension() const = 0;

  virtual void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> out) = 0;
  virtual void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> jac) = 0;
  virtual void Hessian(const VectorXdRef& x, const VectorXdRef& u, const VectorXdRef& b,
                       Eigen::Ref<MatrixXd> hess) {  // NOLINT(performance-unnecessary-value-param)
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(u);
    ALTRO_UNUSED(b);
    ALTRO_UNUSED(hess);
  }
  virtual bool HasHessian() const = 0;

  bool TestCheck(const VectorXdRef& x, const VectorXdRef& u);

  bool CheckJacobian(double eps = kDefaultTolerance, bool verbose = false);
  bool CheckJacobian(const VectorXdRef& x, const VectorXdRef& u, double eps = kDefaultTolerance,
                     bool verbose = false);
  bool CheckHessian(double eps = kDefaultTolerance, bool verbose = false);
  bool CheckHessian(const VectorXdRef& x, const VectorXdRef& u, const VectorXdRef& b,
                    double eps = kDefaultTolerance, bool verbose = false);

 protected:
  static constexpr double kDefaultTolerance = 1e-4;
};

// clang-format on
/**
 * @brief Represents an abstract scalar-valued function
 *
 * A specialization of the `FunctionBase` interface to scalar-valued functions.
 *
 * For notational convenience, we define the gradient to be a column-vector
 * of the first derivative of a scalar function. This is the transpose of the
 * corresponding Jacobian.
 *
 * The Hessian is then just the Jacobian of the gradient.
 *
 * The `CheckGradient` and `CheckHessian` functions can be used to verify the
 * derivatives implemented by the user. Note that when passing parameters to
 * `CheckHessian`, the `b` vector argument still needs to be specified, even
 * though this is not required in the scalar function interface.
 *
 * # Interface
 * The user must define the following functions:
 * - `int StateDimension() const` - number of states (length of x)
 * - `int ControlDimension() const` - number of controls (length of u)
 * - `double Evaluate(const VectorXdRef& x, const VectorXdRef& u)`
 * - `void Gradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<Eigen::VectorXd> out)`
 * - `void Hessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<Eigen::MatrixXd> hess)`
 * - `bool HasHessian() const` - Specify if the Hessian is implemented - optional (assumed to be
 * true)
 *
 * Where we use the following Eigen type alias:
 * 
 *      using VectorXdRef = Eigen::Ref<const Eigen::VectorXd>
 *
 * The user also has the option of defining the static constants:
 * 
 *      static constexpr int NStates
 *      static constexpr int NControls
 *
 * which can be used to provide compile-time size information. These values
 * can be queried on run-time types using the `StateMemorySize`, and `ControlMemorySize`
 * functions.
 *
 */
// clang-format off
class ScalarFunction : public FunctionBase {
 public:
  static const int NOutputs = 1;
  int OutputDimension() const override { return 1; }

  // New Interface
  virtual double Evaluate(const VectorXdRef& x, const VectorXdRef& u) = 0;
  virtual void Gradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> grad) = 0;
  virtual void Hessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> hess) = 0;

  // Pass parent interface to new interface
  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> out) override {
    ALTRO_ASSERT(out.size() == 1, "Output must be of size 1 for scalar functions");
    out(0) = Evaluate(x, u);
  }

  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> jac) override {
    ALTRO_ASSERT(jac.rows() == 1, "Jacobian of a scalar function must have a single row.");
    // Reinterpret the 1xN Jacobian as an Nx1 column vector
    Eigen::Map<VectorNd<jac.ColsAtCompileTime>> grad(jac.data(), jac.cols());
    Gradient(x, u, grad);
  }

  void Hessian(const VectorXdRef& x, const VectorXdRef& u, const VectorXdRef& b,
               Eigen::Ref<MatrixXd> hess) override {
    ALTRO_ASSERT(b.size() == 1 && b.isApproxToConstant(1),
                 "The b vector for scalar Hessians must be a vector of a single 1.");
    ALTRO_UNUSED(b);
    Hessian(x, u, hess);
  }
  bool HasHessian() const override { return true; }

  // Derivative checking
  bool CheckGradient(double eps = kDefaultTolerance, bool verbose = false);
  bool CheckGradient(const VectorXdRef& x, const VectorXdRef& u, double eps = kDefaultTolerance,
                     bool verbose = false);
};

}  // namespace altro
