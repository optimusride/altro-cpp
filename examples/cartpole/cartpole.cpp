//
// Created by brian on 9/6/22.
//

#include <fmt/core.h>

#include "cartpole.hpp"

void Cartpole::Evaluate(const altro::VectorXdRef& x, const altro::VectorXdRef& u, const float t,
                        Eigen::Ref<Eigen::VectorXd> xdot) {
  (void)t;

  double mc = mass_cart_;
  double mp = mass_pole_;
  double l = length_;
  double g = gravity_;

  double th = x[1];
  double v = x[2];
  double w = x[3];

  // mass terms
  double m11 = mc + mp;
  double m12 = mp * l * std::cos(th);
  double m22 = mp * l * l;
  double detM = m11 * m22 - m12 * m12;

  // generalized torques
  double t1 = mp * w * w * l * std::sin(th) + u[0];
  double t2 = -mp * g * l * std::sin(th);

  // accelerations
  double vdot = (m22 * t1 - m12 * t2) / detM;
  double wdot = (-m12 * t1 + m11 * t2) / detM;

  xdot[0] = v;
  xdot[1] = w;
  xdot[2] = vdot;
  xdot[3] = wdot;
}

void Cartpole::Jacobian(const altro::VectorXdRef& x, const altro::VectorXdRef& u, const float t,
                        Eigen::Ref<Eigen::MatrixXd> jac) {
  (void)t;

  double mc = mass_cart_;
  double mp = mass_pole_;
  double l = length_;
  double g = gravity_;

  double th = x[1];  // theta
  double w = x[3];   // omega (theta dot)

  // mass terms
  double m11 = mc + mp;
  double m12 = mp * l * std::cos(th);
  double m22 = mp * l * l;

  double m12_dt = -mp * l * std::sin(th);

  double detM = m11 * m22 - m12 * m12;
  double detM_dt = -2 * m12 * m12_dt;

  double idetM_dt = -1 / (detM * detM) * detM_dt;

  // generalized torques
  double t1 = mp * w * w * l * std::sin(th) + u[0];
  double t2 = -mp * g * l * std::sin(th);

  double t1_dt = mp * w * w * l * std::cos(th);
  double t1_dw = 2 * mp * w * l * std::sin(th);
  double t2_dt = -mp * g * l * std::cos(th);

  // Derivatives of linear acceleration
  // xddot = (m22 * t1 - m12 * t2) / detM
  double xddot_dt =
      (m22 * t1_dt - m12 * t2_dt - m12_dt * t2) / detM + (m22 * t1 - m12 * t2) * idetM_dt;
  double xddot_dw = (m22 * t1_dw) / detM;
  double xddot_du = m22 / detM;

  // Derivatives of angular acceleration
  // tddot = (-m12 * t1 + m11 * t2) / detM
  double tddot_dt =
      (-m12_dt * t1 - m12 * t1_dt + m11 * t2_dt) / detM + (-m12 * t1 + m11 * t2) * idetM_dt;
  double tddot_dw = (-m12 * t1_dw) / detM;
  double tddot_du = -m12 / detM;

  // Assign Jacobian
  jac.setZero();
  jac(0, 2) = 1;
  jac(1, 3) = 1;
  jac(2, 1) = xddot_dt;
  jac(2, 3) = xddot_dw;
  jac(2, 4) = xddot_du;
  jac(3, 1) = tddot_dt;
  jac(3, 3) = tddot_dw;
  jac(3, 4) = tddot_du;
}

void Cartpole::Hessian(const altro::VectorXdRef& x, const altro::VectorXdRef& u, const float t,
                       const altro::VectorXdRef& b, Eigen::Ref<Eigen::MatrixXd> hess) {
  (void)t;

  double mc = mass_cart_;
  double mp = mass_pole_;
  double l = length_;
  double g = gravity_;

  double th = x[1];  // theta
  double w = x[3];   // omega (theta dot)

  // mass terms
  double m11 = mc + mp;
  double m12 = mp * l * std::cos(th);
  double m22 = mp * l * l;

  double m12_dt = -mp * l * std::sin(th);
  double m12_ddt = -mp * l * std::cos(th);

  double detM = m11 * m22 - m12 * m12;
  double detM_dt = -2 * m12 * m12_dt;
  double detM_ddt = -2 * m12_dt * m12_dt - 2 * m12 * m12_ddt;

  double idetM_dt = -1 / (detM * detM) * detM_dt;
  double idetM_ddt = 2 / (detM * detM * detM) * detM_dt * detM_dt - 1 / (detM * detM) * detM_ddt;

  // generalized torques
  double t1 = mp * w * w * l * std::sin(th) + u[0];
  double t1_dt = mp * w * w * l * std::cos(th);
  double t1_ddt = -mp * w * w * l * std::sin(th);
  double t1_dtdw = 2 * mp * w * l * std::cos(th);

  double t1_dw = 2 * mp * w * l * std::sin(th);
  double t1_ddw = 2 * mp * l * std::sin(th);

  double t2 = -mp * g * l * std::sin(th);
  double t2_dt = -mp * g * l * std::cos(th);
  double t2_ddt = mp * g * l * std::sin(th);

  // Jacobian-vector product derivatives

  // xddot = (m22 * t1 - m12 * t2) / detM
  // xddot_dt = (m22 * t1_dt - m12 * t2_dt - m12_dt * t2) / detM + (m22 * t1 - m12 * t2) * idetM_dt;
  double xddot_ddt = (m22 * t1_ddt - 2 * m12_dt * t2_dt - m12 * t2_ddt - m12_ddt * t2) / detM
                     + 2 * (m22 * t1_dt - m12 * t2_dt - m12_dt * t2) * idetM_dt
                     + (m22 * t1 - m12 * t2) * idetM_ddt;
  double xddot_dtdw = (m22 * t1_dtdw) / detM + (m22 * t1_dw) * idetM_dt;
  double xddot_dtdu = m22 * idetM_dt;

  // xddot_dw = (m22 * t1_dw) / detM;
  double xddot_ddw = m22 * t1_ddw / detM;

  // xddot_du = m22 / detM;
  double xddot_dudt = m22 * idetM_dt;

  // tddot = (-m12 * t1 + m11 * t2) / detM
  // tddot_dt = (-m12_dt * t1 - m12 * t1_dt + m11 * t2_dt) / detM + (-m12 * t1 + m11 * t2) * idetM_dt;
  double tddot_ddt = (-m12_ddt * t1 - 2 * m12_dt * t1_dt - m12 * t1_ddt + m11 * t2_ddt) / detM
                     + 2 * (-m12_dt * t1 - m12 * t1_dt + m11 * t2_dt) * idetM_dt
                     + (-m12 * t1 + m11 * t2) * idetM_ddt;
  double tddot_dtdw = (-m12_dt * t1_dw - m12 * t1_dtdw) / detM + (-m12 * t1_dw) * idetM_dt;
  double tddot_dtdu = -m12_dt / detM - m12 * idetM_dt;

  // tddot_dw = (-m12 * t1_dw) / detM;
  double tddot_ddw = (-m12 * t1_ddw) / detM;

  // tddot_du = -m12 / detM;
  double tddot_dudt = -m12_dt / detM - m12 * idetM_dt;

  // Hessian
  hess.setZero();
  hess(1,1) = xddot_ddt * b[2] + tddot_ddt * b[3];
  hess(1,3) = xddot_dtdw * b[2] + tddot_dtdw * b[3];
  hess(1,4) = xddot_dtdu * b[2] + tddot_dtdu * b[3];

  hess(3,1) = xddot_dtdw * b[2] + tddot_dtdw * b[3];
  hess(3,3) = xddot_ddw * b[2] + tddot_ddw * b[3];

  hess(4,1) = xddot_dudt * b[2] + tddot_dudt * b[3];

}