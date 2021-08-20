#include "altro/constraints/constraint.hpp"

#include <iostream>
#include <fmt/format.h>
#include <fmt/ostream.h>

namespace altro {
namespace constraints {

std::string ConstraintInfo::ToString(int precision) const {
  Eigen::IOFormat format(precision, 0, ", ", "", "", "", "[", "]");
  return fmt::format("{} at index {}: {}", label, index, violation.format(format));
}

std::ostream& operator<<(std::ostream& os, const ConstraintInfo& coninfo) {
  return os << coninfo.ToString();
}

}  // namespace constraints
} // namespace altro
