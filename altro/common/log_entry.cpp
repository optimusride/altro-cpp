#include "altro/common/log_entry.hpp"

#include "altro/utils/assert.hpp"

namespace altro {

LogEntry& LogEntry::SetLowerBound(double lb, fmt::color color) {
  ALTRO_ASSERT(lb <= upper_, "Lower bound must be less than or equal to the upper bound.");
  bounded_ = true;
  lower_ = lb;
  color_lower_ = color;
  return *this;
}

LogEntry& LogEntry::SetUpperBound(double ub, fmt::color color) {
  ALTRO_ASSERT(ub >= lower_, "Upper bound must be greater than or equal to the lower bound.");
  bounded_ = true;
  upper_ = ub;
  color_upper_ = color;
  return *this;
}

LogEntry& LogEntry::SetWidth(const int width) {
  width_ = width;
  return *this;
}

LogEntry& LogEntry::SetLevel(const LogLevel level) {
  level_ = level;
  return *this;
}

LogEntry& LogEntry::SetName(const std::string& name) {
  name_ = name;
  return *this;
}

LogEntry& LogEntry::SetType(const EntryType type) {
  type_ = type;
  return *this;
}

}  // namespace altro