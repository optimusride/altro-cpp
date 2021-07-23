#include "altro/common/solver_logger.hpp"

namespace altro {


void SolverLogger::PrintHeader() {
  if (cur_level_ == LogLevel::kSilent) {
    return;
  }
  int total_width = 0;
  bool any_active = false;
  for (const std::string* title : order_) {
    LogEntry& entry = entries_[*title];
    entry.PrintHeader(cur_level_, header_color_);
    if (entry.IsActive(cur_level_)) {
      any_active = true;
      fmt::print(" ");
      total_width += entry.GetWidth() + 1;
    }
  }
  if (any_active) {
    fmt::print("\n");
    fmt::print(fg(header_color_), "{0:-^{1}s}\n", "", total_width);
  }
}

void SolverLogger::PrintData() {
  if (cur_level_ == LogLevel::kSilent) {
    return;
  }
  bool any_active = false;
  for (const std::string* title : order_) {
    LogEntry& entry = entries_[*title];
    entry.Print(cur_level_);
    if (entry.IsActive(cur_level_)) {
      any_active = true;
      fmt::print(" ");
    }
  }
  if (any_active) {
    fmt::print("\n");
  }
}

void SolverLogger::Print() {
  if ((count_ % frequency_) == 0) {
    count_ = 0;
    PrintHeader();
  }
  PrintData();
  count_++;
}


void SolverLogger::Clear() {
  count_ = 0;
  for (auto& kv : entries_) {
    kv.second.Clear();
  }
}

}  // namespace altro