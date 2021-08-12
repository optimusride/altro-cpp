#include "altro/common/profile_entry.hpp"

namespace altro {

namespace {

std::vector<std::string> split(const std::string& name) {
  std::vector<std::string> parts;
  std::string delim = "/";
  size_t pos = 0;
  size_t start = 0;
  while ((pos = name.find(delim, start)) != std::string::npos) {
    parts.emplace_back(name.substr(start, pos - start));
    start = pos + delim.length();
  }
  pos = name.length();
  parts.emplace_back(name.substr(start, pos - start));
  return parts;
}

} // namespace

ProfileEntry::ProfileEntry(const std::string& fullname, time_t time)
    : name(split(fullname)), time(time) {}

ProfileEntry::Ptr ProfileEntry::GetRoot() {
  Ptr root = shared_from_this();
  while (root->parent) {
    root = root->parent;
  }
  return root;
}

void ProfileEntry::CalcStats() {
  time_t total_time = GetRoot()->time;
  time_t parent_time = parent ? parent->time : time;
  int time_us = time.count();
  int total_us = total_time.count();
  const int percent_scaling = 100;
  if (total_us == 0) {
    percent_total = -1;
  } else {
    percent_total = percent_scaling * time_us / total_us;
  }
  if (parent_time.count() == 0) {
    percent_parent = -1;
  } else {
    percent_parent = percent_scaling * time / parent_time;
  }
}

void ProfileEntry::Print(const int width) {
  Print(stdout, width);
}

void ProfileEntry::Print(FILE* io, const int width) {
  int indent = 0;
  for (size_t i = 0; i < name.size() - 1; ++i) {
    indent += name[i].length();
  }
  std::string indented_name = fmt::format("{0: ^{1}}{2:}", "", indent, name.back());
  fmt::print(io, "{1:<{0}}  {2:>8}  {3:>7}  {4:>7}\n", width, indented_name, time.count(),
             percent_total, percent_parent);
  (void)width;
}

}  // namespace altro