#include <fmt/format.h>
#include <memory>

#include "altro/common/timer.hpp"

namespace altro {

/**
 * @brief An entry in the profiler summary table
 * 
 * Records the hierarchical structure of the profiling results. When parsed 
 * correctly, these should be arranged in a tree-like structure that is used
 * to calculate the total time and time relative to the parents.
 * 
 * This is used internally by the Timer class to print the results via PrintSummary.
 * 
 */
struct ProfileEntry : public std::enable_shared_from_this<ProfileEntry> {
  using time_t = std::chrono::microseconds;
  using Ptr = std::shared_ptr<ProfileEntry>;
  ProfileEntry(const std::string& fullname, time_t time);

  std::vector<std::string> name;  // split name stack e.g. "al/ilqr/cost" => ["al", "ilqr", "cost"]
  time_t time;                    // total time in the function
  int percent_total;              // percentage of total time (e.g. cost_time / al_time)
  int percent_parent;             // percentage of the parent time (e.g. cost_time / ilqr_time)
  Ptr parent = nullptr;           // Pointer to the parent ProfileEntry (e.g. ilqr is the parent of cost).

  size_t NumLevels() const { return name.size(); }
  Ptr GetRoot();                  // Get the entry that has no parents (stores total recorded time).
  void CalcStats();               // Calculate percent_total and percent_parent..
  void Print(FILE* io, int width);
  void Print(int width);
};

}  // namespace altro