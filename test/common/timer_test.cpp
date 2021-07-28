#include <fmt/format.h>
#include <fmt/chrono.h>
#include <gtest/gtest.h>
#include <fstream>
#include <memory>

#include "altro/common/profile_entry.hpp"
#include "altro/common/solver_stats.hpp"
#include "altro/common/timer.hpp"

namespace altro {

void SampleProfile(std::map<std::string, std::chrono::microseconds>* times_ptr) {
  using namespace std::chrono_literals;
  std::map<std::string, std::chrono::microseconds>& times = *times_ptr;
  times["al"] = 43088us;
  times["al/convergence_check"] = 0us;
  times["al/dual_update"] = 36us;
  times["al/ilqr"] = 41558us;
  times["al/ilqr/backward_pass"] = 2024us;
  times["al/ilqr/convergence_check"] = 0us;
  times["al/ilqr/cost"] = 247us;
  times["al/ilqr/expansions"] = 29548us;
  times["al/ilqr/forward_pass"] = 8750us;
  times["al/ilqr/forward_pass/cost"] = 2860us;
  times["al/ilqr/forward_pass/rollout"] = 5743us;
  times["al/ilqr/init"] = 0us;
  times["al/ilqr/stats"] = 274us;
  times["al/init"] = 257us;
  times["al/init/cost"] = 150us;
  times["al/penalty_update"] = 4us;
  times["al/stats"] = 25us;
}

TEST(TimerTest, PrintSummary) {
  using namespace std::chrono_literals;

  std::map<std::string, std::chrono::microseconds> times;
  SampleProfile(&times);

  using EntryPtr = std::shared_ptr<ProfileEntry>;
  std::vector<EntryPtr> entries;
  std::vector<EntryPtr> parents;

  // Add first (top level) entry
  auto iter = times.begin();
  entries.emplace_back(std::make_shared<ProfileEntry>("top", 0us));
  parents.emplace_back(entries.front());

  // Add all the logged entries, assigning parents
  int max_width = 0;
  for (; iter != times.end(); ++iter) {
    entries.emplace_back(std::make_shared<ProfileEntry>(iter->first, iter->second));
    EntryPtr entry = entries.back();
    size_t curlevel = entry->NumLevels();
    if (curlevel >= parents.size()) {
      parents.resize(curlevel + 1);
    }
    parents[curlevel] = entry;
    entry->parent = parents[curlevel - 1];

    // Add up all the base level entries
    if (curlevel == 1) {
      entries.front()->time += entry->time;
    }

    int name_width = 0;
    for (const std::string& part : entry->name) {
      name_width += part.length();
    }
    max_width = std::max(max_width, name_width);
  }
  EXPECT_EQ(entries[0], entries[1]->parent);
  EXPECT_EQ(entries[0]->time, entries[1]->time);
  EXPECT_EQ(entries[2]->parent, entries[1]);

  const int pad = 2;
  max_width += pad;
  std::string header = fmt::format("{1:<{0}}  {2:>8}  {3:>7}  {4:>7}", max_width, "Description",
                                   "Time (us)", "%Total", "%Parent");
  fmt::print("{}\n", header);
  fmt::print("{:-^{}}\n", "", header.length());
  for (size_t i = 1; i < entries.size(); ++i) {
    entries[i]->CalcStats();
    entries[i]->Print(max_width);
  }
  EXPECT_EQ(entries[1]->percent_total, 100);
  EXPECT_EQ(entries[5]->percent_total, 100 * 2024 / 43088);
  EXPECT_EQ(entries[5]->percent_parent, 100 * 2024 / 41558);
}

TEST(TimerTest, PrintToFile) {
  using namespace std::chrono_literals;
  std::map<std::string, std::chrono::microseconds> times;
  SampleProfile(&times);

  // Wrap in scope so that output file is closed when the timer is destroyed
  std::string filename;
  {
    SolverStats stats;
    stats.GetOptions().profiler_output_to_file = true;
    stats.GetOptions().log_directory = LOGDIR;
    stats.GetOptions().profile_filename = "profiler_test.out";
    stats.Reset();
    stats.GetTimer()->PrintSummary(&times);
    filename = stats.GetOptions().log_directory + "/" + stats.GetOptions().profile_filename;
  }
  std::ifstream f(filename.c_str());
  EXPECT_TRUE(f.good());
  ASSERT_TRUE(f.is_open());

  std::string line;

  std::string expected = "Description";
  std::getline(f, line);
  EXPECT_EQ(line.substr(0, expected.length()), expected);

  expected = "------";
  std::getline(f, line);
  EXPECT_EQ(line.substr(0, expected.length()), expected);

  expected = "al";
  std::getline(f, line);
  EXPECT_EQ(line.substr(0, expected.length()), expected);

  expected = "  convergence_check";
  std::getline(f, line);
  EXPECT_EQ(line.substr(0, expected.length()), expected);
}

void ComputeKernel() {
  usleep(500);
}

void NoTimer(int N) {
  for (int i = 0; i < N; ++i) {
    ComputeKernel();
  }
}

void WithTimer(int N, const TimerPtr& timer) {
  Stopwatch sw = timer->Start("base");
  for (int i = 0; i < N; ++i) {
    Stopwatch sw = timer->Start("inner");
    ComputeKernel();
  }
}

TEST(TimerTest, TimerBenchmark) {
  int N = 100;
  auto start = std::chrono::high_resolution_clock::now();
  NoTimer(N);
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::microseconds time_notimer =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  TimerPtr timer = Timer::MakeShared();
  timer->Deactivate();
  start = std::chrono::high_resolution_clock::now();
  WithTimer(N, timer);
  stop = std::chrono::high_resolution_clock::now();
  std::chrono::microseconds time_inactive =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  timer->Activate();
  start = std::chrono::high_resolution_clock::now();
  WithTimer(N, timer);
  stop = std::chrono::high_resolution_clock::now();
  std::chrono::microseconds time_active =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  float inactive_overhead = (100.0 * (time_inactive - time_notimer)) / time_notimer;
  float active_overhead = (100.0 * (time_active - time_notimer)) / time_notimer;
  fmt::print("\nTIMER PERFORMANCE SUMMARY\n");
  fmt::print("No Timer:       {}\n", time_notimer);
  fmt::print("Timer Inactive: {}\n", time_inactive);
  fmt::print("Timer Active:   {}\n", time_active);
  fmt::print("Inactive overhead: {}%\n", inactive_overhead);
  fmt::print("Active overhead:   {}%\n", active_overhead);
  fmt::print("Time per call (inactive): {}\n", (time_inactive - time_notimer) / (N + 1));
  fmt::print("Time per call (active):   {}\n\n", (time_active - time_notimer) / (N + 1));

  // These numbers can vary a lot, so these are some pretty high tolerances for sanity checks
  // Typically they're around 2%, or 10us.
  EXPECT_LT(inactive_overhead, 10);
  EXPECT_LT(active_overhead, 10);
}

}  // namespace altro