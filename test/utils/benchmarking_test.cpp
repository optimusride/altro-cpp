#include <gtest/gtest.h>

#include "altro/utils/benchmarking.hpp"

namespace altro {
namespace utils {

TEST(BenchmarkingTest, Basic) {
  using time_t = std::chrono::duration<double, std::micro>; 
  using namespace std::chrono_literals;

  std::vector<time_t> times;
  times.emplace_back(1ms);
  times.emplace_back(2ms);
  times.emplace_back(3ms);
  times.emplace_back(4ms);
  times.emplace_back(10ms);
  BenchmarkResults<time_t> res = BenchmarkResults<time_t>::Calculate(times);
  EXPECT_DOUBLE_EQ(res.mean.count(), 4000.0);
  EXPECT_DOUBLE_EQ(res.median.count(), 3000.0);
  EXPECT_DOUBLE_EQ(res.max.count(), 10000.0);
  EXPECT_DOUBLE_EQ(res.min.count(), 1000.0);
  EXPECT_DOUBLE_EQ(res.std.count(), std::sqrt(10)*1000);
  EXPECT_EQ(res.samples, 5);

  res.Print();
}

TEST(BenchmarkingTest, Benchmark) {
  using time_t = std::chrono::microseconds;
  using namespace std::chrono_literals;

  auto f = []() { usleep(1150); };
  BenchmarkResults<time_t> res = Benchmark<time_t>(f);
  EXPECT_GT(res.min, 1150us);
  EXPECT_GT(res.max, res.min);
  EXPECT_GT(res.mean, res.min);
}

}  // namespace utils
}  // namespace altro