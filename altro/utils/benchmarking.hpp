#include <vector>
#include <chrono>
#include <algorithm>
#include <fmt/format.h>
#include <fmt/chrono.h>

namespace altro {
namespace utils {


/**
 * @brief Records the output of a simple benchmark run, recording statistical 
 * information on the timing results.
 * 
 * @tparam Duration Time measurement. Should be a `std::chrono::duration`, usually
 * `std::chrono::microseconds`. At a minimum, should return a std::ratio via the
 * `period` field.
 */
template <class Duration>
struct BenchmarkResults {
  // Use a floating-point representation of the desired duration.
  using time_t = std::chrono::duration<double, typename Duration::period>;
  time_t mean;
  time_t median;
  time_t std;
  time_t max;
  time_t min;
  int samples;

  /**
   * @brief Creates a BenchmarkResults from a vector of time samples.
   * 
   * @param times Samples collected from a benchmarking run.
   */
  static BenchmarkResults Calculate(std::vector<time_t>& times);
  
  /**
   * @brief Print a summary of the benchmark results
   * 
   */
  void Print();
};

template <class Duration>
BenchmarkResults<Duration> BenchmarkResults<Duration>::Calculate(std::vector<time_t>& times) {
  // Sort for median
  std::sort(times.begin(), times.end());
  int samples = times.size();
  time_t median;
  if ((samples % 2) == 1) {  // odd
    median = times[samples / 2];
  } else {
    median = (times[samples / 2] + times[(samples / 2) - 1]) / 2;
  }
  time_t sum = times[0];
  time_t max = times[0];
  time_t min = times[0];
  for (int i = 1; i < samples; ++i) {
    sum += times[i];
    max = std::max(times[i], max);
    min = std::min(times[i], min);
  }
  time_t mean = sum / samples;
  time_t std = time_t(std::pow((times[0] - mean).count(), 2));
  for (int i = 1; i < samples; ++i) {
    std += time_t(std::pow((times[i] - mean).count(), 2));
  }
  std = time_t(std::sqrt((std / samples).count()));
  BenchmarkResults<Duration> res{mean, median, std, max, min, samples};
  return res;
}

template <class Duration>
void BenchmarkResults<Duration>::Print() {
  fmt::print("Mean:    {}\n", mean);
  fmt::print("Median:  {}\n", median);
  fmt::print("Std:     {}\n", std);
  fmt::print("Max:     {}\n", max);
  fmt::print("Min:     {}\n", min);
  fmt::print("Samples: {}\n", samples);
}


static constexpr int kDefaultSamples = 100;

/**
 * @brief Benchmark a function by running it many times and recording the 
 * total time.
 * 
 * @tparam Duration Time measurement. Should be a `std::chrono::duration`, usually
 * `std::chrono::microseconds`. At a minimum, should return a std::ratio via the
 * `period` field.
 * @tparam Function Any object that supports the () operator with zero arguments.
 * @param f Function to benchmark.
 * @param Nsamples Number of times the function f should be run.
 * @return BenchmarkResults<Duration> Summary of benchmark results, containing 
 * statistical info like mean, median, max, and min times.
 */
template <class Duration, class Function>
BenchmarkResults<Duration> Benchmark(Function f, int Nsamples = kDefaultSamples) {
  using time_t = typename BenchmarkResults<Duration>::time_t; 
  std::vector<time_t> times;
  times.reserve(Nsamples);
  for (int i = 0; i < Nsamples; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto stop = std::chrono::high_resolution_clock::now();
    times.emplace_back(std::chrono::duration_cast<time_t>(stop - start));
  }
  return BenchmarkResults<Duration>::Calculate(times);
}


}  // namespace utils
}  // namespace altro