#pragma once

#include <fmt/format.h>
#include <fmt/ostream.h>

#include "altro/common/solver_options.hpp"
#include "altro/common/solver_logger.hpp"
#include "altro/common/timer.hpp"
#include "altro/utils/utils.hpp"

namespace altro {



/**
 * @brief Describes the current state of the solver
 *
 * Used to describe if the solver successfully solved the problem or to
 * provide a reason why it was unsuccessful.
 */
enum class SolverStatus {
  kSolved = 0,
  kUnsolved = 1,
  kStateLimit = 2,
  kControlLimit = 3,
  kCostIncrease = 4,
  kMaxIterations = 5,
  kMaxOuterIterations = 6,
  kMaxInnerIterations = 7,
  kMaxPenalty = 8,
  kBackwardPassRegularizationFailed = 9,
};

/**
 * @brief Holds statistics recorded during the solve
 * 
 * This class also provides functionality to output data to the terminal,
 * with varying levels of verbosity.
 * 
 * Any new data fields to be recorded and printed via the logger should be 
 * "registered" with the logger in DefaultLogger(). Not all data fields in this
 * struct need to be "registered" with the logger.
 * 
 */
class SolverStats {
 public:
  SolverStats() : timer_(Timer::MakeShared()) { 
    DefaultLogger(); 
  }

  double initial_cost = 0.0;
  int iterations_inner = 0;
  int iterations_outer = 0;
  int iterations_total = 0;
  std::vector<double> cost;
  std::vector<double> alpha;
  std::vector<double> improvement_ratio;  // ratio of actual to expected cost decrease
  std::vector<double> gradient;
  std::vector<double> cost_decrease;
  std::vector<double> regularization;
  std::vector<double> violations;     // The maximum constraint violation for each AL iteration
  std::vector<double> max_penalty;    // Maximum penalty parameter for each AL iteration

  /**
   * @brief Set the cost, constraint, and gradient tolerances for the output.
   * 
   * Any logged valued below these tolerances will be printed in green.
   * 
   * @param cost Cost tolerance, or the change in cost between iterations.
   * @param viol Maximum constraint violation.
   * @param grad Max norm of the gradient.
   */
  void SetTolerances(const double& cost, const double& viol, const double& grad);

  /**
   * @brief Set the capacity of the internally-stored vectors
   * 
   * @param n Size to allocate, generally equal to the maximum number of iterations.
   */
  void SetCapacity(int n);

  /**
   * @brief Reset the statistics, clearing all the vectors and resetting all
   * counters to zero.
   * 
   */
  void Reset();

  /**
   * @brief Set the verbosity level for the console logger
   * 
   * @param level 
   */
  void SetVerbosity(const LogLevel level) { logger_.SetLevel(level); }

  /**
   * @brief Get the verbosity of the console logger
   * 
   * @return Current verbosity level
   */
  LogLevel GetVerbosity() const { return logger_.GetLevel(); }
  SolverLogger& GetLogger() { return logger_; }
  const SolverLogger& GetLogger() const { return logger_; }
  TimerPtr& GetTimer() { return timer_; }
  const TimerPtr& GetTimer() const { return timer_; }
  SolverOptions& GetOptions() { return opts_; }
  const SolverOptions& GetOptions() const { return opts_; }
  std::string ProfileOutputFile();

  /**
   * @brief Print the last iteration to the console
   * 
   */
  void PrintLast() { logger_.Print(); }

  /**
   * @brief Log the data
   * 
   * This command does 2 things:
   * 1) It attempts to send the value to logger, where it will formatted into a 
   * string and stored for later printing.
   * 2) It stashes the value in the corresponding storage vector, always saving 
   * to the last element of the vector.
   * 
   * This function will overwrite the previous value if called multiple times 
   * between calls to NewIteration().
   * 
   * @tparam T data type of the value to be logged. Should be consistent with the data field.
   * @param title Title of the value to be logged. This is the same as the header printed in 
   * the console.
   * @param value Value to be logged.
   */
  template <class T>
  void Log(const std::string& title, T value) {
    logger_.Log(title, value);
    SetData(title, value);
  }

  /**
   * @brief Advance the data forward by one iteration, effectively saving 
   * all the current data.
   * 
   */
  void NewIteration();

  //  TODO(bjackson): Make this private to not confuse the user
  // Requires a friend relationship with SolverOptions
  void ProfilerOutputToFile(bool flag);

 private:

  /**
   * @brief Saves the data in the corresponding vector.
   * 
   * @tparam T data type of the value to be logged.
   * @param title Title of the log entry. Corresponds with the header printed 
   * to the console.
   * @param value Value to be saved.
   */
  template <class T>
  void SetData(const std::string& title, T value);

  /**
   * @brief Create an entry in floats_ that maps the title key to one of the publically-accessible
   * vectors. 
   * 
   * This method is called when setting up the SolverStats object.
   * 
   * @tparam T Data type of the entry field (doubles or ints).
   * @param entry Entry field to which the data vector is to be associated.
   * @param data One of the publically-accessible vectors stored in this class.
   */
  template <class T>
  void SetPtr(const LogEntry& entry, std::vector<T>& data) {
    ALTRO_UNUSED(entry);
    ALTRO_UNUSED(data);
  }

  /**
   * @brief Create the default logging fields
   * 
   */
  void DefaultLogger();

  std::unordered_map<std::string, std::vector<double>*> floats_;

  int len_ = 0;
  SolverLogger logger_;
  TimerPtr timer_;
  SolverOptions opts_;
};

template <class T>
void SolverStats::SetData(const std::string& title, T value) {
  // Automatically register the first iteration to prevent accessing empty vectors
  if (len_ == 0) {
    NewIteration();
  }
  // Search for the title
  auto search_float = floats_.find(title);
  if (search_float != floats_.end()) {
    search_float->second->back() = value;
  }
}

}  // namespace altro