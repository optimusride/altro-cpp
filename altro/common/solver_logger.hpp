// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <fmt/color.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <limits>
#include <map>
#include <unordered_map>

#include "altro/utils/assert.hpp"
#include "altro/common/log_entry.hpp"

namespace altro {

/**
 * @brief Provides a tabular-like logging output
 *
 * The logger contains several different data entries (or fields / columns),
 * which can filled out between calls to the print function. The verbosity
 * can be modified at run time. It also supports simple bounds-based conditional
 * formatting on numerical data entries.
 *
 * The key for each entry is the title specified by the LogEntry (the name
 * printed in the header).
 * 
 * # Example
 * The follow example creates a simple logger with interger and floating point 
 * entries, sets the header color, the header print frequency, and logs and prints
 * some data. 
 * @code {.cpp}
 * // Create logger and add entries
   SolverLogger logger;
   logger.AddEntry(0, "iters", "{:>4}", LogEntry::kInt).SetWidth(6).SetLevel(LogLevel::kOuterDebug);
   logger.AddEntry(1, "cost", "{:.4g}").SetLevel(LogLevel::kOuter);

   // Set options
   logger.SetHeaderColor(fmt::color::cyan);
   logger.SetFrequency(5);
   logger.SetLevel(LogLevel::kInner);

   // Log data
   logger.Log("iters", 1);
   logger.Log("cost", 10.0);
   logger.Print();
   logger.Log("iters", 2);
   logger.Print();  // keeps "10" in "cost" column
   logger.Clear() 
 * @endcode
 */
class SolverLogger {
 public:
  /**
   * @brief Construct a new Solver Logger object
   *
   * @param level Verbosity level. A level of 0 will not print anything.
   */
  explicit SolverLogger(const LogLevel level = LogLevel::kSilent) : cur_level_(level) {}

  LogLevel GetLevel() const { return cur_level_; }
  LogEntry& GetEntry(const std::string& title) { return entries_[title]; }
  int NumEntries() { return entries_.size(); }

  /*************************** Iteration **************************************/
  using iterator = std::unordered_map<std::string, LogEntry>::iterator;
  using const_iterator = std::unordered_map<std::string, LogEntry>::const_iterator;
  iterator begin() { return entries_.begin(); }
  const_iterator begin() const { return entries_.cbegin(); }
  iterator end() { return entries_.end(); }
  const_iterator end() const { return entries_.cend(); }

  /**
   * @brief Add an data entry / field / column to the logger.
   *
   * @tparam Args
   * @param col Data column. Specified the column in which the data should be
   * printed. If col >= 0, it is the 0-based column index. If col < 0, it counts
   * backwards from the end, with col = -1 adding it as the last column.
   * @param args Arguments to be passed to the LogEntry constructor.
   */
  template <class... Args>
  LogEntry& AddEntry(const int& col, Args... args);

  /**
   * @brief Set the verbosity level of the logger.
   * 
   * @param level 
   */
  void SetLevel(const LogLevel level) { cur_level_ = level; }

  /**
   * @brief Disable all output.
   * 
   */
  void Disable() { SetLevel(LogLevel::kSilent); }

  /**
   * @brief Set the frequency at which the header is printed.
   * 
   * If the frequency is set to 5, the header will be printed every 5 iterations.
   * 
   * @param freq 
   */
  void SetFrequency(const int freq) {
    ALTRO_ASSERT(freq >= 0, "Header print frequency must be positive.");
    frequency_ = freq;
  }

  /**
   * @brief Log the data for a given field.
   *
   * It is the user's responsibility to ensure the provided data is consistent
   * with the format spec for the given field.
   * 
   * Will not log the data if the entry isn't active at the current verbosity level.
   *
   * @tparam T data type of the given data.
   * @param title Title of the data column to add the data to. It must be an existing data column
   * (but can be inactive).
   * @param value The value to be logged, formatted, and later printed out.
   */
  template <class T>
  void Log(const std::string& title, T value);

  /**
   * @brief Print the header
   * 
   * Prints the titles of all of the active entries, followed by a horizontal rule 
   * and a line break.
   * Will not print anything if the current verbosity level is 0.
   */
  void PrintHeader();

  /**
   * @brief Print a data row
   * 
   * Prints the data (including conditional formatting) for all active entries.
   * Will not print anything if the current verbosity level is 0.
   */
  void PrintData();

  /**
   * @brief Automatically prints the header at a specified frequency
   * 
   */
  void Print();

  /**
   * @brief Clear all of the data entries in the table.
   * 
   */
  void Clear();

  /**
   * @brief Set the color of the header and it's horizontal rule
   * 
   * @param color One of the colors provided by the fmt library 
   * (e.g. fmt::color::green, fmt::color::yellow, fmt::color::red, fmt::color::white, etc.)
   */
  void SetHeaderColor(const fmt::color color) { header_color_ = color; }

 private:
  static constexpr int kDefaultFrequency = 10;

  LogLevel cur_level_ = LogLevel::kSilent;   // Current verbosity level
  int frequency_ = kDefaultFrequency;        // frequency of the header print
  int count_ = 0;                            // number of prints since header
  std::unordered_map<std::string, LogEntry> entries_;
  std::vector<const std::string*> order_;
  fmt::color header_color_ = fmt::color::white;
};

template <class... Args>
LogEntry& SolverLogger::AddEntry(const int& col, Args... args) {
  ALTRO_ASSERT(
      col <= static_cast<int>(entries_.size()),
      fmt::format("Column ({}) must be less than or equal to the current number of entries ({}).",
                  col, NumEntries()));
  ALTRO_ASSERT(
      col >= -static_cast<int>(entries_.size()) - 1,
      fmt::format(
          "Column ({}) must be greater or equal to than the negative new number of entries ({}).",
          col, -NumEntries() - 1));

  // Create a LogEntry and forward it to the map after extracting the title
  LogEntry entry(std::forward<Args>(args)...);
  const std::string title = entry.GetTitle();
  auto insert = entries_.emplace(std::make_pair(title, std::move(entry)));

  // Get a pointer to the title string used in the map
  // insert.first is an iterator over key-value pairs for the map entries_
  const std::string* title_ptr = &(insert.first->first);

  // Specify the output order
  auto it = order_.begin();
  if (col < 0) {
    // Count from the back if the input is negative
    it = order_.end() + col + 1;
  } else {
    it += col;
  }
  order_.insert(it, title_ptr);

  return insert.first->second;
}

template <class T>
void SolverLogger::Log(const std::string& title, T value) {
  // Short-circuit to skip the hash lookup if logging is disabled
  if (cur_level_ > LogLevel::kSilent && entries_[title].IsActive(cur_level_)) {
    entries_[title].Log(value);
  }
}


}  // namespace altro