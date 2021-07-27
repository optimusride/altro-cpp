#pragma once

#include <fmt/color.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

namespace altro {

/**
 * @brief Verbose output level
 * 
 * Higher levels include all the input from lower levels.
 * 
 * Both "outer" levels print information about the augmented Lagrangian iterations,
 * while the "inner" levels print information about the iLQR iterations. 
 * 
 * When printing "inner" iterations the header is printed at every outer AL iteration.
 * 
 * The secondary "debug" levels provide a few extra fields, while the lowest-level "kDebug"
 * prints everything recorded by the logger.
 * 
 * The specific values included in each level are subject to change in future revisions.
 * 
 */
enum class LogLevel {
  kSilent = 0,
  kOuter = 1,
  kOuterDebug = 2,
  kInner = 3,
  kInnerDebug = 4,
  kDebug = 5,
};

/**
 * @brief Represents an entry in the solver logger
 *
 * This class controls the printing of data or info during a solve, to be
 * included in a tabular-like output. It specifies the field width, formatting,
 * verbosity level, and simple bounds-based conditional formatting for numeric
 * data.
 *
 */
class LogEntry {
 public:
  enum EntryType { kInt, kFloat, kString };

  LogEntry() = default;
  /**
   * @brief Construct a new Log Entry object
   *
   * @param title Title of the entry. This is displayed in the header.
   * @param fmt The output format, specified as a Python-style format string, e.g. "{:>4.2f}".
   * @param level (default = 1) Verbosity level. Lower verbosity levels are printed more often.
   * @param width Field width. This is the width of data column when printed.
   *
   * @pre The width specified by @param width should be greater than the width specified by
   *      @param fmt. This is not checked and is the responsibility of the user to verify.
   */
  explicit LogEntry(std::string title, std::string fmt, const EntryType& type = kFloat)
      : title_(title), name_(std::move(title)), format_(std::move(fmt)), type_(type) {}

  const std::string& GetFormat() const { return format_; }
  const std::string& GetTitle() const { return title_; }
  int GetWidth() const { return width_; }
  LogLevel GetLevel() const { return level_; }
  bool IsActive(const LogLevel level) { return level >= level_; }
  EntryType GetType() const { return type_; }

  /**
   * @brief Set the Lower Bound for numeric data, along with the color if the
   * printed value is lower than this bound
   *
   * @param lb Lower bound
   * @param color Color of data to be printed if the data is lower than lb.
   */
  LogEntry& SetLowerBound(const double& lb, fmt::color color = fmt::color::green);

  /**
   * @brief Set the Upper Bound for numeric data, along with the color if the
   * printed value is greater than this bound
   *
   * @param ub Lower bound
   * @param color Color of data to be printed if the data is greater than ub.
   */
  LogEntry& SetUpperBound(const double& ub, fmt::color color = fmt::color::red);

  /**
   * @brief Set the width of the data column
   *
   * This should be greater than or equal to any width specified by the format
   * string.
   *
   * @param width Width of the data field, in characters.
   */
  LogEntry& SetWidth(const int& width);

  /**
   * @brief Set the verbosity level
   *
   * The entry (and header) will only be printed if the current verbosity level
   * is greater than or equal this value.
   *
   * @param level Verbosity level. level > 0
   */
  LogEntry& SetLevel(const LogLevel& level);

  /**
   * @brief Set the type of the entry.
   * 
   * The type is any of kInt, kDouble, or kString
   * 
   * @param type entry type
   * @return Reference to the log entry.
   */
  LogEntry& SetType(const EntryType& type);

  /**
   * @brief Set the name of the entry
   * 
   * The name should be a short but descriptive string, about what you would 
   * use for a variable name.
   * 
   * For example, if we're logging a penalty parameter, the title could be something
   * short like "pen" or "rho", and the name would be "penalty" or "penalty parameter".
   * 
   * @param name 
   * @return LogEntry& 
   */
  LogEntry& SetName(const std::string& name);

  /**
   * @brief Log a value to be printed later
   *
   * Calls fmt::format to convert the data to a string.
   *
   * @tparam T data type of the provided value. Should be consistent with whatever
   * format is specified.
   * @param value The data value to be formatted and converted to a string for printing.
   */
  template <class T>
  void Log(T value) {
    color_ = GetColor(value);
    data_ = fmt::format(format_, value);
  }

  /**
   * @brief Print the data, only if the specified level is greater than the
   * level of the data entry.
   *
   * @param level Current verbosity level.
   */
  void Print(const LogLevel level = LogLevel::kSilent) {
    if (IsActive(level)) {
      fmt::print(fg(color_), "{:>{}}", data_, width_);
    }
  }

  /**
   * @brief Print the header entry, only if the specified level is greater than
   * the level of the data entry.
   *
   * @param level Current verbosity level.
   * @param color Color of the header (default is white).
   */
  void PrintHeader(const LogLevel level = LogLevel::kSilent, const fmt::color color = fmt::color::white) {
    if (IsActive(level)) {
      fmt::print(fg(color), "{:>{}}", title_, width_);
    }
  }

  /**
   * @brief Shortcut to log and print a value in one command
   *
   * @tparam T Data type of the data to be printed
   * @param value Value to be logged and printed out.
   */
  template <class T>
  void Print(T value) {
    Log(value);
    Print();
  }

  /**
   * @brief Clear the data currently stored in the log. Will print an empty
   * string after calling this function.
   *
   */
  void Clear() { data_.clear(); }

 private:
  /**
   * @brief Get the Color of the data field if it is bounded.
   *
   * @tparam T
   * @param value
   * @return fmt::color
   */
  template <class T>
  fmt::color GetColor(T value) {
    fmt::color color = color_default_;
    if (bounded_) {
      if (value < lower_) {
        color = color_lower_;
      } else if (value > upper_) {
        color = color_upper_;
      }
    }
    return color;
  }

  static constexpr int kDefaultWidth = 10;

  std::string title_;     // title of the entry to appear in the header
  std::string name_;      // descriptive name (e.g. "penalty parameter" vs. "pen")
  std::string format_;    // Python-style format string
  std::string data_;      // storage for the logged data
  EntryType type_;
  LogLevel level_ = LogLevel::kInner;         // verbosity level
  int width_ = kDefaultWidth;        // column width
  bool bounded_ = false;  // does the field have conditional formatting
  double lower_ = -std::numeric_limits<double>::infinity();  // lower bound
  double upper_ = +std::numeric_limits<double>::infinity();  // upper bound
  fmt::color color_ = fmt::color::white;                            // current color to be printed
  fmt::color color_default_ = fmt::color::white;                     // default color (bounds are satisfied)
  fmt::color color_lower_ = fmt::color::green;                      // color if below lower bound
  fmt::color color_upper_ = fmt::color::red;                        // color if above upper bound
};

}  // namespace altro