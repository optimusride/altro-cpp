// Copyright [2021] Optimus Ride Inc.

#include <fmt/format.h>
#include <fmt/color.h>
#include <gtest/gtest.h>

#include "altro/common/solver_options.hpp"

namespace altro {

TEST(SolverLoggerTest, MWE) {
  // Create logger and add entries
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
  logger.Clear();
}

TEST(SolverLoggerTest, Example) {
  SolverLogger logger;
  // Add an integer entry with a column witdh of 6
  logger.AddEntry(0, "iters", "{:>4}", LogEntry::kInt).SetWidth(6).SetLevel(LogLevel::kOuterDebug);

  // Add a float entry width default width of 10 and level kOuter.
  logger.AddEntry(1, "cost", "{:.4g}").SetLevel(LogLevel::kOuter);

  // Add another float entry with exponential format. Place it in the penultimate column.
  // Adds a lower bound to print green if the value is below 1e-4.
  // Uses default level of kInner
  logger.AddEntry(-2, "viol", "{:>.3e}").SetLowerBound(1e-4, fmt::color::green);

  // Add some data
  // NOTE: Must set the level before logging the data, otherwise it will be discarded
  logger.SetLevel(LogLevel::kOuter);
  logger.Log("iters", 1);  // this data is discarded
  logger.Log("cost", 100.0);

  // Print twice (header prints the first time)
  // Only prints "cost" column
  logger.Print();
  logger.Print();

  // Print with new level
  logger.SetLevel(LogLevel::kOuterDebug);
  logger.PrintHeader();    // Force the header to print
  logger.Print();          // prints a blank for the iter column
  logger.Log("iters", 1);  // Add entry for "iters" column
  logger.Print();          // now prints a value 

  // Clear all the values
  logger.Clear();

  // Set to new level
  logger.SetLevel(LogLevel::kInner);
  logger.SetHeaderColor(fmt::color::cyan);  // Change header color
  logger.Print(); // prints a new header and a blank line after clearing
  logger.Log("iter", 2);
  logger.Log("viol", 1e-3);
  logger.Log("cost", 10.0);
  logger.Print();
  logger.Log("viol", 1e-5);
  logger.Print();  // Prints green since it's below the threshold
}

}  // namespace altro