// Copyright [2021] Optimus Ride Inc.

#include <iostream>

#include "altro/utils/assert.hpp"

namespace altro {
namespace utils {
  
/**
 * @brief Informative assertion that allows the developer to include a message
 *
 * Usually invoked via the ALTRO_ASSERT(expr, msg) macro.
 *
 * Inspired by this StackOverflow post:
 * https://stackoverflow.com/questions/3692954/add-custom-messages-in-assert/3692961
 */
void AssertMsg(bool expr, const std::string& msg, const char *expr_str, int line,
               const char *file) {
  if (!expr) {
    std::cerr << "Assert failed:\t" << msg << "\n"
              << "    Evaluated:\t"
              << "'" << expr_str << "'"
              << " in " << file << "::" << line << "" << std::endl;
    abort();
  }
}

} // namespace utils
} // namespace altro