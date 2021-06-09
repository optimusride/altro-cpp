#include "utils/assert.hpp"

namespace altro {
void AssertMsg(bool expr, const char* msg, const char* expr_str, 
               int line, const char* file)
{
  if (!expr) {
    std::cerr << "Assert failed:\t" << msg << "\n"
              << "    Evaluated:\t" << "'" << expr_str << "'"
              << " in " << file << "::" << line << "" << std::endl;
    abort();
  }
}
}  // namespace altro