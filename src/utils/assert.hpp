#pragma once

#include <iostream>

#ifndef NDEBUG
#define ALTRO_ASSERT(Expr, Msg)                                                \
  altro::AssertMsg((Expr), Msg, #Expr, __LINE__, __FILE__)
#else
#define ALTRO_ASSERT(Expr, Msg) ;
#endif

namespace altro {

void AssertMsg(bool expr, const char *msg, const char *expr_str, int line,
               const char *file);

} // namespace altro