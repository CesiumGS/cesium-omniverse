#pragma once

// Provides a portable way of ignoring unused parameters. Unused parameters should be optimized away by the compiler.
//   If and when we can move to C++17, this can be replaced with [[maybe_unused]]
#define UNUSED(expr) do { (void)(expr); } while(0);
