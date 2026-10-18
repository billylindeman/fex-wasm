// SPDX-License-Identifier: MIT
#include <FEXCore/Utils/CompilerDefs.h>

namespace FEXCore::Assert {
// This function can not be inlined
#if defined(__wasm__)
// WASI: no naked functions or native trap instructions; __builtin_trap maps to
// the wasm `unreachable` opcode.
[[noreturn]]
__attribute__((noinline)) void ForcedAssert() {
  __builtin_trap();
}
#else
[[noreturn]]
__attribute__((noinline, naked)) void ForcedAssert() {
#ifdef ARCHITECTURE_x86_64
  asm volatile("ud2");
#else
  asm volatile("hlt #1");
#endif
}
#endif
} // namespace FEXCore::Assert
