// SPDX-License-Identifier: MIT
#include <FEXCore/Utils/LongJump.h>
#include <FEXCore/Utils/LogManager.h>

#include <cstdint>
#include <cstring>

namespace FEXCore::UncheckedLongJump {
#if defined(__wasm__)
// WASM: host-side unwind/longjmp is redesigned with the wasm platform shim.
uint64_t SetJump(JumpBuf&) { return 0; }
void LongJump(const JumpBuf&, uint64_t) {}
#else
#if defined(ARCHITECTURE_arm64)
#ifdef __arm64ec__
#pragma clang diagnostic ignored "-Winline-asm"
#endif
[[nodiscard]]
FEX_DEFAULT_VISIBILITY FEX_NAKED uint64_t SetJump(JumpBuf& Buffer) {
  __asm volatile(R"(
      // x0 contains the jumpbuffer
      stp x19, x20, [x0, #( 0 * 8)];
      stp x21, x22, [x0, #( 2 * 8)];
      stp x23, x24, [x0, #( 4 * 8)];
      stp x25, x26, [x0, #( 6 * 8)];
      stp x27, x28, [x0, #( 8 * 8)];
      stp x29, x30, [x0, #(10 * 8)];

      // FPRs
      stp d8,   d9, [x0, #(12 * 8)];
      stp d10, d11, [x0, #(14 * 8)];
      stp d12, d13, [x0, #(16 * 8)];
      stp d14, d15, [x0, #(18 * 8)];

      // Move SP in to a temporary to store.
      mov x1, sp;
      str x1,  [x0, #(20 * 8)];

      // Return zero to signify this is the SetJump.
      mov x0, #0;
      ret;
    )" ::
                   : "memory");
}

[[noreturn]]
FEX_DEFAULT_VISIBILITY FEX_NAKED void LongJump(const JumpBuf& Buffer, uint64_t Value) {
  __asm volatile(R"(
      // x0 contains the jumpbuffer
      ldp x19, x20, [x0, #( 0 * 8)];
      ldp x21, x22, [x0, #( 2 * 8)];
      ldp x23, x24, [x0, #( 4 * 8)];
      ldp x25, x26, [x0, #( 6 * 8)];
      ldp x27, x28, [x0, #( 8 * 8)];
      ldp x29, x30, [x0, #(10 * 8)];

      // FPRs
      ldp d8,   d9, [x0, #(12 * 8)];
      ldp d10, d11, [x0, #(14 * 8)];
      ldp d12, d13, [x0, #(16 * 8)];
      ldp d14, d15, [x0, #(18 * 8)];

      // Load SP in to temporary then move
      ldr x0,  [x0, #(20 * 8)];
      mov sp, x0;

      // Move value in to result register
      mov x0, x1;
      ret;
    )" ::
                   : "memory");
}

FEX_DEFAULT_VISIBILITY void ManuallyLoadJumpBuf(const JumpBuf& Buffer, uint64_t Value, uint64_t* GPRs, __uint128_t* FPRs, uint64_t* PC) {
  // First 12 values are registers [x19,x30].
  memcpy(&GPRs[19], &Buffer.Registers[0], sizeof(uint64_t) * 12);

  // Next 8 values are [D8,D15]
  // Retain upper 64-bits of the register, only modifying lower 64-bits.
  for (size_t i = 0; i < 8; ++i) {
    memcpy(&FPRs[8 + i], &Buffer.Registers[12 + i], sizeof(uint64_t));
  }

  // Last value is stack pointer
  memcpy(&GPRs[31], &Buffer.Registers[20], sizeof(uint64_t));

  // Load the expected value in to X0
  GPRs[0] = Value;

  // Load the PC with the current LR.
  *PC = GPRs[30];
}

#endif
#endif

} // namespace FEXCore::UncheckedLongJump
