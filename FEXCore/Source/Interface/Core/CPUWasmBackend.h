// SPDX-License-Identifier: MIT
/*
$info$
category: backend ~ IR to host code generation
tags: backend|wasm
$end_info$
*/

#pragma once

// The WASM CPU backend: lowers FEXCore IR to a WebAssembly module, one guest
// block per module (docs/spikes/p1-d2-wasm-backend.md). This is the naive
// no-optimizer backend from doc 02 §3: every IR op maps 1:1 onto wasm
// instructions, all SSA values live in wasm locals, and the guest CPU context
// (FEXCore::Core::CpuStateFrame) lives in the imported linear memory at a
// fixed base, accessed through absolute i32 load/store offsets (doc 01 §5
// zero-translation).
//
// What it must never do: run IR optimizer passes (the IR arrives exactly as
// the frontend produced it), allocate guest memory, or touch the Arm64 JIT.
// It compiles for any target — only the Core.cpp wiring is __wasm__-guarded —
// which lets the native debug tool emit and inspect the same modules.

#include "Interface/Core/CPUBackend.h"

#include <FEXCore/fextl/vector.h>

#include <cstdint>

namespace FEXCore::Core {
struct InternalThreadState;
} // namespace FEXCore::Core

namespace FEXCore::Context {
class ContextImpl;
}

namespace FEXCore::CPU {

// Fixed base offset of the guest CpuStateFrame inside the imported linear
// memory, used when the harness doesn't override it (native debug tool).
// The wasm harness mallocs the context and passes its address via
// SetContextBase before CompileCode, so emitter and runtime cannot drift.
constexpr uint64_t WasmContextBase = 0x01800000;

// Guest VA of the code buffer the native debug tool lifts from (it mmaps a
// page there because the decoder reads guest memory through raw pointers).
constexpr uint64_t WasmGuestCodeBase = 0x01000000;

// A host function injected into generated modules through the extra-imports
// hook (D2 handoff item 3, doc 04 §2 S2 shape): all parameters are i32 (guest
// pointers/lengths are i32 on the 32-bit guest), result i32. In the D3a thunk
// ABI v0 the dispatcher invokes the host export itself (doc 02 §7 "exit to
// host"); the injected import proves the injection path end-to-end — the
// module refuses to instantiate if the host does not supply it.
//
// P1.10 widening (docs/spikes/p1-emitter-widening.md): the subset now covers
// the control-flow/flag path (CondJump/Jump over multi-block IRLists,
// cmp/test flag materialization, adc/sbb, setcc via NZCVSelect) — still
// failing loudly on anything outside it.
struct WasmHostImport {
  const char* Module;
  const char* Name;
  uint32_t ParamCount;
};

class WasmBackend final : public CPUBackend {
public:
  explicit WasmBackend(Context::ContextImpl* ctx, Core::InternalThreadState* Thread);

  // Base offset of the guest CpuStateFrame inside the imported linear memory.
  // The harness sets this before CompileCode so the generated module's context
  // accesses are baked in at the right offset. Defaults to WasmContextBase for
  // the native debug tool; the D3 pipeline will pass it per-block explicitly.
  void SetContextBase(uint64_t Base) { ContextBase = Base; }

  // Reserved guest-address window the dispatcher recognizes at block lookup
  // (doc 04 §3.2). Blocks whose exit RIP lands inside it get the registered
  // host imports injected into their generated module (extra-imports hook).
  void SetThunkWindow(uint64_t Base, uint64_t Size) {
    ThunkWindowBase = Base;
    ThunkWindowSize = Size;
  }

  // Registers a host import for the extra-imports hook. Call before
  // CompileCode; the names must outlive the emitted module bytes.
  void AddHostImport(WasmHostImport Import) { HostImports.push_back(Import); }

  [[nodiscard]]
  CPUBackend::CompiledCode CompileCode(uint64_t Entry, uint64_t Size, bool SingleInst, const FEXCore::IR::IRListView* IR,
                                       FEXCore::Core::DebugData* DebugData, bool CheckTF) override;

  [[nodiscard]]
  fextl::vector<FEXCore::CPU::Relocation> TakeRelocations(uint64_t GuestBaseAddress) override {
    return {};
  }

  // The module emitted by the most recent CompileCode. The D2 harness reads
  // these bytes out of the linear memory and hands them to the host runtime
  // (Node for D2; the browser JS bridge later).
  struct GeneratedModule {
    const uint8_t* Data;
    size_t Size;
    uint64_t EntryRIP;
  };
  [[nodiscard]]
  GeneratedModule GetLastModule() const {
    return {ModuleBytes.data(), ModuleBytes.size(), LastEntryRIP};
  }

private:
  fextl::vector<uint8_t> ModuleBytes;
  uint64_t LastEntryRIP {};
  uint64_t ContextBase = WasmContextBase;
  uint64_t ThunkWindowBase = 0;
  uint64_t ThunkWindowSize = 0;
  fextl::vector<WasmHostImport> HostImports;
};

[[nodiscard]]
fextl::unique_ptr<CPUBackend> CreateWasmBackend(Context::ContextImpl* ctx, Core::InternalThreadState* Thread);

} // namespace FEXCore::CPU
