// SPDX-License-Identifier: MIT
/*
$info$
category: backend ~ IR to host code generation
tags: backend|wasm
$end_info$
*/

// The WASM CPU backend (doc 02 §3, docs/spikes/p1-d2-wasm-backend.md).
//
// D2 scope: a naive, no-optimizer IR→wasm lowering for 32-bit-guest ALU
// blocks. One guest block → one wasm module that imports the harness's
// linear memory (the guest memory, doc 01 §5) and exports one zero-parameter
// function that executes the block against a CpuStateFrame living at a fixed
// base offset in that memory.
//
// Emitter model (mirrors the Arm64 JIT's register contract, Arm64JITCore
// GetReg()):
// - Every SSA node gets a wasm i32 local, indexed by node ID.
// - The 18 "static" locations (16 guest GPRs + PF/AF raw pseudo-regs) get
//   dedicated locals, loaded from the context in the prologue and stored back
//   in the epilogue. A node whose RA annotation is GPRFixed writes its static
//   location (that annotation is what `mov` into the PF/AF pseudo-regs means),
//   as do the StoreRegister/StorePF/StoreAF ops.
// - GPR-class (dynamic) RA annotations are remembered as a last-def map so
//   post-RA arguments that reference a dynamic register resolve to the
//   defining node's local.
// - The compacted NZCV value is a dedicated local, synced to
//   CPUState.flags[RFLAG_NZCV_LOC] — the canonical context slot (see
//   ReconstructCompactedEFLAGS in Core.cpp).
// - 32-bit guest only: GPR values are i32; gregs' upper halves stay zero
//   (32-bit ops zero-extend by construction, and nothing writes them).

#include "Interface/Core/CPUWasmBackend.h"

#include "Interface/Context/Context.h"
#include "Interface/IR/IR.h"
#include "Interface/IR/IntrusiveIRList.h"
#include "Interface/IR/Passes/RegisterAllocationPass.h"
#include "Interface/IR/RegisterAllocationData.h"

#include <FEXCore/Core/CoreState.h>
#include <FEXCore/Core/X86Enums.h>
#include <FEXCore/Utils/LogManager.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace FEXCore::CPU {
namespace {


constexpr uint32_t NUM_STATIC_SLOTS = 18; // 16 guest GPRs + PF + AF
constexpr uint32_t NUM_DYNAMIC_REGS = 32; // RA dynamic GPR/FPR space we track

void PushULEB(fextl::vector<uint8_t>& Out, uint32_t V) {
  do {
    uint8_t Byte = V & 0x7F;
    V >>= 7;
    if (V != 0) {
      Byte |= 0x80;
    }
    Out.push_back(Byte);
  } while (V != 0);
}

void PushSLEB(fextl::vector<uint8_t>& Out, int64_t V) {
  bool More;
  do {
    uint8_t Byte = V & 0x7F;
    V >>= 7;
    More = !((V == 0 && (Byte & 0x40) == 0) || (V == -1 && (Byte & 0x40) != 0));
    if (More) {
      Byte |= 0x80;
    }
    Out.push_back(Byte);
  } while (More);
}

void PushVec(fextl::vector<uint8_t>& Out, std::initializer_list<uint8_t> Bytes) {
  Out.insert(Out.end(), Bytes.begin(), Bytes.end());
}

void PushU32Const(fextl::vector<uint8_t>& Out, uint32_t V) {
  Out.push_back(0x41); // i32.const
  PushSLEB(Out, static_cast<int32_t>(V));
}

void PushU64Const(fextl::vector<uint8_t>& Out, uint64_t V) {
  Out.push_back(0x42); // i64.const
  PushSLEB(Out, static_cast<int64_t>(V));
}

void PushLocalGet(fextl::vector<uint8_t>& Out, uint32_t Local) {
  Out.push_back(0x20); // local.get
  PushULEB(Out, Local);
}

void PushLocalSet(fextl::vector<uint8_t>& Out, uint32_t Local) {
  Out.push_back(0x21); // local.set
  PushULEB(Out, Local);
}

// mem-arg load/store; the address operand is the constant 0, so the absolute
// linear-memory offset rides in the mem-arg offset immediate.
void PushI32Load(fextl::vector<uint8_t>& Out, uint32_t Offset) {
  PushVec(Out, {0x28, 2}); // i32.load, align 4
  PushULEB(Out, Offset);
}

void PushI64Load(fextl::vector<uint8_t>& Out, uint32_t Offset) {
  PushVec(Out, {0x29, 3}); // i64.load, align 8
  PushULEB(Out, Offset);
}

void PushI32Load8U(fextl::vector<uint8_t>& Out, uint32_t Offset) {
  PushVec(Out, {0x2D, 0}); // i32.load8_u, align 1
  PushULEB(Out, Offset);
}

void PushI32Load16U(fextl::vector<uint8_t>& Out, uint32_t Offset) {
  PushVec(Out, {0x2F, 1}); // i32.load16_u, align 2
  PushULEB(Out, Offset);
}

void PushI32Store(fextl::vector<uint8_t>& Out, uint32_t Offset) {
  PushVec(Out, {0x36, 2}); // i32.store, align 4
  PushULEB(Out, Offset);
}

void PushI32Store8(fextl::vector<uint8_t>& Out, uint32_t Offset) {
  PushVec(Out, {0x3A, 0}); // i32.store8, align 1
  PushULEB(Out, Offset);
}

void PushI32Store16(fextl::vector<uint8_t>& Out, uint32_t Offset) {
  PushVec(Out, {0x3B, 1}); // i32.store16, align 2
  PushULEB(Out, Offset);
}

void PushI64Store(fextl::vector<uint8_t>& Out, uint32_t Offset) {
  PushVec(Out, {0x37, 3}); // i64.store, align 8
  PushULEB(Out, Offset);
}

// --- P3.1: wasm atomic RMW (the LOCK-prefixed guest class) -------------------
// The wasm threads proposal allows atomic RMW on non-shared memories (V8
// executes them; only wait/notify require shared). Every emitted op is
// sequentially consistent, which matches the LOCK prefix's x86 contract.

enum class AtomicRmw { Add, Sub, And, Or, Xor, Xchg, Cmpxchg };

// 0xFE-prefixed opcode byte per (family, size): the i32 form's index +2/+3
// for the 8/16 zero-extending forms (binary spec, Atomic Memory Instructions).
void PushAtomicRmw(fextl::vector<uint8_t>& Out, AtomicRmw Op, uint32_t Size, uint32_t Offset) {
  static constexpr uint8_t I32Form[] = {0x1E, 0x25, 0x2C, 0x33, 0x3A, 0x41, 0x48};
  static constexpr uint8_t Rmw8Form[] = {0x20, 0x27, 0x2E, 0x35, 0x3C, 0x43, 0x4A};
  static constexpr uint8_t Rmw16Form[] = {0x21, 0x28, 0x2F, 0x36, 0x3D, 0x44, 0x4B};
  uint8_t Opcode;
  switch (Size) {
  case 1: Opcode = Rmw8Form[static_cast<uint8_t>(Op)]; break;
  case 2: Opcode = Rmw16Form[static_cast<uint8_t>(Op)]; break;
  case 4: Opcode = I32Form[static_cast<uint8_t>(Op)]; break;
  default:
    // Callers gate sizes; a 64-bit guest RMW (cmpxchg64 et al) is not an
    // x86-32 shape. Loud-fail here too so the contract holds everywhere.
    LogMan::Msg::EFmt("WasmBackend: atomic RMW with unsupported size {}", Size);
    Opcode = 0;
  }
  // Atomic accesses require the alignment immediate to be the natural
  // alignment exactly: log2(Size). An unaligned guest effective address
  // traps at runtime — the documented divergence from x86 (see the spike).
  PushVec(Out, {0xFE, Opcode, static_cast<uint8_t>(Size == 1 ? 0 : Size == 2 ? 1 : 2)});
  PushULEB(Out, Offset);
}

// --- wasm SIMD (v128) helpers ------------------------------------------------
// All vector memory accesses use align=1: guest x86 permits unaligned
// scalar/vector accesses, and a wasm alignment hint above the natural size of
// the access would trap on them (fstp qword [esp+4] is a routine shape).

void PushV128Load(fextl::vector<uint8_t>& Out, uint32_t Offset) {
  PushVec(Out, {0xFD, 0x00}); // v128.load
  PushULEB(Out, 0);           // align 1
  PushULEB(Out, Offset);
}

void PushV128Load32Zero(fextl::vector<uint8_t>& Out, uint32_t Offset) {
  PushVec(Out, {0xFD, 0x5C}); // v128.load32_zero (lane 0 = mem32, rest 0)
  PushULEB(Out, 0);
  PushULEB(Out, Offset);
}

void PushV128Load64Zero(fextl::vector<uint8_t>& Out, uint32_t Offset) {
  PushVec(Out, {0xFD, 0x5D}); // v128.load64_zero (lane 0 = mem64, rest 0)
  PushULEB(Out, 0);
  PushULEB(Out, Offset);
}

void PushV128Store(fextl::vector<uint8_t>& Out, uint32_t Offset) {
  PushVec(Out, {0xFD, 0x0B}); // v128.store
  PushULEB(Out, 0);
  PushULEB(Out, Offset);
}

void PushF64Load(fextl::vector<uint8_t>& Out, uint32_t Offset) {
  PushVec(Out, {0x2B, 0}); // f64.load, align 1 (x86 allows unaligned)
  PushULEB(Out, Offset);
}

void PushF64Store(fextl::vector<uint8_t>& Out, uint32_t Offset) {
  PushVec(Out, {0x39, 0}); // f64.store, align 1
  PushULEB(Out, Offset);
}

void PushF32Store(fextl::vector<uint8_t>& Out, uint32_t Offset) {
  PushVec(Out, {0x38, 0}); // f32.store, align 1
  PushULEB(Out, Offset);
}

void PushF64Const(fextl::vector<uint8_t>& Out, uint64_t Bits) {
  Out.push_back(0x44); // f64.const (raw little-endian IEEE bit pattern)
  for (uint32_t i = 0; i < 8; ++i) {
    Out.push_back(static_cast<uint8_t>(Bits >> (8 * i)));
  }
}

void PushF32Const(fextl::vector<uint8_t>& Out, uint32_t Bits) {
  Out.push_back(0x43); // f32.const (raw little-endian IEEE bit pattern)
  for (uint32_t i = 0; i < 4; ++i) {
    Out.push_back(static_cast<uint8_t>(Bits >> (8 * i)));
  }
}

// mem-arg load/store size selector for 1/2/4-byte GPR accesses.
bool PushGPRLoad(fextl::vector<uint8_t>& Out, uint32_t AccessSize, uint32_t Offset) {
  switch (AccessSize) {
  case 1: PushI32Load8U(Out, Offset); return true;
  case 2: PushI32Load16U(Out, Offset); return true;
  case 4: PushI32Load(Out, Offset); return true;
  case 8: PushI64Load(Out, Offset); return true;
  default: return false;
  }
}

bool PushGPRStore(fextl::vector<uint8_t>& Out, uint32_t AccessSize, uint32_t Offset) {
  switch (AccessSize) {
  case 1: PushI32Store8(Out, Offset); return true;
  case 2: PushI32Store16(Out, Offset); return true;
  case 4: PushI32Store(Out, Offset); return true;
  default: return false;
  }
}

// --- condition evaluation ---------------------------------------------------
// Pushes `taken` (exactly 1 or 0) for an NZCV-sourced condition class onto the
// wasm stack. The compacted NZCV local has the Arm layout (N=31 Z=30 C=29
// V=28); the frontend rectifies the stored C to the invert state the op
// expects before emitting the condition, so the bit tests are direct:
// ULT/ULE/UGT/UGE read the stored C bit as-is (see DecodeNZCVCondition).
// SGT/SLE/SLT/ULE are emitted as the negation of their complement.
void EmitNZCVCondition(fextl::vector<uint8_t>& B, uint32_t NZCVLocal, uint32_t CondScratch, IR::CondClass Cond, bool* ok) {
  auto BitTest = [&](uint32_t Bit) { // 1 if bit set
    PushLocalGet(B, NZCVLocal);
    PushU32Const(B, Bit);
    B.push_back(0x71); // i32.and
    PushU32Const(B, Bit);
    B.push_back(0x46); // i32.eq
  };
  auto BitClear = [&](uint32_t Bit) { // 1 if bit clear
    PushLocalGet(B, NZCVLocal);
    PushU32Const(B, Bit);
    B.push_back(0x71); // i32.and
    B.push_back(0x45); // i32.eqz
  };
  auto EqOneOf = [&](uint32_t Mask, uint32_t A, uint32_t Bv) { // 1 if (nzcv & mask) == A or == B
    PushLocalGet(B, NZCVLocal);
    PushU32Const(B, Mask);
    B.push_back(0x71); // i32.and
    PushLocalSet(B, CondScratch); // tee via set; read back below
    PushLocalGet(B, CondScratch);
    PushU32Const(B, A);
    B.push_back(0x46); // i32.eq
    PushLocalGet(B, CondScratch);
    PushU32Const(B, Bv);
    B.push_back(0x46); // i32.eq
    B.push_back(0x72); // i32.or
  };
  switch (Cond) {
  case IR::CondClass::EQ: BitTest(1u << 30); break; // Z
  case IR::CondClass::NEQ: BitClear(1u << 30); break;
  case IR::CondClass::ULT: BitClear(1u << 29); break; // stored C clear
  case IR::CondClass::UGE: BitTest(1u << 29); break;
  case IR::CondClass::MI: BitTest(1u << 31); break; // N
  case IR::CondClass::PL: BitClear(1u << 31); break;
  case IR::CondClass::FU: BitTest(1u << 28); break; // V (float-unordered slot reused)
  case IR::CondClass::FNU: BitClear(1u << 28); break;
  case IR::CondClass::UGT: { // C set && Z clear
    PushLocalGet(B, NZCVLocal);
    PushU32Const(B, 0x60000000);
    B.push_back(0x71); // i32.and
    PushU32Const(B, 1u << 29);
    B.push_back(0x46); // i32.eq
    break;
  }
  case IR::CondClass::ULE: EqOneOf(0x60000000, 0, 0x60000000); break; // !UGT
  case IR::CondClass::SGE: EqOneOf(0x90000000, 0, 0x90000000); break; // N == V
  case IR::CondClass::SLT: { // N != V
    EqOneOf(0x90000000, 0, 0x90000000);
    B.push_back(0x45); // i32.eqz
    break;
  }
  case IR::CondClass::SGT: { // Z clear && N == V
    BitClear(1u << 30);
    EqOneOf(0x90000000, 0, 0x90000000);
    B.push_back(0x71); // i32.and
    break;
  }
  case IR::CondClass::SLE: { // Z set || N != V  ==  !SGT
    BitClear(1u << 30);
    EqOneOf(0x90000000, 0, 0x90000000);
    B.push_back(0x71); // i32.and
    B.push_back(0x45); // i32.eqz
    break;
  }
  default:
    LogMan::Msg::EFmt("WasmBackend: unsupported NZCV condition class {}", static_cast<uint32_t>(Cond));
    *ok = false;
  }
}

// Pushes `taken` (1/0) for a value comparison Cmp1 <Cond> Cmp2 against two
// resolved locals. Used by the non-NZCV CondJump path and by Select.
// 8/16-bit compares are done at 32-bit: both operands are zero-extended by
// construction (Bfe'd results / masked constants).
bool EmitValueCondition(fextl::vector<uint8_t>& B, IR::CondClass Cond, uint32_t A, uint32_t Bv) {
  auto Cmp = [&](uint8_t Op) {
    PushLocalGet(B, A);
    PushLocalGet(B, Bv);
    B.push_back(Op);
  };
  switch (Cond) {
  case IR::CondClass::EQ: Cmp(0x46); return true; // i32.eq
  case IR::CondClass::NEQ: Cmp(0x47); return true; // i32.ne
  case IR::CondClass::UGT: Cmp(0x4B); return true; // i32.gt_u
  case IR::CondClass::UGE: Cmp(0x4F); return true; // i32.ge_u
  case IR::CondClass::ULT: Cmp(0x49); return true; // i32.lt_u
  case IR::CondClass::ULE: Cmp(0x4D); return true; // i32.le_u
  case IR::CondClass::SGT: Cmp(0x4A); return true; // i32.gt_s
  case IR::CondClass::SGE: Cmp(0x4E); return true; // i32.ge_s
  case IR::CondClass::SLT: Cmp(0x48); return true; // i32.lt_s
  case IR::CondClass::SLE: Cmp(0x4C); return true; // i32.le_s
  case IR::CondClass::TSTZ: { // taken when bit Cmp2 of Cmp1 is clear
    PushLocalGet(B, A);
    PushLocalGet(B, Bv);
    B.push_back(0x76); // i32.shr_u
    PushU32Const(B, 1);
    B.push_back(0x71); // i32.and
    B.push_back(0x45); // i32.eqz
    return true;
  }
  case IR::CondClass::TSTNZ: {
    PushLocalGet(B, A);
    PushLocalGet(B, Bv);
    B.push_back(0x76); // i32.shr_u
    PushU32Const(B, 1);
    B.push_back(0x71); // i32.and
    return true;
  }
  default:
    LogMan::Msg::EFmt("WasmBackend: unsupported value condition class {}", static_cast<uint32_t>(Cond));
    return false;
  }
}

// The emitted block function plus its local-index bookkeeping.
//
// SSA node locals are typed (P2.12): GPR results are i32 (the 32-bit guest)
// unless the op produces a full 64-bit integer (the fild m64 load/extend
// chain), and FPR results are v128 — the xmm register file is 128-bit and
// scalar float lanes live in lane 0. Locals are declared in three groups
// (i32, i64, v128); NodeLocal maps a node ID to its index in the right group.
class FunctionEmitter {
public:
  enum class LType : uint8_t { I32, I64, V128 };

  static constexpr uint32_t NUM_FIXED_I32 = NUM_STATIC_SLOTS + 2; // statics + NZCV + cond scratch
  static constexpr uint32_t NUM_STATIC_FPR = 16;                 // xmm0-15 as v128 static slots

  FunctionEmitter(uint32_t SSACount, const fextl::vector<LType>& NodeTypes, uint32_t ScratchCount, uint32_t I64ScratchCount)
    : ScratchTotal(ScratchCount)
    , Types(&NodeTypes) {
    // Group sizes first (every group's base depends on the full count of the
    // groups before it), then assign node indices group by group.
    for (auto T : NodeTypes) {
      ++(T == LType::I64 ? NumI64Nodes : T == LType::V128 ? NumV128Nodes : NumI32Nodes);
    }
    const uint32_t I32Total = NumI32Nodes + NUM_FIXED_I32 + ScratchCount;
    I64Base = I32Total;
    const uint32_t I64Total = NumI64Nodes + I64ScratchCount;
    const uint32_t V128Base = I64Base + I64Total;

    NodeLocalIdx.assign(SSACount, 0);
    uint32_t I32Used = 0, I64Used = 0, V128Used = 0;
    for (uint32_t ID = 0; ID < SSACount; ++ID) {
      switch (ID < NodeTypes.size() ? NodeTypes[ID] : LType::I32) {
      case LType::I64: NodeLocalIdx[ID] = I64Base + I64Used++; break;
      case LType::V128: NodeLocalIdx[ID] = V128Base + V128Used++; break;
      default: NodeLocalIdx[ID] = I32Used++; break;
      }
    }

    StaticLocalBase = NumI32Nodes;
    NZCVLocal = StaticLocalBase + NUM_STATIC_SLOTS;
    CondScratchLocal = NZCVLocal + 1;
    ScratchLocalBase = CondScratchLocal + 1;
    I64ScratchBase = I64Base + NumI64Nodes;
    StaticFPRBase = V128Base + NumV128Nodes;
    NumLocals = StaticFPRBase + NUM_STATIC_FPR;
  }

  uint32_t NodeLocal(uint32_t NodeID) const { return NodeLocalIdx[NodeID]; }
  // Whether the node's local is in the i64 group (the 64-bit integer chain).
  bool NodeIsI64(uint32_t NodeID) const {
    return NodeID < Types->size() && (*Types)[NodeID] == LType::I64;
  }
  // Whether a local index lands in the i64 group (nodes + i64 scratch).
  bool LocalIsI64(uint32_t Local) const {
    return Local >= I64Base && Local < I64Base + NumI64Nodes + I64ScratchUsed;
  }
  uint32_t StaticLocal(uint32_t Slot) const { return StaticLocalBase + Slot; }
  uint32_t StaticFPRLocal(uint32_t Slot) const { return StaticFPRBase + Slot; }
  uint32_t NZCV() const { return NZCVLocal; }
  // Transient scratch for multi-part condition expressions. Safe to share:
  // a condition is computed atomically (local.tee then read) with no nesting.
  uint32_t CondScratch() const { return CondScratchLocal; }
  uint32_t AllocScratch() { return ScratchLocalBase + ScratchUsed++; }
  uint32_t AllocI64Scratch() { return I64ScratchBase + I64ScratchUsed++; }
  uint32_t Locals() const { return NumLocals; }
  uint32_t I32Locals() const { return NumI32Nodes + NUM_FIXED_I32 + ScratchTotal; }
  uint32_t I64Locals() const { return NumI64Nodes + I64ScratchUsed; }
  uint32_t V128Locals() const { return NumV128Nodes + NUM_STATIC_FPR; }

  fextl::vector<uint8_t> Body;

private:
  const fextl::vector<LType>* Types;
  fextl::vector<uint32_t> NodeLocalIdx;
  uint32_t I64Base = 0;
  uint32_t NumI32Nodes = 0;
  uint32_t NumI64Nodes = 0;
  uint32_t NumV128Nodes = 0;
  uint32_t ScratchTotal;
  uint32_t StaticLocalBase;
  uint32_t NZCVLocal;
  uint32_t CondScratchLocal;
  uint32_t ScratchLocalBase;
  uint32_t I64ScratchBase;
  uint32_t StaticFPRBase;
  uint32_t ScratchUsed = 0;
  uint32_t I64ScratchUsed = 0;
  uint32_t NumLocals;
};

class ModuleBuilder {
public:
  // Returns false (with a log) if the IR contains anything outside the D2/D3a
  // subset. EntryRIP is the guest RIP the block was lifted from; it resolves
  // InlineEntrypointOffset values. ThunkBase/ThunkSize is the reserved
  // guest-address window whose exits get HostImports injected (extra-imports
  // hook, doc 04 §3.2).
  bool Emit(const IR::IRListView* IR, uint64_t EntryRIP, uint64_t ContextBase, uint64_t ThunkBase, uint64_t ThunkSize,
            const fextl::vector<WasmHostImport>& HostImports);

  [[nodiscard]]
  const fextl::vector<uint8_t>& Bytes() const {
    return Module;
  }

private:
  // Resolve an SSA argument to the local holding its value. Handles both
  // node references and post-RA register references (the same encodings the
  // Arm64 JIT's GetReg() consumes). FPR references resolve to the xmm
  // register file: FPRFixed = the 16 static v128 slots, FPR = dynamic.
  bool ResolveArg(const IR::IRListView* IR, IR::OrderedNodeWrapper Arg, FunctionEmitter& FE, uint32_t* LocalOut) {
    if (Arg.IsImmediate()) {
      IR::PhysicalRegister PhyReg(Arg);
      switch (PhyReg.AsRegClass()) {
      case IR::RegClass::GPRFixed:
        if (PhyReg.Reg >= NUM_STATIC_SLOTS) {
          LogMan::Msg::EFmt("WasmBackend: static slot {} out of range", PhyReg.Reg);
          return false;
        }
        *LocalOut = FE.StaticLocal(PhyReg.Reg);
        return true;
      case IR::RegClass::GPR:
        if (PhyReg.Reg >= NUM_DYNAMIC_REGS || DynamicLastDef[PhyReg.Reg] == 0) {
          LogMan::Msg::EFmt("WasmBackend: dynamic register {} referenced before def", PhyReg.Reg);
          return false;
        }
        *LocalOut = DynamicLastDef[PhyReg.Reg] - 1; // stored as node ID + 1
        return true;
      case IR::RegClass::FPRFixed:
        if (PhyReg.Reg >= FunctionEmitter::NUM_STATIC_FPR) {
          LogMan::Msg::EFmt("WasmBackend: static FPR slot {} out of range", PhyReg.Reg);
          return false;
        }
        *LocalOut = FE.StaticFPRLocal(PhyReg.Reg);
        return true;
      case IR::RegClass::FPR:
        if (PhyReg.Reg >= NUM_DYNAMIC_REGS || FPRLastDef[PhyReg.Reg] == 0) {
          LogMan::Msg::EFmt("WasmBackend: dynamic FPR {} referenced before def", PhyReg.Reg);
          return false;
        }
        *LocalOut = FPRLastDef[PhyReg.Reg] - 1;
        return true;
      default:
        LogMan::Msg::EFmt("WasmBackend: unsupported register class {}", static_cast<uint32_t>(PhyReg.AsRegClass()));
        return false;
      }
    }
    *LocalOut = FE.NodeLocal(IR->GetID(IR->GetNode(Arg)).Value);
    return true;
  }

  // The Direction arg of MemSet/MemCpy may be a folded constant node — the
  // frontend's known-DF path lands as an OP_INLINECONSTANT node (the D4
  // str-rep-movsb-bwd finding: the STD-preloaded backward rep movs'
  // direction arrives as `InlineConstant 0xffffffffffffffff`). Such a node
  // has no dynamic-register def, so plain ResolveArg maps it to a
  // never-written local (reads 0 → the loop's direction check and the
  // result select both take the forward branch). Materialize the constant
  // into a scratch local instead; the caller's scratch budget covers it.
  // OP_CONSTANT is accepted alongside it for the same reason.
  bool ResolveDirection(const IR::IRListView* IR, IR::OrderedNodeWrapper Arg, FunctionEmitter& FE, uint32_t* LocalOut) {
    if (!Arg.IsImmediate()) {
      const auto OpHeader = IR->GetOp<IR::IROp_Header>(Arg);
      if (OpHeader->Op == IR::IROps::OP_CONSTANT || OpHeader->Op == IR::IROps::OP_INLINECONSTANT) {
        const auto Value = static_cast<int32_t>(OpHeader->Op == IR::IROps::OP_CONSTANT
            ? OpHeader->C<IR::IROp_Constant>()->Constant
            : OpHeader->C<IR::IROp_InlineConstant>()->Constant);
        PushU32Const(FE.Body, Value);
        const uint32_t Scratch = FE.AllocScratch();
        PushLocalSet(FE.Body, Scratch);
        *LocalOut = Scratch;
        return true;
      }
    }
    return ResolveArg(IR, Arg, FE, LocalOut);
  }

  void RecordDef(FunctionEmitter& FE, const IR::IRListView* IR, IR::OrderedNode* Node, uint32_t NodeID) {
    IR::PhysicalRegister PhyReg(Node);
    if (PhyReg.IsInvalid()) {
      return;
    }
    // The last-def maps hold *local index + 1* (0 means "no def"); with typed
    // node locals the local index is no longer the node ID.
    if (PhyReg.AsRegClass() == IR::RegClass::GPR && PhyReg.Reg < NUM_DYNAMIC_REGS) {
      DynamicLastDef[PhyReg.Reg] = FE.NodeLocal(NodeID) + 1;
    } else if (PhyReg.AsRegClass() == IR::RegClass::FPR && PhyReg.Reg < NUM_DYNAMIC_REGS) {
      FPRLastDef[PhyReg.Reg] = FE.NodeLocal(NodeID) + 1;
    }
    // GPRFixed annotations are applied by the explicit store ops below (the
    // node itself is the value that lands in the static location).
  }

  void NoteStaticDef(FunctionEmitter& FE, IR::PhysicalRegister PhyReg, uint32_t NodeLocal) {
    if (PhyReg.AsRegClass() == IR::RegClass::GPRFixed && PhyReg.Reg < NUM_STATIC_SLOTS) {
      PushLocalGet(FE.Body, NodeLocal);
      PushLocalSet(FE.Body, FE.StaticLocal(PhyReg.Reg));
    } else if (PhyReg.AsRegClass() == IR::RegClass::FPRFixed && PhyReg.Reg < FunctionEmitter::NUM_STATIC_FPR) {
      PushLocalGet(FE.Body, NodeLocal);
      PushLocalSet(FE.Body, FE.StaticFPRLocal(PhyReg.Reg));
    }
  }

  // Emits the effective address of a LoadMem/StoreMem, leaving it on the wasm
  // stack. Guest VA == linear-memory offset (doc 01 §5), so the address is
  // used directly; a constant displacement folds into the mem-arg offset.
  // Anything else fails loudly (the honest-failure contract).
  bool EmitEffectiveAddress(const IR::IRListView* IR, IR::OrderedNodeWrapper Addr, IR::OrderedNodeWrapper Offset,
                            IR::MemOffsetType OffsetType, uint8_t OffsetScale, FunctionEmitter& FE, uint32_t* MemArgOffset) {
    // The 32-bit guest model's address width: an I64-typed base (the 8-byte
    // GDT-entry read the segment chain feeds forward) wraps to i32 — the
    // guest's address space is the low 32 bits (guest VA == wasm offset,
    // doc 01 §5).
    auto PushAddrLocal = [&](uint32_t Local) {
      PushLocalGet(FE.Body, Local);
      if (FE.LocalIsI64(Local)) {
        FE.Body.push_back(0xA7); // i32.wrap_i64
      }
    };
    uint32_t Base;
    if (!ResolveArg(IR, Addr, FE, &Base)) {
      return false;
    }
    if (Offset.IsInvalid()) {
      *MemArgOffset = 0;
      PushAddrLocal(Base);
      return true;
    }
    if (!Offset.IsImmediate()) {
      auto* NH = IR->GetOp<IR::IROp_Header>(IR->GetNode(Offset));
      if (NH->Op == IR::IROps::OP_INLINECONSTANT) {
        const uint64_t Const = NH->C<IR::IROp_InlineConstant>()->Constant;
        if (OffsetType == IR::MemOffsetType::SXTX && OffsetScale == 1 && Const <= 0x7FFFFFFFull) {
          *MemArgOffset = static_cast<uint32_t>(Const);
          PushAddrLocal(Base);
          return true;
        }
        PushAddrLocal(Base);
        PushU32Const(FE.Body, static_cast<uint32_t>(Const));
        FE.Body.push_back(0x6A); // i32.add
        *MemArgOffset = 0;
        return true;
      }
    }
    if (OffsetType != IR::MemOffsetType::SXTX && OffsetType != IR::MemOffsetType::UXTW) {
      // UXTW is the segment chain's offset type (UpdatePrefixFromSegment's
      // selector & 0xfff8). In the 32-bit model a 32-bit offset node emits
      // identically under either extension (the wasm i32 add wraps), so it
      // shares the SXTX lowering. Scale > 1 has no 32-bit guest emitter yet.
      if (OffsetType != IR::MemOffsetType::UXTW || OffsetScale != 1) {
        LogMan::Msg::EFmt("WasmBackend: unsupported mem offset type {}", static_cast<uint32_t>(OffsetType));
        return false;
      }
    }
    uint32_t Off;
    if (!ResolveArg(IR, Offset, FE, &Off)) {
      return false;
    }
    uint32_t Shift = 0;
    if (OffsetScale != 1) {
      while ((1u << Shift) < OffsetScale) {
        ++Shift;
      }
      if ((1u << Shift) != OffsetScale) {
        LogMan::Msg::EFmt("WasmBackend: unsupported mem offset scale {}", OffsetScale);
        return false;
      }
    }
    PushAddrLocal(Base);
    PushLocalGet(FE.Body, Off);
    if (Shift != 0) {
      PushU32Const(FE.Body, Shift);
      FE.Body.push_back(0x74); // i32.shl
    }
    FE.Body.push_back(0x6A); // i32.add
    *MemArgOffset = 0;
    return true;
  }

  // Shared lowering for LoadMem/LoadMemTSO (identical struct shapes, no cast:
  // the template binds to the concrete op type). Result goes to DestLocal.
  template<typename OpT>
  bool EmitGPRLoad(const IR::IRListView* IR, const OpT* Op, IR::OpSize OpSize, FunctionEmitter& FE, uint32_t DestLocal) {
    const uint32_t Size = IR::OpSizeToSize(OpSize);
    if (Op->Class != IR::RegClass::GPR) {
      LogMan::Msg::EFmt("WasmBackend: LoadMem with unsupported register class {}", static_cast<uint32_t>(Op->Class));
      return false;
    }
    uint32_t MemArgOffset;
    if (!EmitEffectiveAddress(IR, Op->Addr, Op->Offset, Op->OffsetType, Op->OffsetScale, FE, &MemArgOffset)) {
      return false;
    }
    if (!PushGPRLoad(FE.Body, Size, MemArgOffset)) {
      LogMan::Msg::EFmt("WasmBackend: LoadMem with unsupported size {}", Size);
      return false;
    }
    if (Size == 8 && !FE.LocalIsI64(DestLocal)) {
      // The 8-byte GPR read (the segment chain's 8-byte GDT-entry load) lands
      // in the 32-bit model's default I32 dest unless a consumer typed it I64
      // (fild m64's shape): wrap to the guest's address width.
      FE.Body.push_back(0xA7); // i32.wrap_i64
    }
    PushLocalSet(FE.Body, DestLocal);
    return true;
  }

  // Shared lowering for StoreMem/StoreMemTSO.
  template<typename OpT>
  bool EmitGPRStore(const IR::IRListView* IR, const OpT* Op, IR::OpSize OpSize, FunctionEmitter& FE) {
    const uint32_t Size = IR::OpSizeToSize(OpSize);
    if (Op->Class != IR::RegClass::GPR) {
      LogMan::Msg::EFmt("WasmBackend: StoreMem with unsupported register class {}", static_cast<uint32_t>(Op->Class));
      return false;
    }
    uint32_t Value, MemArgOffset;
    if (!ResolveArg(IR, Op->Value, FE, &Value)) {
      return false;
    }
    if (!EmitEffectiveAddress(IR, Op->Addr, Op->Offset, Op->OffsetType, Op->OffsetScale, FE, &MemArgOffset)) {
      return false;
    }
    PushLocalGet(FE.Body, Value);
    if (!PushGPRStore(FE.Body, Size, MemArgOffset)) {
      LogMan::Msg::EFmt("WasmBackend: StoreMem with unsupported size {}", Size);
      return false;
    }
    return true;
  }

  // P3.1: the LOCK-prefixed RMW class. The IR ops' Dest is the *pre-op*
  // memory value (x86's returned value); the frontend computes flags from it
  // with the ordinary flag ops, so nothing extra is needed here. `Value` is
  // the source operand (for CAS: Expected then Desired — the wasm stack
  // order for cmpxchg is [addr, expected, replacement]).
  template<typename OpT>
  bool EmitAtomicFetch(const IR::IRListView* IR, const OpT* Op, AtomicRmw Kind, IR::OpSize OpSize, FunctionEmitter& FE, uint32_t DestLocal) {
    const uint32_t Size = IR::OpSizeToSize(OpSize);
    if (Size > 4) {
      LogMan::Msg::EFmt("WasmBackend: atomic RMW with unsupported size {}", Size);
      return false;
    }
    uint32_t Value, MemArgOffset;
    if (!ResolveArg(IR, Op->Value, FE, &Value)) {
      return false;
    }
    IR::OrderedNodeWrapper NoOffset{}; // the zero wrapper IsInvalid()s (atomic ops carry no offset field)
    if (!EmitEffectiveAddress(IR, Op->Addr, NoOffset, IR::MemOffsetType::SXTX, 1, FE, &MemArgOffset)) {
      return false;
    }
    PushLocalGet(FE.Body, Value);
    PushAtomicRmw(FE.Body, Kind, Size, MemArgOffset);
    PushLocalSet(FE.Body, DestLocal);
    return true;
  }

  bool EmitCAS(const IR::IRListView* IR, const IR::IROp_CAS* Op, IR::OpSize OpSize, FunctionEmitter& FE, uint32_t DestLocal) {
    const uint32_t Size = IR::OpSizeToSize(OpSize);
    if (Size > 4) {
      LogMan::Msg::EFmt("WasmBackend: CAS with unsupported size {}", Size);
      return false;
    }
    uint32_t Expected, Desired, MemArgOffset;
    if (!ResolveArg(IR, Op->Expected, FE, &Expected) || !ResolveArg(IR, Op->Desired, FE, &Desired)) {
      return false;
    }
    IR::OrderedNodeWrapper NoOffset{}; // the zero wrapper IsInvalid()s (atomic ops carry no offset field)
    if (!EmitEffectiveAddress(IR, Op->Addr, NoOffset, IR::MemOffsetType::SXTX, 1, FE, &MemArgOffset)) {
      return false;
    }
    PushLocalGet(FE.Body, Expected);
    PushLocalGet(FE.Body, Desired);
    PushAtomicRmw(FE.Body, AtomicRmw::Cmpxchg, Size, MemArgOffset);
    PushLocalSet(FE.Body, DestLocal);
    return true;
  }

  // P2.12: FPR-class memory access (the xmm register file's data path). The
  // FPR value is a v128 local; sub-vector loads zero the upper lanes
  // (v128.load32_zero/64_zero), sub-vector stores extract lane 0. UsesFPR is
  // latched so the epilogue knows to write the xmm statics back.
  template<typename OpT>
  bool EmitFPRLoad(const IR::IRListView* IR, const OpT* Op, IR::OpSize OpSize, FunctionEmitter& FE, uint32_t DestLocal) {
    const uint32_t Size = IR::OpSizeToSize(OpSize);
    uint32_t MemArgOffset;
    if (!EmitEffectiveAddress(IR, Op->Addr, Op->Offset, Op->OffsetType, Op->OffsetScale, FE, &MemArgOffset)) {
      return false;
    }
    switch (Size) {
    case 4: PushV128Load32Zero(FE.Body, MemArgOffset); break;
    case 8: PushV128Load64Zero(FE.Body, MemArgOffset); break;
    case 16: PushV128Load(FE.Body, MemArgOffset); break;
    default:
      LogMan::Msg::EFmt("WasmBackend: FPR LoadMem with unsupported size {}", Size);
      return false;
    }
    PushLocalSet(FE.Body, DestLocal);
    return true;
  }

  template<typename OpT>
  bool EmitFPRStore(const IR::IRListView* IR, const OpT* Op, IR::OpSize OpSize, FunctionEmitter& FE, uint32_t ValueLocal) {
    const uint32_t Size = IR::OpSizeToSize(OpSize);
    uint32_t MemArgOffset;
    if (!EmitEffectiveAddress(IR, Op->Addr, Op->Offset, Op->OffsetType, Op->OffsetScale, FE, &MemArgOffset)) {
      return false;
    }
    PushLocalGet(FE.Body, ValueLocal);
    switch (Size) {
    case 4:
      FE.Body.push_back(0xFD);
      FE.Body.push_back(0x1B); // i32x4.extract_lane
      FE.Body.push_back(0);
      PushI32Store(FE.Body, MemArgOffset);
      break;
    case 8:
      FE.Body.push_back(0xFD);
      FE.Body.push_back(0x1D); // i64x2.extract_lane
      FE.Body.push_back(0);
      PushI64Store(FE.Body, MemArgOffset);
      break;
    case 16: PushV128Store(FE.Body, MemArgOffset); break;
    default:
      LogMan::Msg::EFmt("WasmBackend: FPR StoreMem with unsupported size {}", Size);
      return false;
    }
    return true;
  }

  // Address of the x87 stack slot (top + Offset) & 7 inside the context: the
  // mm array is 8 slots of 16 bytes, TOP is the 3-bit stack pointer byte.
  void EmitStackSlotAddr(FunctionEmitter& FE, uint64_t ContextBase, uint8_t Offset) {
    PushU32Const(FE.Body, 0);
    PushI32Load8U(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.flags) + X86State::X87FLAG_TOP_LOC);
    PushU32Const(FE.Body, Offset);
    FE.Body.push_back(0x6A); // i32.add
    PushU32Const(FE.Body, 7);
    FE.Body.push_back(0x71); // i32.and
    PushU32Const(FE.Body, 4);
    FE.Body.push_back(0x74); // i32.shl
    PushU32Const(FE.Body, static_cast<uint32_t>(ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.mm)));
    FE.Body.push_back(0x6A); // i32.add
  }

  // top = (top + Delta) & 7 — the x87 stack pointer update for push (7) and
  // pop/inc (1).
  void EmitSetX87Top(FunctionEmitter& FE, uint64_t ContextBase, uint32_t Delta) {
    PushU32Const(FE.Body, 0);
    PushU32Const(FE.Body, 0);
    PushI32Load8U(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.flags) + X86State::X87FLAG_TOP_LOC);
    PushU32Const(FE.Body, Delta);
    FE.Body.push_back(0x6A); // i32.add
    PushU32Const(FE.Body, 7);
    FE.Body.push_back(0x71); // i32.and
    PushI32Store8(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.flags) + X86State::X87FLAG_TOP_LOC);
  }

  // Lane-0 float extraction from a v128 local (the scalar view of an xmm).
  void PushF64FromV128(FunctionEmitter& FE, uint32_t Local) {
    PushLocalGet(FE.Body, Local);
    PushVec(FE.Body, {0xFD, 0x21, 0x00}); // f64x2.extract_lane 0
  }

  // --- the x87 Invalid-Operation exception flag (FSW bit 0, sticky) ----------
  //
  // The F80 arithmetic tier computes in native f64, so the flag is derived
  // from the operand bit patterns: the IEEE invalid conditions of the op,
  // plus a signaling-NaN operand — matching the interpreter's SoftFloat
  // fallback, which ORs softfloat_flag_invalid into X87FLAG_IE_LOC
  // (F80Fallbacks.h HandleX87Exception). fnstsw reconstructs the FSW from
  // that byte (ReconstructFSW_Helper), so this is the only place the wasm
  // backend can raise it. The flag byte only ever holds 0/1 (fnclex/fninit
  // clear it, everything else sets it), so a conditional store of 1 is the
  // sticky OR. Operand bits ride two i64 scratch locals (f64.reinterpret).
  enum class X87InvalidShape { Add, Sub, Mul, Div };

  void EmitX87InvalidFlagCheck(FunctionEmitter& FE, uint64_t ContextBase, uint32_t A, uint32_t B, X87InvalidShape Shape) {
    auto IsInf = [&](uint32_t L) { // |x| == inf
      PushLocalGet(FE.Body, L);
      FE.Body.push_back(0xBF); // f64.reinterpret_i64
      FE.Body.push_back(0x99); // f64.abs
      PushF64Const(FE.Body, 0x7FF0000000000000ull);
      FE.Body.push_back(0x61); // f64.eq
    };
    auto IsZero = [&](uint32_t L) {
      PushLocalGet(FE.Body, L);
      FE.Body.push_back(0xBF);
      FE.Body.push_back(0x99);
      PushF64Const(FE.Body, 0);
      FE.Body.push_back(0x61);
    };
    auto SignBit = [&](uint32_t L) { // -> i32 0/1
      PushLocalGet(FE.Body, L);
      PushU64Const(FE.Body, 63);
      FE.Body.push_back(0x88); // i64.shr_u
      FE.Body.push_back(0xA7); // i32.wrap_i64
    };
    auto IsSNaN = [&](uint32_t L) { // exponent all-ones AND quiet bit clear
      PushLocalGet(FE.Body, L);
      PushU64Const(FE.Body, 52);
      FE.Body.push_back(0x88);
      PushU64Const(FE.Body, 0x7FF);
      FE.Body.push_back(0x83); // i64.and
      PushU64Const(FE.Body, 0x7FF);
      FE.Body.push_back(0x51); // i64.eq
      PushLocalGet(FE.Body, L);
      PushU64Const(FE.Body, 51);
      FE.Body.push_back(0x88);
      PushU64Const(FE.Body, 1);
      FE.Body.push_back(0x83);
      FE.Body.push_back(0x50); // i64.eqz
      FE.Body.push_back(0x71); // i32.and
    };

    IsSNaN(A);
    IsSNaN(B);
    FE.Body.push_back(0x72); // i32.or
    switch (Shape) {
    case X87InvalidShape::Div:
      IsZero(A); IsZero(B); FE.Body.push_back(0x71);
      IsInf(A); IsInf(B); FE.Body.push_back(0x71);
      FE.Body.push_back(0x72);
      FE.Body.push_back(0x72);
      break;
    case X87InvalidShape::Mul:
      IsZero(A); IsInf(B); FE.Body.push_back(0x71);
      IsInf(A); IsZero(B); FE.Body.push_back(0x71);
      FE.Body.push_back(0x72);
      FE.Body.push_back(0x72);
      break;
    case X87InvalidShape::Add: // both infinite, opposite signs
      IsInf(A); IsInf(B); FE.Body.push_back(0x71);
      SignBit(A); SignBit(B); FE.Body.push_back(0x73); // i32.xor
      FE.Body.push_back(0x71);
      FE.Body.push_back(0x72);
      break;
    case X87InvalidShape::Sub: // both infinite, same sign
      IsInf(A); IsInf(B); FE.Body.push_back(0x71);
      SignBit(A); SignBit(B); FE.Body.push_back(0x73);
      FE.Body.push_back(0x45); // i32.eqz
      FE.Body.push_back(0x71);
      FE.Body.push_back(0x72);
      break;
    }
    FE.Body.push_back(0x04); // if
    FE.Body.push_back(0x40); // void
    PushU32Const(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.flags) + X86State::X87FLAG_IE_LOC);
    PushU32Const(FE.Body, 1);
    PushI32Store8(FE.Body, 0);
    FE.Body.push_back(0x0B); // end
  }

  void PushF32FromV128(FunctionEmitter& FE, uint32_t Local) {
    PushLocalGet(FE.Body, Local);
    PushVec(FE.Body, {0xFD, 0x1F, 0x00}); // f32x4.extract_lane 0
  }

  // Scalar float binop opcode: f32 add/sub/mul/div = 0x92-0x95, f64 = 0xA0-0xA3.
  uint8_t FloatBinOp(IR::IROps Op, uint32_t Bits) {
    const uint8_t Idx = Op == IR::IROps::OP_F80ADDSTACK || Op == IR::IROps::OP_F80ADDVALUE || Op == IR::IROps::OP_VFADDSCALARINSERT ? 0
      : Op == IR::IROps::OP_F80SUBSTACK || Op == IR::IROps::OP_F80SUBVALUE || Op == IR::IROps::OP_F80SUBRVALUE
          || Op == IR::IROps::OP_VFSUBSCALARINSERT
        ? 1
        : Op == IR::IROps::OP_F80MULSTACK || Op == IR::IROps::OP_F80MULVALUE || Op == IR::IROps::OP_VFMULSCALARINSERT ? 2
        : Op == IR::IROps::OP_F80DIVSTACK || Op == IR::IROps::OP_F80DIVVALUE || Op == IR::IROps::OP_F80DIVRVALUE
            || Op == IR::IROps::OP_VFDIVSCALARINSERT
          ? 3
          : 0xFF;
    return (Bits == 32 ? 0x92 : 0xA0) + Idx;
  }

  // Packed float binop: the SIMD opcode is a multi-byte LEB. f32x4
  // add/sub/mul/div = 0xE4-0xE7, f64x2 = 0xF0-0xF3 (all >= 0x80, so the
  // emitted opcode bytes carry the LEB continuation).
  void PushVectorFloatOp(fextl::vector<uint8_t>& Out, IR::IROps Op, uint32_t ElemBits) {
    const uint8_t Idx = Op == IR::IROps::OP_VFADD ? 0
      : Op == IR::IROps::OP_VFSUB ? 1
      : Op == IR::IROps::OP_VFMUL ? 2
      : Op == IR::IROps::OP_VFDIV ? 3
                                   : 0xFF;
    const uint8_t Opcode = (ElemBits == 32 ? 0xE4 : 0xF0) + Idx;
    Out.push_back(0xFD);
    Out.push_back(Opcode);
    Out.push_back(0x01); // the opcode's LEB continuation byte
  }

  // Assemble the compare results: the Arm-shaped NZCV into the NZCV local
  // and, for FCOMI-style consumers, the FCMP_FLAG bits (bit0=EQ bit1=LT
  // bit2=UNORDERED) into the GPR result.
  void EmitCompareResult(FunctionEmitter& FE, uint32_t LT, uint32_t EQ, uint32_t UNORD, uint32_t Dest, bool WantResult) {
    PushLocalGet(FE.Body, LT);
    PushU32Const(FE.Body, 31);
    FE.Body.push_back(0x74); // i32.shl
    PushLocalGet(FE.Body, EQ);
    PushU32Const(FE.Body, 30);
    FE.Body.push_back(0x74);
    FE.Body.push_back(0x72); // i32.or
    PushLocalGet(FE.Body, LT);
    FE.Body.push_back(0x45); // i32.eqz -> !lt
    PushLocalGet(FE.Body, UNORD);
    FE.Body.push_back(0x72); // i32.or
    PushU32Const(FE.Body, 29);
    FE.Body.push_back(0x74);
    FE.Body.push_back(0x72); // i32.or
    PushLocalGet(FE.Body, UNORD);
    PushU32Const(FE.Body, 28);
    FE.Body.push_back(0x74);
    FE.Body.push_back(0x72); // i32.or
    PushLocalSet(FE.Body, FE.NZCV());
    if (!WantResult) {
      return;
    }
    PushLocalGet(FE.Body, EQ);
    PushLocalGet(FE.Body, LT);
    PushU32Const(FE.Body, 1);
    FE.Body.push_back(0x74);
    FE.Body.push_back(0x72); // i32.or
    PushLocalGet(FE.Body, UNORD);
    PushU32Const(FE.Body, 2);
    FE.Body.push_back(0x74);
    FE.Body.push_back(0x72); // i32.or
    PushLocalSet(FE.Body, Dest);
  }

  // Scalar compare of two v128 locals at ElemBits width; writes the Arm NZCV
  // (and optionally the FCMP_FLAG result) — the FCmp/UCOMIS lowering.
  bool EmitFloatCompare(FunctionEmitter& FE, uint32_t A, uint32_t Bv, uint32_t ElemBits, uint32_t Dest, bool WantResult) {
    const uint32_t LT = FE.AllocScratch();
    const uint32_t EQ = FE.AllocScratch();
    const uint32_t UNORD = FE.AllocScratch();
    auto Extract = [&](uint32_t Local) {
      PushLocalGet(FE.Body, Local);
      if (ElemBits == 32) {
        PushVec(FE.Body, {0xFD, 0x1F, 0x00}); // f32x4.extract_lane 0
      } else {
        PushVec(FE.Body, {0xFD, 0x21, 0x00}); // f64x2.extract_lane 0
      }
    };
    auto Cmp = [&](uint8_t Op32, uint8_t Op64) { FE.Body.push_back(ElemBits == 32 ? Op32 : Op64); };
    Extract(A);
    Extract(Bv);
    Cmp(0x5D, 0x63); // lt
    PushLocalSet(FE.Body, LT);
    Extract(A);
    Extract(Bv);
    Cmp(0x5B, 0x61); // eq
    PushLocalSet(FE.Body, EQ);
    // unordered: either operand is NaN (self-equality fails)
    Extract(A);
    Extract(A);
    Cmp(0x5B, 0x61);
    FE.Body.push_back(0x45); // i32.eqz
    Extract(Bv);
    Extract(Bv);
    Cmp(0x5B, 0x61);
    FE.Body.push_back(0x45);
    FE.Body.push_back(0x72); // i32.or
    PushLocalSet(FE.Body, UNORD);
    EmitCompareResult(FE, LT, EQ, UNORD, Dest, WantResult);
    return true;
  }

  void EmitPrologue(FunctionEmitter& FE, uint64_t ContextBase) {
    for (uint32_t i = 0; i < 16; ++i) {
      PushU32Const(FE.Body, 0);
      PushI32Load(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.gregs[i]));
      PushLocalSet(FE.Body, FE.StaticLocal(i));
    }
    PushU32Const(FE.Body, 0);
    PushI32Load(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.pf_raw));
    PushLocalSet(FE.Body, FE.StaticLocal(16));
    PushU32Const(FE.Body, 0);
    PushI32Load(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.af_raw));
    PushLocalSet(FE.Body, FE.StaticLocal(17));
    PushU32Const(FE.Body, 0);
    PushI32Load(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.flags) + X86State::RFLAG_NZCV_LOC);
    PushLocalSet(FE.Body, FE.NZCV());
    if (UsesFPR) {
      // The xmm register file only syncs when the block touches FPR state
      // (integer-only blocks keep the D2 prologue shape).
      for (uint32_t i = 0; i < FunctionEmitter::NUM_STATIC_FPR; ++i) {
        PushU32Const(FE.Body, 0);
        PushV128Load(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.xmm) + i * 16);
        PushLocalSet(FE.Body, FE.StaticFPRLocal(i));
      }
    }
  }

  bool EmitExit(const IR::IRListView* IR, uint64_t EntryRIP, IR::OrderedNodeWrapper NewRIP, FunctionEmitter& FE, uint64_t ContextBase,
                uint64_t* InlineRIP) {
    // RIP source: either an inline constant (direct branches, D2) or the
    // register/local holding the computed value (e.g. `ret` pops it, D3a).
    *InlineRIP = 0;
    bool HaveRIPValue = false;
    uint32_t RIPLocal = 0;
    uint64_t NewRIPValue = 0;
    if (NewRIP.IsImmediate()) {
      IR::PhysicalRegister PhyReg(NewRIP);
      if (PhyReg.AsRegClass() == IR::RegClass::GPRFixed && PhyReg.Reg < NUM_STATIC_SLOTS) {
        RIPLocal = FE.StaticLocal(PhyReg.Reg);
        HaveRIPValue = true;
      } else if (PhyReg.AsRegClass() == IR::RegClass::GPR && PhyReg.Reg < NUM_DYNAMIC_REGS && DynamicLastDef[PhyReg.Reg] != 0) {
        RIPLocal = DynamicLastDef[PhyReg.Reg] - 1;
        HaveRIPValue = true;
      } else {
        LogMan::Msg::EFmt("WasmBackend: ExitFunction NewRIP register {} unsupported", PhyReg.Reg);
        return false;
      }
    } else {
      auto* NH = IR->GetOp<IR::IROp_Header>(IR->GetNode(NewRIP));
      if (NH->Op == IR::IROps::OP_INLINEENTRYPOINTOFFSET) {
        NewRIPValue = EntryRIP + NH->C<IR::IROp_InlineEntrypointOffset>()->Offset;
      } else if (NH->Op == IR::IROps::OP_INLINECONSTANT) {
        NewRIPValue = NH->C<IR::IROp_InlineConstant>()->Constant;
      } else {
        LogMan::Msg::EFmt("WasmBackend: ExitFunction NewRIP is a computed value");
        return false;
      }
      *InlineRIP = NewRIPValue;
    }

    for (uint32_t i = 0; i < 16; ++i) {
      PushU32Const(FE.Body, 0);
      PushLocalGet(FE.Body, FE.StaticLocal(i));
      PushI32Store(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.gregs[i]));
    }
    PushU32Const(FE.Body, 0);
    PushLocalGet(FE.Body, FE.StaticLocal(16));
    PushI32Store(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.pf_raw));
    PushU32Const(FE.Body, 0);
    PushLocalGet(FE.Body, FE.StaticLocal(17));
    PushI32Store(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.af_raw));
    PushU32Const(FE.Body, 0);
    PushLocalGet(FE.Body, FE.NZCV());
    PushI32Store(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.flags) + X86State::RFLAG_NZCV_LOC);
    if (UsesFPR) {
      for (uint32_t i = 0; i < FunctionEmitter::NUM_STATIC_FPR; ++i) {
        PushU32Const(FE.Body, 0);
        PushLocalGet(FE.Body, FE.StaticFPRLocal(i));
        PushV128Store(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.xmm) + i * 16);
      }
    }

    // RIP is a u64 in the context; write the full width.
    PushU32Const(FE.Body, 0);
    if (HaveRIPValue) {
      PushLocalGet(FE.Body, RIPLocal);
      FE.Body.push_back(0xAD); // i64.extend_i32_u
    } else {
      PushU64Const(FE.Body, NewRIPValue);
    }
    PushI64Store(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.rip));
    return true;
  }

  uint32_t DynamicLastDef[NUM_DYNAMIC_REGS] {};
  uint32_t FPRLastDef[NUM_DYNAMIC_REGS] {};
  // Set during emission when any op touches FPR state; drives the
  // conditional xmm static-slot sync in the prologue/epilogue.
  bool UsesFPR = false;

  fextl::vector<uint8_t> Module;

  // P1.10: the IRList of a guest block ending in a conditional branch has
  // three code blocks (entry + two exits); CondJump routes between them.
  fextl::vector<IR::OrderedNode*> BlockNodes;
  fextl::vector<uint8_t> VisitedBlocks;
  bool AnyExitTargetsThunk = false;
  // 64-bit products (UMul/SMull): the node's local holds the low half, the
  // high half lives in a scratch local keyed by the low local's index.
  std::unordered_map<uint32_t, uint32_t> I64HiByLocal;
  uint32_t I64ScratchCount = 0;

  // Resolve a CondJump/Jump target (a CodeBlock node reference) to a block
  // index.
  bool ResolveBlockIndex(const IR::IRListView* IR, IR::OrderedNodeWrapper Target, uint32_t* Out) {
    if (Target.IsInvalid()) {
      LogMan::Msg::EFmt("WasmBackend: branch target is invalid");
      return false;
    }
    const uint32_t TargetID = IR->GetID(IR->GetNode(Target)).Value;
    for (uint32_t I = 0; I < BlockNodes.size(); ++I) {
      if (IR->GetID(BlockNodes[I]).Value == TargetID) {
        *Out = I;
        return true;
      }
    }
    LogMan::Msg::EFmt("WasmBackend: branch target is not a block of this IRList");
    return false;
  }

  // Emit one IR code block. Blocks are emitted as a structured wasm tree: a
  // block ending in CondJump(T, F) becomes
  //   <cond> -> br_if $F ; <T>; end ; <F>
  // where T and F are emitted recursively (fresh forward blocks — the
  // frontend creates both targets when multiblock is off). A backward edge
  // (re-entering an already-emitted block) needs a wasm relooper and fails
  // loudly.
  bool EmitCodeBlock(const IR::IRListView* IR, uint64_t EntryRIP, uint64_t ContextBase, uint64_t ThunkBase, uint64_t ThunkSize,
                     FunctionEmitter& FE, uint32_t BlockIdx);

  // Shared lowering for TestNZ/TestZ (identical struct shapes): NZ from
  // (Src1 & Src2) masked to the op size, C = V = 0 (the Arm64 `cmn` contract
  // used by SetNZ_ZeroCV's small path).
  bool EmitTestFlags(const IR::IRListView* IR, IR::OrderedNodeWrapper Src1, IR::OrderedNodeWrapper Src2, IR::OpSize Size,
                     FunctionEmitter& FE, uint32_t Dest) {
    uint32_t A, B;
    if (!ResolveArg(IR, Src1, FE, &A) || !ResolveArg(IR, Src2, FE, &B)) {
      return false;
    }
    const uint32_t Bits = IR::OpSizeAsBits(Size);
    if (Bits == 0 || Bits > 32) {
      LogMan::Msg::EFmt("WasmBackend: Test with unsupported size {}", Bits);
      return false;
    }
    PushLocalGet(FE.Body, A);
    PushLocalGet(FE.Body, B);
    FE.Body.push_back(0x71); // i32.and
    if (Bits != 32) {
      PushU32Const(FE.Body, (1u << Bits) - 1);
      FE.Body.push_back(0x71); // i32.and
    }
    PushLocalSet(FE.Body, Dest);
    // N = sign bit of the masked value, moved to the NZCV's N position (bit
    // 31): for sub-32-bit tests (test bl,bl) the masked sign bit sits at
    // (Bits-1) — the 8-bit `test bl,bl; js` shape read a clear N and the
    // signed branch never took (the EH type_info decode's indirect deref —
    // the F3e wall). Z = value == 0, C = V = 0.
    PushLocalGet(FE.Body, Dest);
    PushU32Const(FE.Body, 1u << (Bits - 1));
    FE.Body.push_back(0x71); // i32.and
    if (Bits != 32) {
      PushU32Const(FE.Body, 32 - Bits);
      FE.Body.push_back(0x74); // i32.shl
    }
    PushLocalGet(FE.Body, Dest);
    FE.Body.push_back(0x45); // i32.eqz
    PushU32Const(FE.Body, 30);
    FE.Body.push_back(0x74); // i32.shl
    FE.Body.push_back(0x72); // i32.or
    PushLocalSet(FE.Body, FE.NZCV());
    return true;
  }
};

bool ModuleBuilder::EmitCodeBlock(const IR::IRListView* IR, uint64_t EntryRIP, uint64_t ContextBase, uint64_t ThunkBase,
                                  uint64_t ThunkSize, FunctionEmitter& FE, uint32_t BlockIdx) {
  if (BlockIdx >= BlockNodes.size()) {
    LogMan::Msg::EFmt("WasmBackend: block index {} out of range", BlockIdx);
    return false;
  }
  if (VisitedBlocks[BlockIdx]) {
    LogMan::Msg::EFmt("WasmBackend: backward branch into IR block {} unsupported (no relooper)", BlockIdx);
    return false;
  }
  VisitedBlocks[BlockIdx] = true;

  fextl::vector<std::pair<IR::OrderedNode*, IR::IROp_Header*>> Ops;
  for (auto [CodeNode, IROp] : IR->GetCode(BlockNodes[BlockIdx])) {
    Ops.emplace_back(CodeNode, IROp);
  }
  if (Ops.empty()) {
    LogMan::Msg::EFmt("WasmBackend: IR block {} is empty", BlockIdx);
    return false;
  }
  // The terminal op is the last op that isn't block bookkeeping (every code
  // block is wrapped in BeginBlock/EndBlock).
  auto IsBookkeeping = [](IR::IROps Op) {
    return Op == IR::IROps::OP_IRHEADER || Op == IR::IROps::OP_CODEBLOCK || Op == IR::IROps::OP_BEGINBLOCK
        || Op == IR::IROps::OP_ENDBLOCK;
  };
  size_t TerminalIdx = Ops.size();
  for (size_t I = Ops.size(); I-- > 0;) {
    if (!IsBookkeeping(Ops[I].second->Op)) {
      TerminalIdx = I;
      break;
    }
  }
  if (TerminalIdx == Ops.size()) {
    LogMan::Msg::EFmt("WasmBackend: IR block {} contains no ops", BlockIdx);
    return false;
  }
  const auto LastOp = Ops[TerminalIdx].second->Op;
  if (LastOp != IR::IROps::OP_EXITFUNCTION && LastOp != IR::IROps::OP_CONDJUMP && LastOp != IR::IROps::OP_JUMP) {
    LogMan::Msg::EFmt("WasmBackend: IR block {} does not end with a branch/exit op (last: {})", BlockIdx, IR::GetName(LastOp).data());
    return false;
  }

  bool WroteStaticThisNode = false;
  for (auto& [CodeNode, IROp] : Ops) {
    const uint32_t NodeID = IR->GetID(CodeNode).Value;
    const uint32_t Dest = FE.NodeLocal(NodeID);
    const bool IsLast = CodeNode == Ops[TerminalIdx].first;

    switch (IROp->Op) {
    case IR::IROps::OP_IRHEADER:
    case IR::IROps::OP_CODEBLOCK:
    case IR::IROps::OP_BEGINBLOCK:
    case IR::IROps::OP_ENDBLOCK:
    case IR::IROps::OP_GUESTOPCODE:
    case IR::IROps::OP_INVALIDATEFLAGS:
      // Block bookkeeping / debug metadata / RA hints: no code.
      // (GuestOpcode is emitted for every instruction whose operands could
      // touch memory, which the conservative CanHaveSideEffects check
      // includes GPR writes for. InvalidateFlags is a JITDispatchOverride
      // NoOp — a pure flag-cache invalidation hint for the RFC pass.)
      continue;

    case IR::IROps::OP_CONSTANT: {
      // A constant that does not fit u32 was pre-typed I64 (it only exists
      // as a 64-bit GPR value — e.g. the FLDF64 bit-pattern constant).
      if (FE.NodeIsI64(IR->GetID(CodeNode).Value)) {
        PushU64Const(FE.Body, IROp->C<IR::IROp_Constant>()->Constant);
      } else {
        PushU32Const(FE.Body, static_cast<uint32_t>(IROp->C<IR::IROp_Constant>()->Constant));
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_INLINECONSTANT: {
      PushU32Const(FE.Body, static_cast<uint32_t>(IROp->C<IR::IROp_InlineConstant>()->Constant));
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_INLINEENTRYPOINTOFFSET: {
      PushU32Const(FE.Body, static_cast<uint32_t>(EntryRIP + IROp->C<IR::IROp_InlineEntrypointOffset>()->Offset));
      PushLocalSet(FE.Body, Dest);
      break;
    }

    case IR::IROps::OP_ADD: {
      uint32_t A, B;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A) || !ResolveArg(IR, IROp->Args[1], FE, &B)) {
        return false;
      }
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(0x6A); // i32.add
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_SUB: {
      uint32_t A, B;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A) || !ResolveArg(IR, IROp->Args[1], FE, &B)) {
        return false;
      }
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(0x6B); // i32.sub
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_OR: {
      uint32_t A, B;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A) || !ResolveArg(IR, IROp->Args[1], FE, &B)) {
        return false;
      }
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(0x72); // i32.or
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_AND: {
      uint32_t A, B;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A) || !ResolveArg(IR, IROp->Args[1], FE, &B)) {
        return false;
      }
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(0x71); // i32.and
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_ANDN: {
      // Src1 & ~Src2 — the IR.json contract, pinned by the AArch64 host
      // lowering (ALUOps.cpp DEF_BINOP_WITH_CONSTANT(Andn, bic, bic):
      // `bic dest, Src1, Src2`). Wasm has no and-not primitive, so invert
      // Src2 via the OP_NOT form (the -1 xor) and AND. Emitted by the
      // sub-32-bit ADD/SUB/SBB/ADC flag paths (CalculateOF with a constant
      // source: `sbb $0xff,%al` in libstdc++'s _M_widen_init — the FD3dc
      // wall). Size is i32 or i64 per the IR's EmitValidation; the wasm
      // backend's GPR model types every Andn node I32 (the i64 group is
      // the fild/movq special cases only), so the i64 form is a named gap.
      if (IROp->Size != IR::OpSize::i32Bit) {
        LogMan::Msg::EFmt("WasmBackend: Andn with unsupported size {} (only the 32-bit GPR model is lowered)", IR::OpSizeAsBits(IROp->Size));
        return false;
      }
      uint32_t A, B;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A) || !ResolveArg(IR, IROp->Args[1], FE, &B)) {
        return false;
      }
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      PushU32Const(FE.Body, 0xFFFFFFFF);
      FE.Body.push_back(0x73); // i32.xor (bitwise NOT of Src2)
      FE.Body.push_back(0x71); // i32.and — Src1 & ~Src2
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_ORLSHR: {
      // Src1 | (Src2 >> BitShift) — the segment-base merge idiom
      // (UpdatePrefixFromSegment's high-word extraction). The 32-bit model:
      // an I64-typed source (the 8-byte GDT-entry read) wraps to i32 before
      // the shift/or — the op's i32 OpSize is the guest's address width,
      // and the meaningful bits of the merge live in the low 32.
      // The i64Bit form is the packed-RFLAG flag assembly's shape
      // (GetPackedRFLAG's SF/ZF OR-in) — named loud-fail: the flag
      // materialization around it (the deferred AF/PF read idiom) has its
      // own never-exercised gaps, so the LAHF/PUSHF family stays a named
      // skip (the corpus's 'orlshr' family) until that lands as a unit.
      const auto* Op = IROp->C<IR::IROp_Orlshr>();
      if (IROp->Size != IR::OpSize::i32Bit) {
        LogMan::Msg::EFmt("WasmBackend: Orlshr with unsupported size {} (the i64 flag-assembly form is a named gap)", IR::OpSizeAsBits(IROp->Size));
        return false;
      }
      uint32_t S1, S2;
      if (!ResolveArg(IR, Op->Src1, FE, &S1) || !ResolveArg(IR, Op->Src2, FE, &S2)) {
        return false;
      }
      PushLocalGet(FE.Body, S1);
      if (FE.LocalIsI64(S1)) {
        FE.Body.push_back(0xA7); // i32.wrap_i64
      }
      PushLocalGet(FE.Body, S2);
      if (FE.LocalIsI64(S2)) {
        FE.Body.push_back(0xA7); // i32.wrap_i64
      }
      PushU32Const(FE.Body, Op->BitShift);
      FE.Body.push_back(0x76); // i32.shr_u
      FE.Body.push_back(0x72); // i32.or
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_XOR: {
      uint32_t A, B;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A) || !ResolveArg(IR, IROp->Args[1], FE, &B)) {
        return false;
      }
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(0x73); // i32.xor
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_NOT: {
      uint32_t A;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A)) {
        return false;
      }
      PushLocalGet(FE.Body, A);
      PushU32Const(FE.Body, 0xFFFFFFFF);
      FE.Body.push_back(0x73); // i32.xor (bitwise NOT)
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_BFE: {
      // Byte-register model: 8/16-bit GPR reads extract from the 32-bit
      // static slot (LoadGPRRegister lowers to Bfe).
      const auto* Op = IROp->C<IR::IROp_Bfe>();
      uint32_t A;
      if (!ResolveArg(IR, Op->Src, FE, &A)) {
        return false;
      }
      // A 64-bit product's halves are extracted straight from the paired
      // locals: low is the node's own local, high the keyed scratch.
      const auto HiIt = I64HiByLocal.find(A);
      if (HiIt != I64HiByLocal.end()) {
        if (Op->lsb == 0 && Op->Width == 32) {
          PushLocalGet(FE.Body, A); // already the low half
          PushLocalSet(FE.Body, Dest);
          break;
        }
        if (Op->lsb == 32 && Op->Width == 32) {
          PushLocalGet(FE.Body, HiIt->second);
          PushLocalSet(FE.Body, Dest);
          break;
        }
        LogMan::Msg::EFmt("WasmBackend: Bfe {}:{} of a 64-bit product unsupported", Op->lsb, Op->Width);
        return false;
      }
      if (Op->Width == 0 || Op->lsb + Op->Width > 32) {
        LogMan::Msg::EFmt("WasmBackend: Bfe {}:{} out of range", Op->lsb, Op->Width);
        return false;
      }
      PushLocalGet(FE.Body, A);
      if (Op->lsb != 0) {
        PushU32Const(FE.Body, Op->lsb);
        FE.Body.push_back(0x76); // i32.shr_u
      }
      if (Op->Width != 32) {
        PushU32Const(FE.Body, (1u << Op->Width) - 1);
        FE.Body.push_back(0x71); // i32.and
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_RMWHANDLE: {
      // A poison-guard pass-through move (the RET family's SP handling):
      // the result is simply the wrapped value's local.
      const auto* Op = IROp->C<IR::IROp_RMWHandle>();
      uint32_t Src;
      if (!ResolveArg(IR, Op->Value, FE, &Src)) {
        return false;
      }
      PushLocalGet(FE.Body, Src);
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_SBFE: {
      // Sign-extending extract: shift left then arithmetic shift right.
      const auto* Op = IROp->C<IR::IROp_Sbfe>();
      uint32_t A;
      if (!ResolveArg(IR, Op->Src, FE, &A)) {
        return false;
      }
      if (Op->Width == 64 && Op->lsb == 0) {
        // The fild m64 chain's sign-extend: from an i64 node it is a copy,
        // from an i32 node an extension (P2.12's I64 locals).
        PushLocalGet(FE.Body, A);
        PushLocalSet(FE.Body, Dest);
        break;
      }
      PushLocalGet(FE.Body, A);
      const auto HiIt = I64HiByLocal.find(A);
      if (HiIt != I64HiByLocal.end()) {
        if (Op->lsb == 0 && Op->Width == 32) {
          // already the low half
          PushLocalSet(FE.Body, Dest);
          break;
        }
        if (Op->lsb == 32 && Op->Width == 32) {
          PushLocalGet(FE.Body, HiIt->second);
          PushLocalSet(FE.Body, Dest);
          break;
        }
        LogMan::Msg::EFmt("WasmBackend: Sbfe {}:{} of a 64-bit product unsupported", Op->lsb, Op->Width);
        return false;
      }
      if (Op->Width == 0 || Op->Width > 32 || Op->lsb + Op->Width > 32) {
        LogMan::Msg::EFmt("WasmBackend: Sbfe {}:{} out of range", Op->lsb, Op->Width);
        return false;
      }
      if (Op->Width != 32) {
        const uint32_t LeftShift = 32 - Op->lsb - Op->Width;
        const uint32_t RightShift = 32 - Op->Width;
        if (LeftShift != 0) {
          PushU32Const(FE.Body, LeftShift);
          FE.Body.push_back(0x74); // i32.shl
        }
        PushU32Const(FE.Body, RightShift);
        FE.Body.push_back(0x75); // i32.shr_s
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_BFI: {
      // Insert Src's low Width bits at lsb into Dest (the containing GPR's
      // old value — the byte-register write-back model).
      const auto* Op = IROp->C<IR::IROp_Bfi>();
      uint32_t D, S;
      if (!ResolveArg(IR, Op->Dest, FE, &D) || !ResolveArg(IR, Op->Src, FE, &S)) {
        return false;
      }
      if (Op->Width == 0 || Op->Width > 32 || Op->lsb + Op->Width > 32) {
        LogMan::Msg::EFmt("WasmBackend: Bfi {}:{} out of range", Op->lsb, Op->Width);
        return false;
      }
      if (Op->Width == 32) {
        PushLocalGet(FE.Body, S);
      } else {
        const uint32_t Mask = (1u << Op->Width) - 1;
        // (Dest & ~(Mask << lsb)) | ((Src & Mask) << lsb)
        PushLocalGet(FE.Body, D);
        PushU32Const(FE.Body, ~(Mask << Op->lsb));
        FE.Body.push_back(0x71); // i32.and
        PushLocalGet(FE.Body, S);
        PushU32Const(FE.Body, Mask);
        FE.Body.push_back(0x71); // i32.and
        PushU32Const(FE.Body, Op->lsb);
        FE.Body.push_back(0x74); // i32.shl
        FE.Body.push_back(0x72); // i32.or
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_ADDWITHFLAGS: {
      // Result = Src1 + Src2; NZCV recomputed (Arm layout: N=31 Z=30 C=29 V=28).
      uint32_t A, B;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A) || !ResolveArg(IR, IROp->Args[1], FE, &B)) {
        return false;
      }
      // res = a + b
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(0x6A); // i32.add
      PushLocalSet(FE.Body, Dest);
      // N = res & 0x80000000
      PushLocalGet(FE.Body, Dest);
      PushU32Const(FE.Body, 0x80000000);
      FE.Body.push_back(0x71); // i32.and
      // Z = (res == 0) ? 1<<30 : 0
      PushLocalGet(FE.Body, Dest);
      FE.Body.push_back(0x45); // i32.eqz
      PushU32Const(FE.Body, 30);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      // C = unsigned carry-out of a + b
      PushLocalGet(FE.Body, A);
      FE.Body.push_back(0xAD); // i64.extend_i32_u
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(0xAD); // i64.extend_i32_u
      FE.Body.push_back(0x7C); // i64.add
      PushU64Const(FE.Body, 0x100000000ULL);
      FE.Body.push_back(0x5A); // i64.ge_u (yields i32)
      PushU32Const(FE.Body, 29);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      // V = ((a ^ ~b) & (a ^ res)) >>> 31
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      PushU32Const(FE.Body, 0xFFFFFFFF);
      FE.Body.push_back(0x73); // i32.xor
      FE.Body.push_back(0x73); // i32.xor
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, Dest);
      FE.Body.push_back(0x73); // i32.xor
      FE.Body.push_back(0x71); // i32.and
      PushU32Const(FE.Body, 31);
      FE.Body.push_back(0x76); // i32.shr_u
      PushU32Const(FE.Body, 28);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      PushLocalSet(FE.Body, FE.NZCV());
      break;
    }
    case IR::IROps::OP_SUBWITHFLAGS: {
      // res = a - b. Stored C follows the Arm `subs` contract (C = NOT
      // borrow), which the block-exit RectifyCarryInvert XORs into the
      // canonical inverted form; V from the signed-difference formula.
      uint32_t A, B;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A) || !ResolveArg(IR, IROp->Args[1], FE, &B)) {
        return false;
      }
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(0x6B); // i32.sub
      PushLocalSet(FE.Body, Dest);
      // N
      PushLocalGet(FE.Body, Dest);
      PushU32Const(FE.Body, 0x80000000);
      FE.Body.push_back(0x71); // i32.and
      // Z
      PushLocalGet(FE.Body, Dest);
      FE.Body.push_back(0x45); // i32.eqz
      PushU32Const(FE.Body, 30);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      // C = NOT borrow = (a u>= b)
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(0x4F); // i32.ge_u
      PushU32Const(FE.Body, 29);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      // V = ((a ^ b) & (a ^ res)) >>> 31
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(0x73); // i32.xor
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, Dest);
      FE.Body.push_back(0x73); // i32.xor
      FE.Body.push_back(0x71); // i32.and
      PushU32Const(FE.Body, 31);
      FE.Body.push_back(0x76); // i32.shr_u
      PushU32Const(FE.Body, 28);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      PushLocalSet(FE.Body, FE.NZCV());
      break;
    }
    case IR::IROps::OP_SUBNZCV:
    case IR::IROps::OP_ANDWITHFLAGS: {
      // Flags-only SubNZCV / AndWithFlags (no result node). AndWithFlags is
      // `test`/`and`: logical flags — N/Z from the AND, C = V = 0 (the Arm
      // `ands`/`tst` contract). SubNZCV (used by SetNZ_ZeroCV's x-0 trick and
      // CalculateFlags_SUB's sub-32 path) behaves like SubWithFlags flags-wise.
      // The GPR sources arrive full-width; the op's Size selects the compared
      // operand (the "truncates per SubNZCV" contract) — an unmasked 32-bit
      // compare made `cmp ax, imm8` never set Z (the P3.8c lstrcpyW wall).
      uint32_t A, B;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A) || !ResolveArg(IR, IROp->Args[1], FE, &B)) {
        return false;
      }
      const bool IsLogical = IROp->Op == IR::IROps::OP_ANDWITHFLAGS;
      const uint32_t Bits = IR::OpSizeAsBits(IROp->Size);
      if (Bits == 0 || Bits > 64) {
        LogMan::Msg::EFmt("WasmBackend: SubNZCV/AndWithFlags unsupported size {}", Bits);
        return false;
      }
      // 8/16-bit compares run on the size-masked operands (the "truncates per
      // SubNZCV" contract); 32-bit keeps the historical unmasked form and the
      // 64-bit flag paths (the MUL high-half derivation) keep the resolved
      // locals' low-half compare — the pre-P3.8c shape, oracle-pinned by the
      // mul corpus.
      const bool Masked = Bits == 8 || Bits == 16;
      const uint32_t Mask = (1u << Bits) - 1;
      auto PushMasked = [&](uint32_t Local) {
        PushLocalGet(FE.Body, Local);
        if (Masked) {
          PushU32Const(FE.Body, Mask);
          FE.Body.push_back(0x71); // i32.and
        }
      };
      uint32_t ResLocal = Dest;
      if (IsLogical) {
        PushLocalGet(FE.Body, A);
        PushLocalGet(FE.Body, B);
        FE.Body.push_back(0x71); // i32.and
        if (Masked) {
          PushU32Const(FE.Body, Mask);
          FE.Body.push_back(0x71); // i32.and
        }
        PushLocalSet(FE.Body, Dest);
      } else {
        ResLocal = FE.AllocScratch();
        PushLocalGet(FE.Body, A);
        PushLocalGet(FE.Body, B);
        FE.Body.push_back(0x6B); // i32.sub
        if (Masked) {
          PushU32Const(FE.Body, Mask);
          FE.Body.push_back(0x71); // i32.and
        }
        PushLocalSet(FE.Body, ResLocal);
      }
      // N = sign bit of the masked result, shifted into bit 31; the 32-bit
      // form is the unmasked degenerate of the same shape.
      PushLocalGet(FE.Body, ResLocal);
      if (Masked) {
        PushU32Const(FE.Body, 1u << (Bits - 1));
        FE.Body.push_back(0x71); // i32.and
        PushU32Const(FE.Body, 32 - Bits);
        FE.Body.push_back(0x74); // i32.shl
      } else {
        PushU32Const(FE.Body, 0x80000000);
        FE.Body.push_back(0x71); // i32.and
      }
      PushLocalGet(FE.Body, ResLocal);
      FE.Body.push_back(0x45); // i32.eqz
      PushU32Const(FE.Body, 30);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      if (!IsLogical) {
        // C = NOT borrow = (a u>= b) on the masked operands (Arm `subs`)
        PushMasked(A);
        PushMasked(B);
        FE.Body.push_back(0x4F); // i32.ge_u
        PushU32Const(FE.Body, 29);
        FE.Body.push_back(0x74); // i32.shl
        FE.Body.push_back(0x72); // i32.or
        // V = ((a ^ b) & (a ^ res)) >>> (Bits-1 for the masked sizes, else 31),
        // shifted into bit 28
        PushMasked(A);
        PushMasked(B);
        FE.Body.push_back(0x73); // i32.xor
        PushMasked(A);
        PushLocalGet(FE.Body, ResLocal);
        FE.Body.push_back(0x73); // i32.xor
        FE.Body.push_back(0x71); // i32.and
        PushU32Const(FE.Body, Masked ? Bits - 1 : 31);
        FE.Body.push_back(0x76); // i32.shr_u
        PushU32Const(FE.Body, 28);
        FE.Body.push_back(0x74); // i32.shl
        FE.Body.push_back(0x72); // i32.or
      }
      PushLocalSet(FE.Body, FE.NZCV());
      break;
    }
    case IR::IROps::OP_TESTNZ: {
      // 8/16-bit flag materialization (SetNZ_ZeroCV's small path): NZ from
      // (Src1 & Src2) masked to the op size, C = V = 0.
      const auto* Op = IROp->C<IR::IROp_TestNZ>();
      if (!EmitTestFlags(IR, Op->Src1, Op->Src2, Op->Header.Size, FE, Dest)) {
        return false;
      }
      break;
    }
    case IR::IROps::OP_TESTZ: {
      // Same flag shape as TestNZ (Src1 & Src2 masked); separate type.
      const auto* Op = IROp->C<IR::IROp_TestZ>();
      if (!EmitTestFlags(IR, Op->Src1, Op->Src2, Op->Header.Size, FE, Dest)) {
        return false;
      }
      break;
    }
    case IR::IROps::OP_ADCWITHFLAGS: {
      // res = a + b + C-in. The frontend rectifies the stored C to the
      // x86-true state (CFInverted=false) before this op, so C-in is bit 29
      // read directly; the carry-out is stored non-inverted.
      uint32_t A, B;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A) || !ResolveArg(IR, IROp->Args[1], FE, &B)) {
        return false;
      }
      // res = a + b + cf
      PushLocalGet(FE.Body, FE.NZCV());
      PushU32Const(FE.Body, 29);
      FE.Body.push_back(0x76); // i32.shr_u
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x71); // i32.and
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(0x6A); // i32.add
      FE.Body.push_back(0x6A); // i32.add
      PushLocalSet(FE.Body, Dest);
      // N
      PushLocalGet(FE.Body, Dest);
      PushU32Const(FE.Body, 0x80000000);
      FE.Body.push_back(0x71); // i32.and
      // Z
      PushLocalGet(FE.Body, Dest);
      FE.Body.push_back(0x45); // i32.eqz
      PushU32Const(FE.Body, 30);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      // C = carry-out of a + b + cf (i64 sum >= 2^32)
      PushLocalGet(FE.Body, A);
      FE.Body.push_back(0xAD); // i64.extend_i32_u
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(0xAD); // i64.extend_i32_u
      FE.Body.push_back(0x7C); // i64.add
      PushLocalGet(FE.Body, FE.NZCV());
      PushU32Const(FE.Body, 29);
      FE.Body.push_back(0x76); // i32.shr_u
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x71); // i32.and
      FE.Body.push_back(0xAD); // i64.extend_i32_u
      FE.Body.push_back(0x7C); // i64.add
      PushU64Const(FE.Body, 0x100000000ULL);
      FE.Body.push_back(0x5A); // i64.ge_u
      PushU32Const(FE.Body, 29);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      // V = ((a ^ res) & (b ^ res)) >>> 31 (three-way signed overflow)
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, Dest);
      FE.Body.push_back(0x73); // i32.xor
      PushLocalGet(FE.Body, B);
      PushLocalGet(FE.Body, Dest);
      FE.Body.push_back(0x73); // i32.xor
      FE.Body.push_back(0x71); // i32.and
      PushU32Const(FE.Body, 31);
      FE.Body.push_back(0x76); // i32.shr_u
      PushU32Const(FE.Body, 28);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      PushLocalSet(FE.Body, FE.NZCV());
      break;
    }
    case IR::IROps::OP_SBBWITHFLAGS: {
      // res = a - b - C-in. The frontend keeps the stored C *inverted* across
      // sbb (CFInverted=true), so the x86 borrow-in is ~bit29; the borrow-out
      // is stored to bit 29 (inverted convention).
      uint32_t A, B;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A) || !ResolveArg(IR, IROp->Args[1], FE, &B)) {
        return false;
      }
      // res = (a - b) - (~storedC & 1)
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(0x6B); // i32.sub
      PushLocalGet(FE.Body, FE.NZCV());
      PushU32Const(FE.Body, 29);
      FE.Body.push_back(0x76); // i32.shr_u
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x71); // i32.and
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x73); // i32.xor
      FE.Body.push_back(0x6B); // i32.sub
      PushLocalSet(FE.Body, Dest);
      // N
      PushLocalGet(FE.Body, Dest);
      PushU32Const(FE.Body, 0x80000000);
      FE.Body.push_back(0x71); // i32.and
      // Z
      PushLocalGet(FE.Body, Dest);
      FE.Body.push_back(0x45); // i32.eqz
      PushU32Const(FE.Body, 30);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      // C = NOT borrow = a u>= (b + cf)
      PushLocalGet(FE.Body, A);
      FE.Body.push_back(0xAD); // i64.extend_i32_u
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(0xAD); // i64.extend_i32_u
      PushLocalGet(FE.Body, FE.NZCV());
      PushU32Const(FE.Body, 29);
      FE.Body.push_back(0x76); // i32.shr_u
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x71); // i32.and
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x73); // i32.xor
      FE.Body.push_back(0xAD); // i64.extend_i32_u
      FE.Body.push_back(0x7C); // i64.add
      FE.Body.push_back(0x5A); // i64.ge_u
      PushU32Const(FE.Body, 29);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      // V = ((a ^ b) & (a ^ res)) >>> 31
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(0x73); // i32.xor
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, Dest);
      FE.Body.push_back(0x73); // i32.xor
      FE.Body.push_back(0x71); // i32.and
      PushU32Const(FE.Body, 31);
      FE.Body.push_back(0x76); // i32.shr_u
      PushU32Const(FE.Body, 28);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      PushLocalSet(FE.Body, FE.NZCV());
      break;
    }
    case IR::IROps::OP_NZCVSELECT:
    case IR::IROps::OP_NZCVSELECTINCREMENT: {
      // cond ? TrueVal : FalseVal (increment variant: FalseVal + 1 — the
      // Arm64 `csinc` contract used by IncrementByCarry).
      const IR::CondClass Cond = IROp->Op == IR::IROps::OP_NZCVSELECTINCREMENT
                                  ? IROp->C<IR::IROp_NZCVSelectIncrement>()->Cond
                                  : IROp->C<IR::IROp_NZCVSelect>()->Cond;
      const IR::OrderedNodeWrapper TrueVal = IROp->Op == IR::IROps::OP_NZCVSELECTINCREMENT
                                              ? IROp->C<IR::IROp_NZCVSelectIncrement>()->TrueVal
                                              : IROp->C<IR::IROp_NZCVSelect>()->TrueVal;
      const IR::OrderedNodeWrapper FalseVal = IROp->Op == IR::IROps::OP_NZCVSELECTINCREMENT
                                               ? IROp->C<IR::IROp_NZCVSelectIncrement>()->FalseVal
                                               : IROp->C<IR::IROp_NZCVSelect>()->FalseVal;
      uint32_t T, F;
      if (!ResolveArg(IR, TrueVal, FE, &T) || !ResolveArg(IR, FalseVal, FE, &F)) {
        return false;
      }
      // i32.select pops [val1, val2, cond]: push TrueVal, FalseVal, then the
      // condition computed below.
      PushLocalGet(FE.Body, T);
      PushLocalGet(FE.Body, F);
      if (IROp->Op == IR::IROps::OP_NZCVSELECTINCREMENT) {
        PushU32Const(FE.Body, 1);
        FE.Body.push_back(0x6A); // i32.add
      }
      bool ok = true;
      EmitNZCVCondition(FE.Body, FE.NZCV(), FE.CondScratch(), Cond, &ok);
      if (!ok) {
        return false;
      }
      FE.Body.push_back(0x1B); // i32.select
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_ADDSHIFT: {
      // Src1 + (Src2 << ShiftAmount) — the SIB/LEA address shape. Only LSL
      // is meaningful for address formation; anything else fails loudly.
      const auto* Op = IROp->C<IR::IROp_AddShift>();
      if (Op->Shift != IR::ShiftType::LSL) {
        LogMan::Msg::EFmt("WasmBackend: AddShift with unsupported shift type {}", static_cast<uint32_t>(Op->Shift));
        return false;
      }
      if (Op->ShiftAmount >= 32) {
        LogMan::Msg::EFmt("WasmBackend: AddShift with unsupported amount {}", Op->ShiftAmount);
        return false;
      }
      uint32_t A, B;
      if (!ResolveArg(IR, Op->Src1, FE, &A) || !ResolveArg(IR, Op->Src2, FE, &B)) {
        return false;
      }
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      if (Op->ShiftAmount != 0) {
        PushU32Const(FE.Body, Op->ShiftAmount);
        FE.Body.push_back(0x74); // i32.shl
      }
      FE.Body.push_back(0x6A); // i32.add
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_COPY: {
      // RA-inserted register move.
      uint32_t A;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A)) {
        return false;
      }
      PushLocalGet(FE.Body, A);
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_ADDNZCV: {
      // 8/16-bit add flags without a result (CalculateFlags_ADD's small
      // path): the byte/word sum's carry-out feeds stored C (non-inverted),
      // N/Z/V come from the masked result.
      const auto* Op = IROp->C<IR::IROp_AddNZCV>();
      uint32_t A, B;
      if (!ResolveArg(IR, Op->Src1, FE, &A) || !ResolveArg(IR, Op->Src2, FE, &B)) {
        return false;
      }
      const uint32_t Bits = IR::OpSizeAsBits(Op->Header.Size);
      if (Bits == 0 || Bits >= 32 || Bits == 32) {
        LogMan::Msg::EFmt("WasmBackend: AddNZCV with unsupported size {}", Bits);
        return false;
      }
      // s = (a & mask) + (b & mask) — fits without wrapping.
      PushLocalGet(FE.Body, A);
      PushU32Const(FE.Body, (1u << Bits) - 1);
      FE.Body.push_back(0x71); // i32.and
      PushLocalGet(FE.Body, B);
      PushU32Const(FE.Body, (1u << Bits) - 1);
      FE.Body.push_back(0x71); // i32.and
      FE.Body.push_back(0x6A); // i32.add
      PushLocalSet(FE.Body, Dest); // s in Dest (scratch use of the result local)
      // N = sign bit of the masked sum, moved to the NZCV's N position
      // (bit 31) — the same sub-32-bit shape defect the TestFlags lowering
      // had (the F3e wall).
      PushLocalGet(FE.Body, Dest);
      PushU32Const(FE.Body, 1u << (Bits - 1));
      FE.Body.push_back(0x71); // i32.and
      PushU32Const(FE.Body, 32 - Bits);
      FE.Body.push_back(0x74); // i32.shl
      // Z = (sum & mask) == 0
      PushLocalGet(FE.Body, Dest);
      PushU32Const(FE.Body, (1u << Bits) - 1);
      FE.Body.push_back(0x71); // i32.and
      FE.Body.push_back(0x45); // i32.eqz
      PushU32Const(FE.Body, 30);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      // C = carry-out = s >> Bits (non-inverted add convention)
      PushLocalGet(FE.Body, Dest);
      PushU32Const(FE.Body, Bits);
      FE.Body.push_back(0x76); // i32.shr_u
      PushU32Const(FE.Body, 29);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      // V = ((a ^ ~b) & (a ^ res8)) >>> (Bits-1) — recomputed on masked values
      PushLocalGet(FE.Body, A);
      PushU32Const(FE.Body, (1u << Bits) - 1);
      FE.Body.push_back(0x71); // i32.and
      PushLocalGet(FE.Body, B);
      PushU32Const(FE.Body, (1u << Bits) - 1);
      FE.Body.push_back(0x71); // i32.and
      PushU32Const(FE.Body, (1u << Bits) - 1);
      FE.Body.push_back(0x73); // i32.xor
      FE.Body.push_back(0x73); // i32.xor
      PushLocalGet(FE.Body, A);
      PushU32Const(FE.Body, (1u << Bits) - 1);
      FE.Body.push_back(0x71); // i32.and
      PushLocalGet(FE.Body, Dest);
      PushU32Const(FE.Body, (1u << Bits) - 1);
      FE.Body.push_back(0x71); // i32.and
      FE.Body.push_back(0x73); // i32.xor
      FE.Body.push_back(0x71); // i32.and
      PushU32Const(FE.Body, Bits - 1);
      FE.Body.push_back(0x76); // i32.shr_u
      PushU32Const(FE.Body, 28);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      PushLocalSet(FE.Body, FE.NZCV());
      break;
    }
    case IR::IROps::OP_LSHL:
    case IR::IROps::OP_LSHR:
    case IR::IROps::OP_ASHR:
    case IR::IROps::OP_ROR: {
      // Variable shifts/rotates. Wasm's i32 shifts/rotate mask the count to
      // 5 bits natively, matching x86's count masking for 32-bit ops.
      // The i64 source shape (the segment chain's 8-byte GDT-entry read
      // feeding `_Lshr(i64, ..., 32)`): the value stays i64 through the
      // shift and the default I32 dest wraps (the guest's address width);
      // the count widens to i64 for the op's type to check.
      uint32_t A, B;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A) || !ResolveArg(IR, IROp->Args[1], FE, &B)) {
        return false;
      }
      const bool SrcI64 = FE.LocalIsI64(A);
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      if (SrcI64) {
        FE.Body.push_back(0xAD); // i64.extend_i32_u (the count operand's type)
      }
      switch (IROp->Op) {
      case IR::IROps::OP_LSHL: FE.Body.push_back(SrcI64 ? 0x86 : 0x74); break; // shl
      case IR::IROps::OP_LSHR: FE.Body.push_back(SrcI64 ? 0x88 : 0x76); break; // shr_u
      case IR::IROps::OP_ASHR: FE.Body.push_back(SrcI64 ? 0x87 : 0x75); break; // shr_s
      default: FE.Body.push_back(SrcI64 ? 0x8B : 0x78); break; // rotr
      }
      if (SrcI64 && !FE.LocalIsI64(Dest)) {
        FE.Body.push_back(0xA7); // i32.wrap_i64
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_EXTR: {
      // The double-precision shift (shld/shrd immediate pair): ConcatValue =
      // $Upper:$Lower, Result = ConcatValue[LSB+Size-1 : LSB] (IR.json's
      // contract; the Arm64 JIT's `extr`). EXTR itself writes no flags — the
      // dispatcher's deferred-flag ops (CalculateFlags_ShiftRightDouble*
      // family) ride their own already-served lowerings, so there is no CF
      // story here to fake. The count masking is the dispatcher's
      // GetConstantShift (& 0x1F for 32-bit) — the LSB arriving here is
      // pre-masked and the i386 guest's shape is always the 32-bit one:
      // concat the two 32-bit halves into an i64 (the P2.12 movq primitive's
      // extend/wrap pair), shift right by the immediate, wrap to i32.
      const auto* Op = IROp->C<IR::IROp_Extr>();
      uint32_t Upper, Lower;
      if (!ResolveArg(IR, Op->Upper, FE, &Upper) || !ResolveArg(IR, Op->Lower, FE, &Lower)) {
        return false;
      }
      const uint32_t SizeBits = IR::OpSizeAsBits(IROp->Size);
      if (SizeBits != 32) {
        // Named loud-fail (the i64Bit-Orlshr precedent): the i386 guest
        // never emits a 64-bit EXTR (the 32-bit shld/shrd immediates lower
        // to _Extr(i32); the 64-bit ones only exist in a 64-bit guest).
        LogMan::Msg::EFmt("WasmBackend: EXTR with unsupported size {}", SizeBits);
        return false;
      }
      if (FE.LocalIsI64(Lower) || FE.LocalIsI64(Upper)) {
        LogMan::Msg::EFmt("WasmBackend: EXTR with I64-typed operands unsupported");
        return false;
      }
      PushLocalGet(FE.Body, Lower);
      FE.Body.push_back(0xAD); // i64.extend_i32_u
      PushLocalGet(FE.Body, Upper);
      FE.Body.push_back(0xAD); // i64.extend_i32_u
      PushU64Const(FE.Body, 32);
      FE.Body.push_back(0x86); // i64.shl
      FE.Body.push_back(0x84); // i64.or
      PushU64Const(FE.Body, Op->LSB);
      FE.Body.push_back(0x88); // i64.shr_u
      FE.Body.push_back(0xA7); // i32.wrap_i64
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_XORSHIFT: {
      // Src1 ^ shift(Src2) — the Arm64 `eor` with a shifted operand; all four
      // shift shapes are meaningful here.
      const auto* Op = IROp->C<IR::IROp_XorShift>();
      if (Op->ShiftAmount >= 32) {
        LogMan::Msg::EFmt("WasmBackend: XorShift with unsupported amount {}", Op->ShiftAmount);
        return false;
      }
      uint32_t A, B;
      if (!ResolveArg(IR, Op->Src1, FE, &A) || !ResolveArg(IR, Op->Src2, FE, &B)) {
        return false;
      }
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      if (Op->ShiftAmount != 0 || Op->Shift != IR::ShiftType::LSL) {
        PushU32Const(FE.Body, Op->ShiftAmount);
        switch (Op->Shift) {
        case IR::ShiftType::LSL: FE.Body.push_back(0x74); break; // i32.shl
        case IR::ShiftType::LSR: FE.Body.push_back(0x76); break; // i32.shr_u
        case IR::ShiftType::ASR: FE.Body.push_back(0x75); break; // i32.shr_s
        default: FE.Body.push_back(0x78); break; // i32.rotr
        }
      }
      FE.Body.push_back(0x73); // i32.xor
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_MUL: {
      // 32-bit product's low half (the three-operand imul result).
      uint32_t A, B;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A) || !ResolveArg(IR, IROp->Args[1], FE, &B)) {
        return false;
      }
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(0x6C); // i32.mul
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_UMUL:
    case IR::IROps::OP_SMULL: {
      // 64-bit product of 32-bit operands. The node's own i32 local holds the
      // low half; a scratch local holds the high half and is keyed to this
      // node's local so a following Bfe/Sbfe(32, lsb=32) can find it.
      const bool IsSigned = IROp->Op == IR::IROps::OP_SMULL;
      uint32_t A, B;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A) || !ResolveArg(IR, IROp->Args[1], FE, &B)) {
        return false;
      }
      const auto Extend = IsSigned ? 0xAC : 0xAD; // i64.extend_i32_s/u
      // high half: (a op b) >> 32, wrapped
      PushLocalGet(FE.Body, A);
      FE.Body.push_back(Extend);
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(Extend);
      FE.Body.push_back(0x7E); // i64.mul
      PushU64Const(FE.Body, 32);
      FE.Body.push_back(IsSigned ? 0x87 : 0x88); // i64.shr_s/u
      FE.Body.push_back(0xA7);                   // i32.wrap_i64
      const uint32_t Hi = FE.AllocScratch();
      PushLocalSet(FE.Body, Hi);
      // low half: wrap(a op b)
      PushLocalGet(FE.Body, A);
      FE.Body.push_back(Extend);
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(Extend);
      FE.Body.push_back(0x7E); // i64.mul
      FE.Body.push_back(0xA7); // i32.wrap_i64
      PushLocalSet(FE.Body, Dest);
      I64HiByLocal.emplace(Dest, Hi);
      break;
    }
    case IR::IROps::OP_DIV:
    case IR::IROps::OP_UDIV: {
      // 64/32 divide (edx:eax op divisor). i64 division traps on a zero
      // divisor (the x86 #DE), surfacing as a host-visible RuntimeError; a
      // quotient that does not fit 32 bits also raises #DE on x86, so it is
      // lowered to an explicit wasm `unreachable` trap.
      const bool IsSigned = IROp->Op == IR::IROps::OP_DIV;
      uint32_t Lower, Upper, Divisor;
      if (!ResolveArg(IR, IROp->Args[0], FE, &Lower) || !ResolveArg(IR, IROp->Args[2], FE, &Divisor)) {
        return false;
      }
      if (IROp->Args[1].IsInvalid()) {
        LogMan::Msg::EFmt("WasmBackend: Div without an upper half unsupported");
        return false;
      }
      if (!ResolveArg(IR, IROp->Args[1], FE, &Upper)) {
        return false;
      }
      const auto Div64 = IsSigned ? 0x7F : 0x80;  // i64.div_s/u
      const auto Rem64 = IsSigned ? 0x81 : 0x82;  // i64.rem_s/u
      const auto SignExt = IsSigned ? 0xAC : 0xAD; // i64.extend_i32_s/u
      // dividend = (Upper <ext> 32) | zeroext(Lower)
      auto EmitDividend = [&]() {
        PushLocalGet(FE.Body, Upper);
        FE.Body.push_back(SignExt);
        PushU64Const(FE.Body, 32);
        FE.Body.push_back(0x86); // i64.shl
        PushLocalGet(FE.Body, Lower);
        FE.Body.push_back(0xAD); // i64.extend_i32_u
        FE.Body.push_back(0x84); // i64.or
      };
      const uint32_t V64 = FE.AllocI64Scratch();
      const uint32_t Q64 = FE.AllocI64Scratch();
      const uint32_t R64 = FE.AllocI64Scratch();
      PushLocalGet(FE.Body, Divisor);
      FE.Body.push_back(0xAD); // i64.extend_i32_u
      PushLocalSet(FE.Body, V64);
      EmitDividend();
      PushLocalGet(FE.Body, V64);
      FE.Body.push_back(Div64);
      PushLocalSet(FE.Body, Q64);
      EmitDividend();
      PushLocalGet(FE.Body, V64);
      FE.Body.push_back(Rem64);
      PushLocalSet(FE.Body, R64);
      // #DE when the quotient does not fit 32 bits (high half non-zero)
      PushLocalGet(FE.Body, Q64);
      PushU64Const(FE.Body, 32);
      FE.Body.push_back(0x88); // i64.shr_u
      PushU64Const(FE.Body, 0);
      FE.Body.push_back(0x53); // i64.ne
      PushVec(FE.Body, {0x04, 0x40}); // if (the quotient's high half is set)
      FE.Body.push_back(0x00);         // unreachable — the x86 #DE
      FE.Body.push_back(0x0B);         // end if
      // route both results (post-RA immediates: the AllocateGPR homes)
      const std::pair<IR::OrderedNodeWrapper, uint32_t> DivResults[] = {
        {IROp->Args[3], Q64}, {IROp->Args[4], R64}
      };
      for (const auto& [Out, I64Local] : DivResults) {
        PushLocalGet(FE.Body, I64Local);
        FE.Body.push_back(0xA7); // i32.wrap_i64
        if (Out.IsImmediate()) {
          const IR::PhysicalRegister PhyReg(Out);
          if (PhyReg.AsRegClass() == IR::RegClass::GPRFixed && PhyReg.Reg < NUM_STATIC_SLOTS) {
            PushLocalSet(FE.Body, FE.StaticLocal(PhyReg.Reg));
          } else if (PhyReg.AsRegClass() == IR::RegClass::GPR && PhyReg.Reg < NUM_DYNAMIC_REGS) {
            const uint32_t Scratch = FE.AllocScratch();
            PushLocalSet(FE.Body, Scratch);
            DynamicLastDef[PhyReg.Reg] = Scratch + 1;
          } else {
            LogMan::Msg::EFmt("WasmBackend: Div result register {} unsupported", PhyReg.Reg);
            return false;
          }
        } else {
          PushLocalSet(FE.Body, FE.NodeLocal(IR->GetID(IR->GetNode(Out)).Value));
        }
      }
      break;
    }
    case IR::IROps::OP_CONDSUBNZCV: {
      // Arm `ccmp` semantics: if (Src1 <Cond> Src2) the NZCV becomes the
      // flags of (Src1 - Src2), else the FalseNZCV immediate (a 4-bit Arm
      // NZCV: bit3=N bit2=Z bit1=C bit0=V). Lowered branch-free: compute the
      // then-flags, the else-value, then i32.select on the condition.
      const auto* Op = IROp->C<IR::IROp_CondSubNZCV>();
      uint32_t S1, S2;
      if (!ResolveArg(IR, Op->Src1, FE, &S1) || !ResolveArg(IR, Op->Src2, FE, &S2)) {
        return false;
      }
      // then-leg value: flags of S1 - S2 (the Arm `subs` contract: stored C = NOT borrow)
      const uint32_t ResLocal = FE.AllocScratch();
      PushLocalGet(FE.Body, S1);
      PushLocalGet(FE.Body, S2);
      FE.Body.push_back(0x6B); // i32.sub
      PushLocalSet(FE.Body, ResLocal);
      PushLocalGet(FE.Body, ResLocal);
      PushU32Const(FE.Body, 0x80000000);
      FE.Body.push_back(0x71); // i32.and
      PushLocalGet(FE.Body, ResLocal);
      FE.Body.push_back(0x45); // i32.eqz
      PushU32Const(FE.Body, 30);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      PushLocalGet(FE.Body, S1);
      PushLocalGet(FE.Body, S2);
      FE.Body.push_back(0x4F); // i32.ge_u (NOT borrow)
      PushU32Const(FE.Body, 29);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      PushLocalGet(FE.Body, S1);
      PushLocalGet(FE.Body, S2);
      FE.Body.push_back(0x73); // i32.xor
      PushLocalGet(FE.Body, S1);
      PushLocalGet(FE.Body, ResLocal);
      FE.Body.push_back(0x73); // i32.xor
      FE.Body.push_back(0x71); // i32.and
      PushU32Const(FE.Body, 31);
      FE.Body.push_back(0x76); // i32.shr_u
      PushU32Const(FE.Body, 28);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      PushLocalSet(FE.Body, ResLocal); // stash the then-flags
      // else-leg value: the FalseNZCV immediate expanded to the stored layout
      const uint32_t ElseLocal = FE.AllocScratch();
      PushU32Const(FE.Body, ((Op->FalseNZCV & 0x8) ? 1u << 31 : 0) | ((Op->FalseNZCV & 0x4) ? 1u << 30 : 0)
                             | ((Op->FalseNZCV & 0x2) ? 1u << 29 : 0) | ((Op->FalseNZCV & 0x1) ? 1u << 28 : 0));
      PushLocalSet(FE.Body, ElseLocal);
      // i32.select pops [val1, val2, cond]
      PushLocalGet(FE.Body, ResLocal);
      PushLocalGet(FE.Body, ElseLocal);
      // The `ccmp` condition tests the CURRENT NZCV (the flags the preceding
      // SubNZCV left), not the operand values.
      bool ok = true;
      EmitNZCVCondition(FE.Body, FE.NZCV(), FE.CondScratch(), Op->Cond, &ok);
      if (!ok) {
        return false;
      }
      FE.Body.push_back(0x1B); // i32.select
      PushLocalSet(FE.Body, FE.NZCV());
      break;
    }
    case IR::IROps::OP_LOADNAMEDVECTORCONSTANT: {
      // The cvt paths materialize the x86 conversion boundary as a named
      // v128 constant; the CVTMAX family is the only one the scalar-float
      // lowering needs (docs/spikes/p2-x87-sse.md).
      const auto* Op = IROp->C<IR::IROp_LoadNamedVectorConstant>();
      uint64_t Lo = 0, Hi = 0;
      switch (Op->Constant) {
      case IR::NamedVectorConstant::NAMED_VECTOR_ZERO:
        // The pxor-zero idiom's constant (LoadZeroVector): all-zero lanes.
        Lo = Hi = 0;
        break;
      case IR::NamedVectorConstant::NAMED_VECTOR_CVTMAX_F32_I32:
      case IR::NamedVectorConstant::NAMED_VECTOR_CVTMAX_F32_I32_UPPER:
        Lo = Hi = 0x4F0000004F000000ULL; // 2147483648.0f splatted
        break;
      case IR::NamedVectorConstant::NAMED_VECTOR_CVTMAX_F32_I64:
        Lo = Hi = 0x5F0000005F000000ULL; // 2^63 as f32 splatted
        break;
      case IR::NamedVectorConstant::NAMED_VECTOR_CVTMAX_F64_I32:
      case IR::NamedVectorConstant::NAMED_VECTOR_CVTMAX_F64_I32_UPPER:
        Lo = Hi = 0x41E0000000000000ULL; // 2147483648.0 splatted
        break;
      case IR::NamedVectorConstant::NAMED_VECTOR_CVTMAX_F64_I64:
        Lo = Hi = 0x43E0000000000000ULL; // 2^63 splatted
        break;
      default:
        LogMan::Msg::EFmt("WasmBackend: LoadNamedVectorConstant {} unsupported", static_cast<uint32_t>(Op->Constant));
        return false;
      }
      UsesFPR = true;
      PushVec(FE.Body, {0xFD, 0x0C}); // v128.const
      for (uint32_t i = 0; i < 8; ++i) {
        FE.Body.push_back(static_cast<uint8_t>(Lo >> (8 * i)));
      }
      for (uint32_t i = 0; i < 8; ++i) {
        FE.Body.push_back(static_cast<uint8_t>(Hi >> (8 * i)));
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_VSTOFGPRINSERT: {
      // cvtsi2ss/sd: Vector with lane 0 = the converted GPR, upper preserved.
      const auto* Op = IROp->C<IR::IROp_VSToFGPRInsert>();
      uint32_t V, Src;
      if (!ResolveArg(IR, Op->Vector, FE, &V) || !ResolveArg(IR, Op->Src, FE, &Src)) {
        return false;
      }
      UsesFPR = true;
      const uint32_t SrcBits = IR::OpSizeAsBits(Op->SrcElementSize);
      const uint32_t ElemBits = IR::OpSizeAsBits(IROp->ElementSize);
      PushLocalGet(FE.Body, V);
      PushLocalGet(FE.Body, Src);
      if (SrcBits == 32 && ElemBits == 32) {
        FE.Body.push_back(0xB2); // f32.convert_i32_s
        PushVec(FE.Body, {0xFD, 0x20, 0x00}); // f32x4.replace_lane 0
      } else if (SrcBits == 32 && ElemBits == 64) {
        // The x86-32 cvtsi2sd shape (the notepad DPI path, 0x401ac8): a
        // 32-bit signed GPR widened into the f64 lane.
        FE.Body.push_back(0xB7); // f64.convert_i32_s
        PushVec(FE.Body, {0xFD, 0x22, 0x00}); // f64x2.replace_lane 0
      } else if (SrcBits == 64 && ElemBits == 64) {
        FE.Body.push_back(0xB9); // f64.convert_i64_s
        PushVec(FE.Body, {0xFD, 0x22, 0x00}); // f64x2.replace_lane 0
      } else {
        LogMan::Msg::EFmt("WasmBackend: VSToFGPRInsert {}->{} unsupported", SrcBits, ElemBits);
        return false;
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_SELECT: {
      // (Cmp1 <Cond> Cmp2) ? TrueVal : FalseVal — value comparison, not NZCV.
      // Float condition classes (the cvt paths' in-range selects) compare the
      // lane-0 scalars of two v128 operands.
      const auto* Op = IROp->C<IR::IROp_Select>();
      uint32_t C1, C2, T, F;
      if (!ResolveArg(IR, Op->Cmp1, FE, &C1) || !ResolveArg(IR, Op->Cmp2, FE, &C2) || !ResolveArg(IR, Op->TrueVal, FE, &T)
          || !ResolveArg(IR, Op->FalseVal, FE, &F)) {
        return false;
      }
      // i32.select pops [val1, val2, cond]: values first, condition last.
      PushLocalGet(FE.Body, T);
      PushLocalGet(FE.Body, F);
      switch (Op->Cond) {
      case IR::CondClass::FGT: {
        // Ordered greater (false on NaN) — wasm's f32/f64.gt.
        if (IR::OpSizeToSize(Op->CompareSize) == 8) {
          PushF64FromV128(FE, C1);
          PushF64FromV128(FE, C2);
          FE.Body.push_back(0x64); // f64.gt
        } else {
          PushF32FromV128(FE, C1);
          PushF32FromV128(FE, C2);
          FE.Body.push_back(0x5E); // f32.gt
        }
        break;
      }
      default:
        if (!EmitValueCondition(FE.Body, Op->Cond, C1, C2)) {
          return false;
        }
        break;
      }
      FE.Body.push_back(0x1B); // i32.select
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_LOADPF:
    case IR::IROps::OP_LOADAF: {
      // SRA pseudo-ops: the node lives in the PF/AF static slot.
      PushLocalGet(FE.Body, FE.StaticLocal(IROp->Op == IR::IROps::OP_LOADPF ? 16 : 17));
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_SHIFTFLAGS: {
      // Variable-shift flags (the Arm64 `ands` + CF/OF extraction contract):
      // count == 0 leaves flags alone and routes PFInput to the PF output;
      // otherwise N/Z come from the result, arm C = 0 then the last bit
      // shifted out is inserted, and V = sign change (except ASR).
      const auto* Op = IROp->C<IR::IROp_ShiftFlags>();
      uint32_t Result, Src1, Src2, PFInput;
      if (!ResolveArg(IR, Op->Result, FE, &Result) || !ResolveArg(IR, Op->Src1, FE, &Src1)
          || !ResolveArg(IR, Op->Src2, FE, &Src2) || !ResolveArg(IR, Op->PFInput, FE, &PFInput)) {
        return false;
      }
      const uint32_t PFLocal = FE.StaticLocal(16);
      const uint32_t Count = FE.AllocScratch();
      PushLocalGet(FE.Body, Src2);
      PushU32Const(FE.Body, 0x1F);
      FE.Body.push_back(0x71); // i32.and
      PushLocalSet(FE.Body, Count);
      PushLocalGet(FE.Body, Count);
      PushVec(FE.Body, {0x04, 0x40}); // if (count != 0)
      // N/Z from the result (arm ands: C = V = 0)
      PushLocalGet(FE.Body, Result);
      PushU32Const(FE.Body, 0x80000000);
      FE.Body.push_back(0x71); // i32.and
      PushLocalGet(FE.Body, Result);
      FE.Body.push_back(0x45); // i32.eqz
      PushU32Const(FE.Body, 30);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      // C = last bit shifted out
      if (Op->Shift == IR::ShiftType::LSL) {
        PushLocalGet(FE.Body, Src1);
        PushLocalGet(FE.Body, Count);
        PushU32Const(FE.Body, 32);
        FE.Body.push_back(0x6B); // i32.sub
        FE.Body.push_back(0x76); // i32.shr_u
      } else {
        PushLocalGet(FE.Body, Src1);
        PushLocalGet(FE.Body, Count);
        PushU32Const(FE.Body, 1);
        FE.Body.push_back(0x6B); // i32.sub
        FE.Body.push_back(0x76); // i32.shr_u
      }
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x71); // i32.and
      if (Op->InvertCF) {
        PushU32Const(FE.Body, 1);
        FE.Body.push_back(0x73); // i32.xor
      }
      PushU32Const(FE.Body, 29);
      FE.Body.push_back(0x74); // i32.shl
      FE.Body.push_back(0x72); // i32.or
      // V = sign change between Src1 and Result (defined for count == 1, set
      // unconditionally like the Arm lowering; not for ASR)
      if (Op->Shift != IR::ShiftType::ASR) {
        PushLocalGet(FE.Body, Src1);
        PushLocalGet(FE.Body, Result);
        FE.Body.push_back(0x73); // i32.xor
        PushU32Const(FE.Body, 31);
        FE.Body.push_back(0x76); // i32.shr_u
        PushU32Const(FE.Body, 28);
        FE.Body.push_back(0x74); // i32.shl
        FE.Body.push_back(0x72); // i32.or
      }
      PushLocalSet(FE.Body, FE.NZCV());
      PushLocalGet(FE.Body, Result);
      PushLocalSet(FE.Body, PFLocal);
      FE.Body.push_back(0x05); // else
      PushLocalGet(FE.Body, PFInput);
      PushLocalSet(FE.Body, PFLocal);
      FE.Body.push_back(0x0B); // end if
      break;
    }
    case IR::IROps::OP_LOADNZCV: {
      PushLocalGet(FE.Body, FE.NZCV());
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_STORENZCV: {
      uint32_t A;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A)) {
        return false;
      }
      PushLocalGet(FE.Body, A);
      PushLocalSet(FE.Body, FE.NZCV());
      break;
    }
    case IR::IROps::OP_STOREPF:
    case IR::IROps::OP_STOREAF: {
      uint32_t A;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A)) {
        return false;
      }
      PushLocalGet(FE.Body, A);
      PushLocalSet(FE.Body, FE.StaticLocal(IROp->Op == IR::IROps::OP_STOREPF ? 16 : 17));
      WroteStaticThisNode = true;
      break;
    }
    case IR::IROps::OP_STOREREGISTER: {
      const auto DestReg = IR::PhysicalRegister(CodeNode);
      if (DestReg.AsRegClass() != IR::RegClass::GPRFixed || DestReg.Reg >= NUM_STATIC_SLOTS) {
        LogMan::Msg::EFmt("WasmBackend: StoreRegister to unsupported class/reg");
        return false;
      }
      uint32_t A;
      if (!ResolveArg(IR, IROp->C<IR::IROp_StoreRegister>()->Value, FE, &A)) {
        return false;
      }
      PushLocalGet(FE.Body, A);
      PushLocalSet(FE.Body, FE.StaticLocal(DestReg.Reg));
      WroteStaticThisNode = true;
      break;
    }
    case IR::IROps::OP_ENTRYPOINTOFFSET: {
      // Non-inline variant of InlineEntrypointOffset: same value, but the
      // node gets an SSA local (e.g. the return address a `call` pushes).
      PushU32Const(FE.Body, static_cast<uint32_t>(EntryRIP + IROp->C<IR::IROp_EntrypointOffset>()->Offset));
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_PUSH: {
      // res = Addr - ValueSize; [res] = Value; result = res (pre-index
      // writeback, mirroring the Arm64 DEF_OP(Push) contract).
      const auto* Op = IROp->C<IR::IROp_Push>();
      const uint32_t ValueSize = IR::OpSizeToSize(Op->ValueSize);
      uint32_t Value, Addr;
      if (ValueSize == 0 || ValueSize > 4) {
        LogMan::Msg::EFmt("WasmBackend: Push with unsupported ValueSize {}", ValueSize);
        return false;
      }
      if (!ResolveArg(IR, Op->Value, FE, &Value) || !ResolveArg(IR, Op->Addr, FE, &Addr)) {
        return false;
      }
      PushLocalGet(FE.Body, Addr);
      PushU32Const(FE.Body, ValueSize);
      FE.Body.push_back(0x6B); // i32.sub
      PushLocalSet(FE.Body, Dest);
      PushLocalGet(FE.Body, Dest);
      PushLocalGet(FE.Body, Value);
      if (!PushGPRStore(FE.Body, ValueSize, 0)) {
        LogMan::Msg::EFmt("WasmBackend: Push with unsupported store size {}", ValueSize);
        return false;
      }
      break;
    }
    case IR::IROps::OP_POP: {
      // [Addr] -> OutValue; Addr += Size (post-index), mirroring the Arm64
      // DEF_OP(Pop) contract. The value has no SSA node: it goes to a
      // scratch local whose index becomes the OutValue register's last def.
      const auto* Op = IROp->C<IR::IROp_Pop>();
      const uint32_t Size = IR::OpSizeToSize(Op->Size);
      uint32_t Addr;
      if (Size == 0 || Size > 4) {
        LogMan::Msg::EFmt("WasmBackend: Pop with unsupported Size {}", Size);
        return false;
      }
      if (!ResolveArg(IR, Op->InoutAddr, FE, &Addr)) {
        return false;
      }
      PushLocalGet(FE.Body, Addr);
      if (!PushGPRLoad(FE.Body, Size, 0)) {
        LogMan::Msg::EFmt("WasmBackend: Pop with unsupported load size {}", Size);
        return false;
      }
      const uint32_t ValueLocal = FE.AllocScratch();
      PushLocalSet(FE.Body, ValueLocal);
      PushLocalGet(FE.Body, Addr);
      PushU32Const(FE.Body, Size);
      FE.Body.push_back(0x6A); // i32.add
      // Post-index write-back: the incremented address goes to the InoutAddr
      // arg's location (the Arm64 DEF_OP(Pop) contract) — the Pop node itself
      // may carry no RA annotation (e.g. a `ret`'s pop).
      uint32_t WriteBack;
      if (Op->InoutAddr.IsImmediate()) {
        const IR::PhysicalRegister AddrReg(Op->InoutAddr);
        if (AddrReg.AsRegClass() == IR::RegClass::GPRFixed && AddrReg.Reg < NUM_STATIC_SLOTS) {
          WriteBack = FE.StaticLocal(AddrReg.Reg);
        } else if (AddrReg.AsRegClass() == IR::RegClass::GPR && AddrReg.Reg < NUM_DYNAMIC_REGS && DynamicLastDef[AddrReg.Reg] != 0) {
          WriteBack = DynamicLastDef[AddrReg.Reg] - 1;
        } else {
          LogMan::Msg::EFmt("WasmBackend: Pop Addr register {} unsupported", AddrReg.Reg);
          return false;
        }
      } else {
        WriteBack = Dest;
      }
      PushLocalSet(FE.Body, WriteBack);
      if (Op->OutValue.IsImmediate()) {
        const IR::PhysicalRegister PhyReg(Op->OutValue);
        if (PhyReg.AsRegClass() == IR::RegClass::GPRFixed && PhyReg.Reg < NUM_STATIC_SLOTS) {
          PushLocalGet(FE.Body, ValueLocal);
          PushLocalSet(FE.Body, FE.StaticLocal(PhyReg.Reg));
        } else if (PhyReg.AsRegClass() == IR::RegClass::GPR && PhyReg.Reg < NUM_DYNAMIC_REGS) {
          DynamicLastDef[PhyReg.Reg] = ValueLocal + 1;
        } else {
          LogMan::Msg::EFmt("WasmBackend: Pop OutValue register {} unsupported", PhyReg.Reg);
          return false;
        }
      } else {
        PushLocalGet(FE.Body, ValueLocal);
        PushLocalSet(FE.Body, FE.NodeLocal(IR->GetID(IR->GetNode(Op->OutValue)).Value));
      }
      break;
    }
    case IR::IROps::OP_POPTWO: {
      // Post-RA fusion of two consecutive pops (Arm64: ldp post-index).
      // value1 = [Addr]; value2 = [Addr + Size]; Addr += 2*Size. Both values
      // have no SSA node: each goes to a scratch local routed like Pop's
      // OutValue. The Addr write-back goes to the InoutAddr arg's location
      // directly (the fused node has no RA annotation of its own — same v0
      // shape as PushTwo).
      const auto* Op = IROp->C<IR::IROp_PopTwo>();
      const uint32_t Size = IR::OpSizeToSize(Op->Size);
      if (Size == 0 || Size > 4) {
        LogMan::Msg::EFmt("WasmBackend: PopTwo with unsupported Size {}", Size);
        return false;
      }
      if (!Op->InoutAddr.IsImmediate()) {
        LogMan::Msg::EFmt("WasmBackend: PopTwo Addr must be a static register");
        return false;
      }
      const IR::PhysicalRegister AddrReg(Op->InoutAddr);
      if (AddrReg.AsRegClass() != IR::RegClass::GPRFixed || AddrReg.Reg >= NUM_STATIC_SLOTS) {
        LogMan::Msg::EFmt("WasmBackend: PopTwo Addr register {} unsupported", AddrReg.Reg);
        return false;
      }
      const uint32_t AddrLocal = FE.StaticLocal(AddrReg.Reg);
      PushLocalGet(FE.Body, AddrLocal);
      if (!PushGPRLoad(FE.Body, Size, 0)) {
        LogMan::Msg::EFmt("WasmBackend: PopTwo with unsupported load size {}", Size);
        return false;
      }
      const uint32_t Value1Local = FE.AllocScratch();
      PushLocalSet(FE.Body, Value1Local);
      PushLocalGet(FE.Body, AddrLocal);
      PushU32Const(FE.Body, Size);
      FE.Body.push_back(0x6A); // i32.add
      if (!PushGPRLoad(FE.Body, Size, 0)) {
        LogMan::Msg::EFmt("WasmBackend: PopTwo with unsupported load size {}", Size);
        return false;
      }
      const uint32_t Value2Local = FE.AllocScratch();
      PushLocalSet(FE.Body, Value2Local);
      PushLocalGet(FE.Body, AddrLocal);
      PushU32Const(FE.Body, 2 * Size);
      FE.Body.push_back(0x6A); // i32.add
      PushLocalSet(FE.Body, AddrLocal);
      const std::pair<IR::OrderedNodeWrapper, uint32_t> PopResults[] = {
        {Op->OutValue1, Value1Local}, {Op->OutValue2, Value2Local}
      };
      for (const auto& [OutValue, ValueLocal] : PopResults) {
        if (OutValue.IsImmediate()) {
          const IR::PhysicalRegister PhyReg(OutValue);
          if (PhyReg.AsRegClass() == IR::RegClass::GPRFixed && PhyReg.Reg < NUM_STATIC_SLOTS) {
            PushLocalGet(FE.Body, ValueLocal);
            PushLocalSet(FE.Body, FE.StaticLocal(PhyReg.Reg));
          } else if (PhyReg.AsRegClass() == IR::RegClass::GPR && PhyReg.Reg < NUM_DYNAMIC_REGS) {
            DynamicLastDef[PhyReg.Reg] = ValueLocal + 1;
          } else {
            LogMan::Msg::EFmt("WasmBackend: PopTwo OutValue register {} unsupported", PhyReg.Reg);
            return false;
          }
        } else {
          PushLocalGet(FE.Body, ValueLocal);
          PushLocalSet(FE.Body, FE.NodeLocal(IR->GetID(IR->GetNode(OutValue)).Value));
        }
      }
      break;
    }
    case IR::IROps::OP_PUSHSTACK: {
      // Push an FPR value onto the guest x87 stack (F64 mode: the value is a
      // 64-bit float in lane 0). mm[(top-1)&7] = Src; top = (top-1)&7.
      // An m80 LoadSize (the SoftFloat shape) is not implemented: loud-fail.
      const auto* Op = IROp->C<IR::IROp_PushStack>();
      if (IR::OpSizeToSize(Op->LoadSize) == 16) {
        LogMan::Msg::EFmt("WasmBackend: PushStack of an 80-bit value unsupported (no SoftFloat path)");
        return false;
      }
      uint32_t Src;
      if (!ResolveArg(IR, Op->X80Src, FE, &Src)) {
        return false;
      }
      UsesFPR = true;
      EmitStackSlotAddr(FE, ContextBase, 7); // (top - 1) & 7 == (top + 7) & 7
      PushLocalGet(FE.Body, Src);
      PushV128Store(FE.Body, 0);
      EmitSetX87Top(FE, ContextBase, 7);
      break;
    }
    case IR::IROps::OP_POPSTACKDESTROY: {
      // top = (top+1)&7; the popped value is dropped.
      EmitSetX87Top(FE, ContextBase, 1);
      break;
    }
    case IR::IROps::OP_COPYPUSHSTACK: {
      // mm[(top-1)&7] = mm[(top+Loc)&7]; top = (top-1)&7.
      // The store address goes first: v128.store pops [addr, value], so the
      // loaded value must sit on top of the stack when the store consumes it.
      const auto* Op = IROp->C<IR::IROp_CopyPushStack>();
      UsesFPR = true;
      EmitStackSlotAddr(FE, ContextBase, 7);
      EmitStackSlotAddr(FE, ContextBase, Op->StackLocation);
      PushV128Load(FE.Body, 0);
      PushV128Store(FE.Body, 0);
      EmitSetX87Top(FE, ContextBase, 7);
      break;
    }
    case IR::IROps::OP_READSTACKVALUE: {
      // FPR = mm[(top+Loc)&7] (the full 16-byte slot; the upper half is the
      // F80 extension area, unused in F64 mode).
      const auto* Op = IROp->C<IR::IROp_ReadStackValue>();
      UsesFPR = true;
      EmitStackSlotAddr(FE, ContextBase, Op->StackLocation);
      PushV128Load(FE.Body, 0);
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_STORESTACKMEM: {
      // [Addr] = convert(mm[top]); the pop is a separate PopStackDestroy.
      // SourceSize 64 (F64 mode) only; 128 would be the m80 SoftFloat path.
      const auto* Op = IROp->C<IR::IROp_StoreStackMem>();
      if (IR::OpSizeToSize(Op->SourceSize) != 8) {
        LogMan::Msg::EFmt("WasmBackend: StoreStackMem from an 80-bit stack slot unsupported");
        return false;
      }
      uint32_t MemArgOffset;
      if (!EmitEffectiveAddress(IR, Op->Addr, Op->Offset, Op->OffsetType, Op->OffsetScale, FE, &MemArgOffset)) {
        return false;
      }
      UsesFPR = true;
      EmitStackSlotAddr(FE, ContextBase, 0);
      PushF64Load(FE.Body, 0);
      const uint32_t StoreSize = IR::OpSizeToSize(Op->StoreSize);
      if (StoreSize == 8) {
        PushF64Store(FE.Body, MemArgOffset);
      } else if (StoreSize == 4) {
        FE.Body.push_back(0xB6); // f32.demote_f64
        PushF32Store(FE.Body, MemArgOffset);
      } else {
        LogMan::Msg::EFmt("WasmBackend: StoreStackMem with unsupported store size {}", StoreSize);
        return false;
      }
      break;
    }
    case IR::IROps::OP_INCSTACKTOP:
      EmitSetX87Top(FE, ContextBase, 1);
      break;
    case IR::IROps::OP_DECSTACKTOP:
      EmitSetX87Top(FE, ContextBase, 7);
      break;
    case IR::IROps::OP_INITSTACK:
      // top = 0. (fninit's FCW/FTW stores are explicit StoreContext ops.)
      PushU32Const(FE.Body, 0);
      PushU32Const(FE.Body, 0);
      PushI32Store8(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.flags) + X86State::X87FLAG_TOP_LOC);
      break;
    case IR::IROps::OP_INVALIDATESTACK: {
      // Clear the slot's valid bit in AbridgedFTW (bit (top+Loc)&7); 0xff
      // invalidates every slot (EMMS).
      const auto* Op = IROp->C<IR::IROp_InvalidateStack>();
      if (Op->StackLocation == 0xff) {
        PushU32Const(FE.Body, 0);
        PushU32Const(FE.Body, 0);
        PushI32Store8(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.AbridgedFTW));
        break;
      }
      PushU32Const(FE.Body, 0); // AbridgedFTW address
      PushU32Const(FE.Body, 0);
      PushI32Load8U(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.AbridgedFTW));
      PushU32Const(FE.Body, 0);
      PushI32Load8U(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.flags) + X86State::X87FLAG_TOP_LOC);
      PushU32Const(FE.Body, Op->StackLocation);
      FE.Body.push_back(0x6A); // i32.add
      PushU32Const(FE.Body, 7);
      FE.Body.push_back(0x71); // i32.and
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x74); // i32.shl -> 1 << idx
      PushU32Const(FE.Body, 0xFF);
      FE.Body.push_back(0x73); // i32.xor -> mask
      FE.Body.push_back(0x71); // i32.and
      PushI32Store8(FE.Body, 0);
      break;
    }
    case IR::IROps::OP_SYNCSTACKTOSLOW: {
      // The naive backend keeps the x87 stack in the context at all times,
      // so synchronizing is reading the current TOP. The result (the top) is
      // the node's value, as in the Arm64 slow path.
      PushU32Const(FE.Body, 0);
      PushI32Load8U(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.flags) + X86State::X87FLAG_TOP_LOC);
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_STACKFORCESLOW:
      // No virtual-stack tracking to force: the context is already the truth.
      break;
    case IR::IROps::OP_SETROUNDINGMODE:
      // Wasm has no dynamic rounding mode (always round-to-nearest); the
      // documented divergence is in docs/spikes/p2-x87-sse.md. The FCW value
      // itself is stored/read verbatim by the surrounding context ops.
      break;
    case IR::IROps::OP_LOADDF: {
      // ±1: sign-extended byte load of the raw DF (the Arm64 `ldrsb`
      // contract — flags[RFLAG_DF_RAW_LOC] is 0x1/0xFF).
      PushU32Const(FE.Body, 0);
      PushI32Load8U(FE.Body, ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State.flags) + X86State::RFLAG_DF_RAW_LOC);
      FE.Body.push_back(0xC0); // i32.extend8_s
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_ORNROR: {
      // Arm ORN with an immediate rotate: Dest = Src1 | ~(Src2 ROR BitShift).
      const auto* Op = IROp->C<IR::IROp_Ornror>();
      uint32_t S1, S2;
      if (!ResolveArg(IR, Op->Src1, FE, &S1) || !ResolveArg(IR, Op->Src2, FE, &S2)) {
        return false;
      }
      PushLocalGet(FE.Body, S1);
      PushLocalGet(FE.Body, S2);
      PushU32Const(FE.Body, Op->BitShift & 31);
      FE.Body.push_back(0x78); // i32.rotr
      PushU32Const(FE.Body, 0xFFFFFFFF);
      FE.Body.push_back(0x73); // i32.xor -> the inverted operand
      FE.Body.push_back(0x72); // i32.or
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_MEMCPY: {
      // rep movs: an element-wise loop with a runtime direction check.
      // memory.copy is the forward fast path and a P5 perf item; the loop is
      // the correctness baseline (overlap semantics differ per direction —
      // hardware walks the elements from the DF side first).
      const auto* Op = IROp->C<IR::IROp_MemCpy>();
      uint32_t Dest, Src, Length, Dir;
      if (!ResolveArg(IR, Op->Dest, FE, &Dest) || !ResolveArg(IR, Op->Src, FE, &Src) ||
          !ResolveArg(IR, Op->Length, FE, &Length) || !ResolveDirection(IR, Op->Direction, FE, &Dir)) {
        return false;
      }
      const uint32_t ElemBytes = IR::OpSizeToSize(Op->Size);
      if (ElemBytes != 1 && ElemBytes != 2 && ElemBytes != 4) {
        LogMan::Msg::EFmt("WasmBackend: MemCpy with unsupported element size {}", ElemBytes);
        return false;
      }
      const uint32_t Shift = __builtin_ctz(ElemBytes);
      const uint32_t Count = FE.AllocScratch();
      const uint32_t Len = FE.AllocScratch();
      const uint32_t Off = FE.AllocScratch();
      const uint32_t Total = FE.AllocScratch();
      PushLocalGet(FE.Body, Length);
      PushLocalSet(FE.Body, Count);
      PushLocalGet(FE.Body, Length);
      PushLocalSet(FE.Body, Len);
      PushVec(FE.Body, {0x02, 0x40});  // block (void)
      PushVec(FE.Body, {0x03, 0x40});  //   loop (void)
      PushLocalGet(FE.Body, Len);
      FE.Body.push_back(0x45);          //   i32.eqz
      PushVec(FE.Body, {0x0D, 0x01});   //   br_if 1 -> the block (done)
      // off = Dir < 0 ? (Len-1) << s : (Count-Len) << s — the element index
      // this iteration copies, walked from the DF side.
      PushLocalGet(FE.Body, Dir);
      PushU32Const(FE.Body, 0);
      FE.Body.push_back(0x48);          // i32.lt_s
      PushVec(FE.Body, {0x04, 0x7F});   // if (result i32)
      PushLocalGet(FE.Body, Len);
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x6B);          // i32.sub
      PushU32Const(FE.Body, Shift);
      FE.Body.push_back(0x74);          // i32.shl
      FE.Body.push_back(0x05);          // else
      PushLocalGet(FE.Body, Count);
      PushLocalGet(FE.Body, Len);
      FE.Body.push_back(0x6B);          // i32.sub
      PushU32Const(FE.Body, Shift);
      FE.Body.push_back(0x74);          // i32.shl
      FE.Body.push_back(0x0B);          // end if
      PushLocalSet(FE.Body, Off);
      // The element copy: [Dest+Off] <- [Src+Off]. The wasm store pops
      // [addr, value] with the value on top, so the destination address is
      // pushed first and the load's result rides on top of it. (The landed
      // shape pushed a stray i32.const 0 before each access, stored to
      // address 0, and left the stack unbalanced; the 8-bit memargs also
      // lacked their offset immediate — all three never exercised until the
      // F3d walk.)
      PushLocalGet(FE.Body, Dest);
      PushLocalGet(FE.Body, Off);
      FE.Body.push_back(0x6A);          // i32.add
      PushLocalGet(FE.Body, Src);
      PushLocalGet(FE.Body, Off);
      FE.Body.push_back(0x6A);          // i32.add
      switch (ElemBytes) {
      case 1: PushVec(FE.Body, {0x2D, 0x00, 0x00}); break; // i32.load8_u
      case 2: PushVec(FE.Body, {0x2F, 0x01, 0x00}); break; // i32.load16_u
      case 4: PushVec(FE.Body, {0x28, 0x02, 0x00}); break; // i32.load
      }
      switch (ElemBytes) {
      case 1: PushVec(FE.Body, {0x3A, 0x00, 0x00}); break; // i32.store8
      case 2: PushVec(FE.Body, {0x3B, 0x01, 0x00}); break; // i32.store16
      case 4: PushVec(FE.Body, {0x36, 0x02, 0x00}); break; // i32.store
      }
      PushLocalGet(FE.Body, Len);
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x6B);          // i32.sub
      PushLocalSet(FE.Body, Len);
      PushVec(FE.Body, {0x0C, 0x00});   //   br 0 -> loop
      FE.Body.push_back(0x0B);          // end loop
      FE.Body.push_back(0x0B);          // end block
      // Results: final addresses = base + Length*Dir.
      // -(Count << s) if Dir < 0 else (Count << s); i32.select pops
      // [val1, val2, cond] — condition last. The negation's 0 operand is
      // pushed BEFORE the shifted count: sub pops the rhs first, and the
      // landed order computed (Count<<s) - 0 — the forward result from both
      // select arms (the F3d str-rep-movsb-bwd finding).
      PushU32Const(FE.Body, 0);
      PushLocalGet(FE.Body, Count);
      PushU32Const(FE.Body, Shift);
      FE.Body.push_back(0x74);          // (Count << s)
      FE.Body.push_back(0x6B);          // 0 - (Count << s)
      PushLocalGet(FE.Body, Count);
      PushU32Const(FE.Body, Shift);
      FE.Body.push_back(0x74);          // (Count << s)
      PushLocalGet(FE.Body, Dir);
      PushU32Const(FE.Body, 0);
      FE.Body.push_back(0x48);          // i32.lt_s -> Dir < 0
      FE.Body.push_back(0x1B);          // i32.select
      PushLocalSet(FE.Body, Total);
      const std::pair<IR::OrderedNodeWrapper, uint32_t> Results[] = {
        {Op->OutDstAddress, Dest}, {Op->OutSrcAddress, Src}
      };
      for (const auto& [Out, Base] : Results) {
        PushLocalGet(FE.Body, Base);
        PushLocalGet(FE.Body, Total);
        FE.Body.push_back(0x6A);        // i32.add
        if (Out.IsImmediate()) {
          const IR::PhysicalRegister PhyReg(Out);
          if (PhyReg.AsRegClass() == IR::RegClass::GPRFixed && PhyReg.Reg < NUM_STATIC_SLOTS) {
            PushLocalSet(FE.Body, FE.StaticLocal(PhyReg.Reg));
          } else if (PhyReg.AsRegClass() == IR::RegClass::GPR && PhyReg.Reg < NUM_DYNAMIC_REGS) {
            const uint32_t Scratch = FE.AllocScratch();
            PushLocalSet(FE.Body, Scratch);
            DynamicLastDef[PhyReg.Reg] = Scratch + 1;
          } else {
            LogMan::Msg::EFmt("WasmBackend: MemCpy result register {} unsupported", PhyReg.Reg);
            return false;
          }
        } else {
          PushLocalSet(FE.Body, FE.NodeLocal(IR->GetID(IR->GetNode(Out)).Value));
        }
      }
      break;
    }
    case IR::IROps::OP_MEMSET: {
      // rep stos: the same direction-aware element loop storing Value.
      const auto* Op = IROp->C<IR::IROp_MemSet>();
      uint32_t Addr, Value, Length, Dir;
      if (!ResolveArg(IR, Op->Addr, FE, &Addr) || !ResolveArg(IR, Op->Value, FE, &Value) ||
          !ResolveArg(IR, Op->Length, FE, &Length) || !ResolveDirection(IR, Op->Direction, FE, &Dir)) {
        return false;
      }
      // The ELEMENT size is the op's $Size arg (STOSOp's OpSizeFromSrc);
      // the IROp header's Size is the DestSize (the returned final
      // address, i64Bit) — reading it here made every real invocation
      // loud-fail (the F3d finding: "MemSet with unsupported element
      // size 8" on a plain rep stosd).
      const uint32_t ElemBytes = IR::OpSizeToSize(Op->Size);
      if (ElemBytes != 1 && ElemBytes != 2 && ElemBytes != 4) {
        LogMan::Msg::EFmt("WasmBackend: MemSet with unsupported element size {}", ElemBytes);
        return false;
      }
      const uint32_t Shift = __builtin_ctz(ElemBytes);
      const uint32_t Count = FE.AllocScratch();
      const uint32_t Len = FE.AllocScratch();
      const uint32_t Off = FE.AllocScratch();
      const uint32_t Elem = FE.AllocScratch();
      const uint32_t Total = FE.AllocScratch();
      PushLocalGet(FE.Body, Length);
      PushLocalSet(FE.Body, Count);
      PushLocalGet(FE.Body, Length);
      PushLocalSet(FE.Body, Len);
      PushLocalGet(FE.Body, Value);
      switch (ElemBytes) {
      case 1: PushU32Const(FE.Body, 0xFF); FE.Body.push_back(0x71); break; // i32.and
      case 2: PushU32Const(FE.Body, 0xFFFF); FE.Body.push_back(0x71); break;
      case 4: break;
      }
      PushLocalSet(FE.Body, Elem);
      PushVec(FE.Body, {0x02, 0x40});  // block (void)
      PushVec(FE.Body, {0x03, 0x40});  //   loop (void)
      PushLocalGet(FE.Body, Len);
      FE.Body.push_back(0x45);          //   i32.eqz
      PushVec(FE.Body, {0x0D, 0x01});   //   br_if 1 -> the block (done)
      PushLocalGet(FE.Body, Dir);
      PushU32Const(FE.Body, 0);
      FE.Body.push_back(0x48);          // i32.lt_s
      PushVec(FE.Body, {0x04, 0x7F});   // if (result i32)
      PushLocalGet(FE.Body, Len);
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x6B);
      PushU32Const(FE.Body, Shift);
      FE.Body.push_back(0x74);
      FE.Body.push_back(0x05);
      PushLocalGet(FE.Body, Count);
      PushLocalGet(FE.Body, Len);
      FE.Body.push_back(0x6B);
      PushU32Const(FE.Body, Shift);
      FE.Body.push_back(0x74);
      FE.Body.push_back(0x0B);
      PushLocalSet(FE.Body, Off);
      PushLocalGet(FE.Body, Addr);
      PushLocalGet(FE.Body, Off);
      FE.Body.push_back(0x6A);          // i32.add
      PushLocalGet(FE.Body, Elem);
      switch (ElemBytes) {
      case 1: PushVec(FE.Body, {0x3A, 0x00, 0x00}); break; // i32.store8
      case 2: PushVec(FE.Body, {0x3B, 0x01, 0x00}); break; // i32.store16
      case 4: PushVec(FE.Body, {0x36, 0x02, 0x00}); break; // i32.store
      }
      PushLocalGet(FE.Body, Len);
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x6B);
      PushLocalSet(FE.Body, Len);
      PushVec(FE.Body, {0x0C, 0x00});   //   br 0 -> loop
      FE.Body.push_back(0x0B);          // end loop
      FE.Body.push_back(0x0B);          // end block
      // Result: final address = Addr + Length*Dir (routed like MemCpy's;
      // the negation's 0 goes first — sub pops the rhs first).
      PushU32Const(FE.Body, 0);
      PushLocalGet(FE.Body, Count);
      PushU32Const(FE.Body, Shift);
      FE.Body.push_back(0x74);
      FE.Body.push_back(0x6B);          // 0 - (Count << s)
      PushLocalGet(FE.Body, Count);
      PushU32Const(FE.Body, Shift);
      FE.Body.push_back(0x74);
      PushLocalGet(FE.Body, Dir);
      PushU32Const(FE.Body, 0);
      FE.Body.push_back(0x48);          // i32.lt_s
      FE.Body.push_back(0x1B);          // i32.select
      PushLocalSet(FE.Body, Total);
      PushLocalGet(FE.Body, Addr);
      PushLocalGet(FE.Body, Total);
      FE.Body.push_back(0x6A);          // i32.add
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_LOADCONTEXT: {
      // GPR-class context read (x87 TOP, FCW, the flag bytes). The result
      // node is I32-typed for sizes the guest addresses bytewise.
      const auto* Op = IROp->C<IR::IROp_LoadContext>();
      if (Op->Class != IR::RegClass::GPR) {
        LogMan::Msg::EFmt("WasmBackend: LoadContext with unsupported register class {}", static_cast<uint32_t>(Op->Class));
        return false;
      }
      const uint32_t Size = IR::OpSizeToSize(IROp->Size);
      const uint32_t AbsOffset = ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State) + Op->Offset;
      PushU32Const(FE.Body, 0);
      if (Size == 1) {
        PushI32Load8U(FE.Body, AbsOffset);
      } else if (Size == 2) {
        PushI32Load16U(FE.Body, AbsOffset);
      } else if (Size == 4) {
        PushI32Load(FE.Body, AbsOffset);
      } else {
        LogMan::Msg::EFmt("WasmBackend: LoadContext with unsupported size {}", Size);
        return false;
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_STORECONTEXT: {
      const auto* Op = IROp->C<IR::IROp_StoreContext>();
      if (Op->Class != IR::RegClass::GPR) {
        LogMan::Msg::EFmt("WasmBackend: StoreContext with unsupported register class {}", static_cast<uint32_t>(Op->Class));
        return false;
      }
      uint32_t Value;
      if (!ResolveArg(IR, Op->Value, FE, &Value)) {
        return false;
      }
      const uint32_t Size = IR::OpSizeToSize(IROp->Size);
      const uint32_t AbsOffset = ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State) + Op->Offset;
      PushU32Const(FE.Body, 0);
      PushLocalGet(FE.Body, Value);
      if (Size == 1) {
        PushI32Store8(FE.Body, AbsOffset);
      } else if (Size == 2) {
        PushI32Store16(FE.Body, AbsOffset);
      } else if (Size == 4) {
        PushI32Store(FE.Body, AbsOffset);
      } else {
        LogMan::Msg::EFmt("WasmBackend: StoreContext with unsupported size {}", Size);
        return false;
      }
      break;
    }
    case IR::IROps::OP_LOADCONTEXTINDEXED: {
      // N6: the segment-register write chain's indexed context read
      // (UpdatePrefixFromSegment: segment_arrays[Index] — the GDT/LDT array
      // pointer, indexed by the selector's TI bit). The address is dynamic
      // (AbsBase + Index * Stride), so the address operand carries the sum
      // and the mem-arg offset stays 0. The 64-bit context pointer
      // truncates to the 32-bit linear-memory offset (guest VA == wasm
      // offset, doc 01 §5); the private GDT lives inside the CPUState in
      // linear memory, so the truncation is exact.
      const auto* Op = IROp->C<IR::IROp_LoadContextIndexed>();
      if (Op->Class != IR::RegClass::GPR) {
        LogMan::Msg::EFmt("WasmBackend: LoadContextIndexed with unsupported register class {}", static_cast<uint32_t>(Op->Class));
        return false;
      }
      uint32_t Index;
      if (!ResolveArg(IR, Op->Index, FE, &Index)) {
        return false;
      }
      const uint32_t Size = IR::OpSizeToSize(IROp->Size);
      const uint32_t AbsBase = ContextBase + offsetof(FEXCore::Core::CpuStateFrame, State) + Op->BaseOffset;
      PushU32Const(FE.Body, AbsBase);
      PushLocalGet(FE.Body, Index);
      PushU32Const(FE.Body, Op->Stride);
      FE.Body.push_back(0x6C); // i32.mul
      FE.Body.push_back(0x6A); // i32.add
      if (Size == 8) {
        PushI64Load(FE.Body, 0);
        if (!FE.LocalIsI64(Dest)) {
          FE.Body.push_back(0xA7); // i32.wrap_i64 (the 32-bit model's address width)
        }
      } else if (Size == 4) {
        PushI32Load(FE.Body, 0);
      } else if (Size == 2) {
        PushI32Load16U(FE.Body, 0);
      } else if (Size == 1) {
        PushI32Load8U(FE.Body, 0);
      } else {
        LogMan::Msg::EFmt("WasmBackend: LoadContextIndexed with unsupported size {}", Size);
        return false;
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_F80ADDSTACK:
    case IR::IROps::OP_F80MULSTACK: {
      // mm[(top+SrcStack1)&7] = mm[(top+SrcStack1)&7] <op> mm[(top+S2)&7]
      // (the two-operand form's result lands in SrcStack1).
      const uint8_t S1 = IROp->Op == IR::IROps::OP_F80ADDSTACK ? IROp->C<IR::IROp_F80AddStack>()->SrcStack1
                                                               : IROp->C<IR::IROp_F80MulStack>()->SrcStack1;
      const uint8_t S2 = IROp->Op == IR::IROps::OP_F80ADDSTACK ? IROp->C<IR::IROp_F80AddStack>()->SrcStack2
                                                               : IROp->C<IR::IROp_F80MulStack>()->SrcStack2;
      UsesFPR = true;
      const uint32_t ABits = FE.AllocI64Scratch();
      const uint32_t BBits = FE.AllocI64Scratch();
      EmitStackSlotAddr(FE, ContextBase, S1); // store address first (f64.store pops [addr, value])
      EmitStackSlotAddr(FE, ContextBase, S1);
      PushF64Load(FE.Body, 0);
      FE.Body.push_back(0xBD); // i64.reinterpret_f64
      PushLocalSet(FE.Body, ABits);
      EmitStackSlotAddr(FE, ContextBase, S2);
      PushF64Load(FE.Body, 0);
      FE.Body.push_back(0xBD);
      PushLocalSet(FE.Body, BBits);
      EmitX87InvalidFlagCheck(FE, ContextBase, ABits, BBits,
        IROp->Op == IR::IROps::OP_F80ADDSTACK ? X87InvalidShape::Add : X87InvalidShape::Mul);
      EmitStackSlotAddr(FE, ContextBase, S1);
      PushF64Load(FE.Body, 0);
      EmitStackSlotAddr(FE, ContextBase, S2);
      PushF64Load(FE.Body, 0);
      FE.Body.push_back(FloatBinOp(IROp->Op, 64));
      PushF64Store(FE.Body, 0);
      break;
    }
    case IR::IROps::OP_F80SUBSTACK:
    case IR::IROps::OP_F80DIVSTACK: {
      // mm[(top+DstStack)&7] = mm[(top+SrcStack1)&7] <op> mm[(top+S2)&7].
      const uint8_t Dst = IROp->Op == IR::IROps::OP_F80SUBSTACK ? IROp->C<IR::IROp_F80SubStack>()->DstStack
                                                                : IROp->C<IR::IROp_F80DivStack>()->DstStack;
      const uint8_t S1 = IROp->Op == IR::IROps::OP_F80SUBSTACK ? IROp->C<IR::IROp_F80SubStack>()->SrcStack1
                                                               : IROp->C<IR::IROp_F80DivStack>()->SrcStack1;
      const uint8_t S2 = IROp->Op == IR::IROps::OP_F80SUBSTACK ? IROp->C<IR::IROp_F80SubStack>()->SrcStack2
                                                               : IROp->C<IR::IROp_F80DivStack>()->SrcStack2;
      UsesFPR = true;
      const uint32_t ABits = FE.AllocI64Scratch();
      const uint32_t BBits = FE.AllocI64Scratch();
      EmitStackSlotAddr(FE, ContextBase, Dst); // store address first
      EmitStackSlotAddr(FE, ContextBase, S1);
      PushF64Load(FE.Body, 0);
      FE.Body.push_back(0xBD); // i64.reinterpret_f64
      PushLocalSet(FE.Body, ABits);
      EmitStackSlotAddr(FE, ContextBase, S2);
      PushF64Load(FE.Body, 0);
      FE.Body.push_back(0xBD);
      PushLocalSet(FE.Body, BBits);
      EmitX87InvalidFlagCheck(FE, ContextBase, ABits, BBits,
        IROp->Op == IR::IROps::OP_F80SUBSTACK ? X87InvalidShape::Sub : X87InvalidShape::Div);
      EmitStackSlotAddr(FE, ContextBase, S1);
      PushF64Load(FE.Body, 0);
      EmitStackSlotAddr(FE, ContextBase, S2);
      PushF64Load(FE.Body, 0);
      FE.Body.push_back(FloatBinOp(IROp->Op, 64));
      PushF64Store(FE.Body, 0);
      break;
    }
    case IR::IROps::OP_F80ADDVALUE:
    case IR::IROps::OP_F80SUBVALUE:
    case IR::IROps::OP_F80MULVALUE:
    case IR::IROps::OP_F80DIVVALUE: {
      // mm[(top+SrcStack)&7] <op>= Src.
      const auto* Op = IROp->C<IR::IROp_F80AddValue>();
      uint32_t Src;
      if (!ResolveArg(IR, Op->X80Src, FE, &Src)) {
        return false;
      }
      UsesFPR = true;
      const uint32_t ABits = FE.AllocI64Scratch();
      const uint32_t BBits = FE.AllocI64Scratch();
      EmitStackSlotAddr(FE, ContextBase, Op->SrcStack); // store address first
      EmitStackSlotAddr(FE, ContextBase, Op->SrcStack);
      PushF64Load(FE.Body, 0);
      FE.Body.push_back(0xBD); // i64.reinterpret_f64
      PushLocalSet(FE.Body, ABits);
      PushF64FromV128(FE, Src);
      FE.Body.push_back(0xBD);
      PushLocalSet(FE.Body, BBits);
      const auto Shape = IROp->Op == IR::IROps::OP_F80ADDVALUE ? X87InvalidShape::Add
                       : IROp->Op == IR::IROps::OP_F80SUBVALUE ? X87InvalidShape::Sub
                       : IROp->Op == IR::IROps::OP_F80MULVALUE ? X87InvalidShape::Mul
                                                               : X87InvalidShape::Div;
      EmitX87InvalidFlagCheck(FE, ContextBase, ABits, BBits, Shape);
      EmitStackSlotAddr(FE, ContextBase, Op->SrcStack);
      PushF64Load(FE.Body, 0);
      PushF64FromV128(FE, Src);
      FE.Body.push_back(FloatBinOp(IROp->Op, 64));
      PushF64Store(FE.Body, 0);
      break;
    }
    case IR::IROps::OP_F80SUBRVALUE:
    case IR::IROps::OP_F80DIVRVALUE: {
      // mm[(top+SrcStack)&7] = Src <op> mm[(top+SrcStack)&7].
      const auto* Op = IROp->C<IR::IROp_F80SubRValue>();
      uint32_t Src;
      if (!ResolveArg(IR, Op->X80Src, FE, &Src)) {
        return false;
      }
      UsesFPR = true;
      const uint32_t ABits = FE.AllocI64Scratch();
      const uint32_t BBits = FE.AllocI64Scratch();
      EmitStackSlotAddr(FE, ContextBase, Op->SrcStack); // store address first
      PushF64FromV128(FE, Src);
      FE.Body.push_back(0xBD); // i64.reinterpret_f64
      PushLocalSet(FE.Body, ABits);
      EmitStackSlotAddr(FE, ContextBase, Op->SrcStack);
      PushF64Load(FE.Body, 0);
      FE.Body.push_back(0xBD);
      PushLocalSet(FE.Body, BBits);
      EmitX87InvalidFlagCheck(FE, ContextBase, ABits, BBits,
        IROp->Op == IR::IROps::OP_F80SUBRVALUE ? X87InvalidShape::Sub : X87InvalidShape::Div);
      PushF64FromV128(FE, Src);
      EmitStackSlotAddr(FE, ContextBase, Op->SrcStack);
      PushF64Load(FE.Body, 0);
      FE.Body.push_back(FloatBinOp(IROp->Op, 64));
      PushF64Store(FE.Body, 0);
      break;
    }
    case IR::IROps::OP_F80CMPVALUE:
    case IR::IROps::OP_F80CMPSTACK: {
      // Scalar compare of mm[top] against Src (or another stack slot). Two
      // consumers read the result: the x87-flag path (ConvertNZCVToX87
      // selects bits from the block NZCV) and FCOMI's Bfe of the FCMP_FLAG
      // bits (bit0=EQ bit1=LT bit2=UNORDERED). Both are produced here: the
      // Arm-shaped NZCV lands in the NZCV local, the FCMP_FLAG bits in the
      // GPR result.
      const bool FromValue = IROp->Op == IR::IROps::OP_F80CMPVALUE;
      const auto* Op = IROp->C<IR::IROp_F80CmpValue>();
      uint32_t SrcLocal = 0;
      if (FromValue) {
        if (!ResolveArg(IR, Op->X80Src, FE, &SrcLocal)) {
          return false;
        }
      }
      UsesFPR = true;
      const uint32_t LT = FE.AllocScratch();
      const uint32_t EQ = FE.AllocScratch();
      const uint32_t UNORD = FE.AllocScratch();
      if (FromValue) {
        EmitStackSlotAddr(FE, ContextBase, 0);
        PushF64Load(FE.Body, 0);
        PushF64FromV128(FE, SrcLocal);
        FE.Body.push_back(0x63); // f64.lt
        PushLocalSet(FE.Body, LT);
        EmitStackSlotAddr(FE, ContextBase, 0);
        PushF64Load(FE.Body, 0);
        PushF64FromV128(FE, SrcLocal);
        FE.Body.push_back(0x61); // f64.eq
        PushLocalSet(FE.Body, EQ);
        // unordered: either operand is NaN (self-equality fails)
        EmitStackSlotAddr(FE, ContextBase, 0);
        PushF64Load(FE.Body, 0);
        EmitStackSlotAddr(FE, ContextBase, 0);
        PushF64Load(FE.Body, 0);
        FE.Body.push_back(0x61); // f64.eq (a==a)
        FE.Body.push_back(0x45); // i32.eqz
        PushF64FromV128(FE, SrcLocal);
        PushF64FromV128(FE, SrcLocal);
        FE.Body.push_back(0x61); // f64.eq (b==b)
        FE.Body.push_back(0x45); // i32.eqz
        FE.Body.push_back(0x72); // i32.or
        PushLocalSet(FE.Body, UNORD);
      } else {
        const auto* StackOp = IROp->C<IR::IROp_F80CmpStack>();
        EmitStackSlotAddr(FE, ContextBase, 0);
        PushF64Load(FE.Body, 0);
        EmitStackSlotAddr(FE, ContextBase, StackOp->SrcStack);
        PushF64Load(FE.Body, 0);
        FE.Body.push_back(0x63); // f64.lt
        PushLocalSet(FE.Body, LT);
        EmitStackSlotAddr(FE, ContextBase, 0);
        PushF64Load(FE.Body, 0);
        EmitStackSlotAddr(FE, ContextBase, StackOp->SrcStack);
        PushF64Load(FE.Body, 0);
        FE.Body.push_back(0x61); // f64.eq
        PushLocalSet(FE.Body, EQ);
        EmitStackSlotAddr(FE, ContextBase, 0);
        PushF64Load(FE.Body, 0);
        EmitStackSlotAddr(FE, ContextBase, 0);
        PushF64Load(FE.Body, 0);
        FE.Body.push_back(0x61);
        FE.Body.push_back(0x45); // !eq(a,a) -> a is NaN
        EmitStackSlotAddr(FE, ContextBase, StackOp->SrcStack);
        PushF64Load(FE.Body, 0);
        EmitStackSlotAddr(FE, ContextBase, StackOp->SrcStack);
        PushF64Load(FE.Body, 0);
        FE.Body.push_back(0x61);
        FE.Body.push_back(0x45); // !eq(b,b) -> b is NaN
        FE.Body.push_back(0x72); // i32.or
        PushLocalSet(FE.Body, UNORD);
      }
      // Arm-shaped NZCV (N=lt, Z=eq, C=!lt|unordered, V=unordered) plus the
      // FCMP_FLAG result bits (bit0=EQ bit1=LT bit2=UNORDERED).
      EmitCompareResult(FE, LT, EQ, UNORD, Dest, true);
      break;
    }
    case IR::IROps::OP_FLOAT_FTOF: {
      // Float conversion between the scalar lanes. Dest/Header.Size is the
      // destination element size, the field the source element size.
      const auto* Op = IROp->C<IR::IROp_Float_FToF>();
      uint32_t Src;
      if (!ResolveArg(IR, Op->Scalar, FE, &Src)) {
        return false;
      }
      UsesFPR = true;
      const uint32_t DstBits = IR::OpSizeAsBits(IROp->Size);
      const uint32_t SrcBits = IR::OpSizeAsBits(Op->SrcElementSize);
      if (DstBits == 64 && SrcBits == 32) {
        PushF32FromV128(FE, Src);
        FE.Body.push_back(0xBB); // f64.promote_f32
        PushVec(FE.Body, {0xFD, 0x14}); // f64x2.splat
      } else if (DstBits == 32 && SrcBits == 64) {
        PushF64FromV128(FE, Src);
        FE.Body.push_back(0xB6); // f32.demote_f64
        PushVec(FE.Body, {0xFD, 0x13}); // f32x4.splat
      } else if (DstBits == SrcBits) {
        PushLocalGet(FE.Body, Src);
      } else {
        LogMan::Msg::EFmt("WasmBackend: Float_FToF {}->{} unsupported", SrcBits, DstBits);
        return false;
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_FLOAT_FROMGPR_S: {
      // Signed GPR -> scalar float, into lane 0 (upper lanes zeroed).
      const auto* Op = IROp->C<IR::IROp_Float_FromGPR_S>();
      uint32_t Src;
      if (!ResolveArg(IR, Op->Src, FE, &Src)) {
        return false;
      }
      UsesFPR = true;
      const uint32_t DstBits = IR::OpSizeAsBits(IROp->Size);
      const uint32_t SrcBits = IR::OpSizeAsBits(Op->SrcElementSize);
      if (DstBits == 64 && SrcBits == 32) {
        PushLocalGet(FE.Body, Src);
        FE.Body.push_back(0xB7); // f64.convert_i32_s
        PushVec(FE.Body, {0xFD, 0x14}); // f64x2.splat
      } else if (DstBits == 64 && SrcBits == 64) {
        PushLocalGet(FE.Body, Src);
        FE.Body.push_back(0xB9); // f64.convert_i64_s
        PushVec(FE.Body, {0xFD, 0x14});
      } else if (DstBits == 32 && SrcBits == 32) {
        PushLocalGet(FE.Body, Src);
        FE.Body.push_back(0xB2); // f32.convert_i32_s
        PushVec(FE.Body, {0xFD, 0x13}); // f32x4.splat
      } else {
        LogMan::Msg::EFmt("WasmBackend: Float_FromGPR_S {}->{} unsupported", SrcBits, DstBits);
        return false;
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_FLOAT_TOGPR_ZS:
    case IR::IROps::OP_FLOAT_TOGPR_S: {
      // Scalar -> signed GPR. Truncating (ZS) matches cvttss2si/fisttp; the
      // rounding variant (S) matches fistp's default FCW rounding. x86
      // yields the integer-indefinite value (INT_MIN) for out-of-range and
      // NaN inputs; wasm's trunc traps there, so the value is guarded by a
      // select to 0.0 before the trunc (then the guard re-selects INT_MIN).
      const bool Truncating = IROp->Op == IR::IROps::OP_FLOAT_TOGPR_ZS;
      const auto* Op = IROp->C<IR::IROp_Float_ToGPR_ZS>();
      uint32_t Src;
      if (!ResolveArg(IR, Op->Scalar, FE, &Src)) {
        return false;
      }
      UsesFPR = true;
      const uint32_t DstBits = IR::OpSizeAsBits(IROp->Size);
      const uint32_t SrcBits = IR::OpSizeAsBits(Op->SrcElementSize);
      if (DstBits != 32) {
        LogMan::Msg::EFmt("WasmBackend: Float_ToGPR with unsupported dest size {}", DstBits);
        return false;
      }
      // in-range = (v >= -2^31) & (v < 2^31) & (v == v). The condition is
      // computed after pushing the select operands so it lands on top
      // (select pops [val1, val2, cond]).
      if (SrcBits == 64) {
        // in-range -> scratch (consumed by both selects)
        const uint32_t InRange = FE.AllocScratch();
        PushF64FromV128(FE, Src); // val1 = the guarded value
        PushF64Const(FE.Body, 0); // val2
        PushF64FromV128(FE, Src);
        PushF64Const(FE.Body, 0xC1E0000000000000ULL); // -2147483648.0
        FE.Body.push_back(0x66);                      // f64.ge
        PushF64FromV128(FE, Src);
        PushF64Const(FE.Body, 0x41E0000000000000ULL); // 2147483648.0
        FE.Body.push_back(0x63);                      // f64.lt
        FE.Body.push_back(0x71);                      // i32.and
        PushF64FromV128(FE, Src);
        PushF64FromV128(FE, Src);
        FE.Body.push_back(0x61); // f64.eq (NaN check)
        FE.Body.push_back(0x71); // i32.and
        PushLocalSet(FE.Body, InRange);
        PushLocalGet(FE.Body, InRange);
        FE.Body.push_back(0x1B); // select -> v' = in_range ? v : 0.0
        if (!Truncating) {
          FE.Body.push_back(0x9E); // f64.nearest (round-to-nearest-even)
        }
        FE.Body.push_back(0xAA); // i32.trunc_f64_s (safe: in-range value)
        // out-of-range/NaN -> INT_MIN (the x86 integer-indefinite value)
        PushU32Const(FE.Body, 0x80000000);
        PushLocalGet(FE.Body, InRange);
        FE.Body.push_back(0x1B); // i32.select
      } else if (SrcBits == 32) {
        const uint32_t InRange32 = FE.AllocScratch();
        PushF32FromV128(FE, Src); // val1
        PushF32Const(FE.Body, 0); // val2
        PushF32FromV128(FE, Src);
        PushF32Const(FE.Body, 0xCF000000); // -2147483648.0f
        FE.Body.push_back(0x60);           // f32.ge
        PushF32FromV128(FE, Src);
        PushF32Const(FE.Body, 0x4F000000); // 2147483648.0f
        FE.Body.push_back(0x5D);           // f32.lt
        FE.Body.push_back(0x71);           // i32.and
        PushF32FromV128(FE, Src);
        PushF32FromV128(FE, Src);
        FE.Body.push_back(0x5B); // f32.eq (NaN check)
        FE.Body.push_back(0x71); // i32.and
        PushLocalSet(FE.Body, InRange32);
        PushLocalGet(FE.Body, InRange32);
        FE.Body.push_back(0x1B); // select -> v'
        if (!Truncating) {
          FE.Body.push_back(0x90); // f32.nearest (round-to-nearest-even)
        }
        FE.Body.push_back(0xA8); // i32.trunc_f32_s (safe: in-range value)
        PushU32Const(FE.Body, 0x80000000);
        PushLocalGet(FE.Body, InRange32);
        FE.Body.push_back(0x1B); // i32.select
      } else {
        LogMan::Msg::EFmt("WasmBackend: Float_ToGPR with unsupported src size {}", SrcBits);
        return false;
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_VECTOR_FTOZS: {
      // Vector float -> vector int truncate (fistp m64's path): lane 0 f64
      // becomes a lane 0 i64 via the same guarded conversion as above.
      const auto* Op = IROp->C<IR::IROp_Vector_FToZS>();
      uint32_t Src;
      if (!ResolveArg(IR, Op->Vector, FE, &Src)) {
        return false;
      }
      UsesFPR = true;
      const uint32_t InRange = FE.AllocScratch();
      PushF64FromV128(FE, Src);
      PushF64Const(FE.Body, 0xC3E0000000000000ULL); // -2^63
      FE.Body.push_back(0x66);                      // f64.ge
      PushF64FromV128(FE, Src);
      PushF64Const(FE.Body, 0x43E0000000000000ULL); // 2^63
      FE.Body.push_back(0x63);                      // f64.lt
      FE.Body.push_back(0x71);                      // i32.and
      PushF64FromV128(FE, Src);
      PushF64FromV128(FE, Src);
      FE.Body.push_back(0x61); // f64.eq (NaN check)
      FE.Body.push_back(0x71); // i32.and
      PushLocalSet(FE.Body, InRange);
      PushF64FromV128(FE, Src); // val1 = the guarded value
      PushF64Const(FE.Body, 0); // val2
      PushLocalGet(FE.Body, InRange);
      FE.Body.push_back(0x1B); // i64.select -> v'
      FE.Body.push_back(0xB0); // i64.trunc_f64_s (safe: in-range value)
      // out-of-range/NaN -> INT64_MIN (the x87 integer-indefinite value)
      PushU64Const(FE.Body, INT64_MIN);
      PushLocalGet(FE.Body, InRange);
      FE.Body.push_back(0x1B); // i64.select
      PushVec(FE.Body, {0xFD, 0x12}); // i64x2.splat
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_VCASTFROMGPR: {
      // Element 0 = the GPR value, upper elements zero (movd/movq to xmm).
      const auto* Op = IROp->C<IR::IROp_VCastFromGPR>();
      const uint32_t ElemBits = IR::OpSizeAsBits(IROp->ElementSize);
      if (ElemBits == 64 && !Op->Src.IsImmediate()) {
        auto* SrcHeader = IR->GetOp<IR::IROp_Header>(IR->GetNode(Op->Src));
        if (SrcHeader->Op == IR::IROps::OP_CONSTANT) {
          // fild-free path: FLDF64_Const's bit-pattern constant (an i64 that
          // cannot live in an i32 local) becomes an f64.const directly.
          PushVec(FE.Body, {0xFD, 0x0C}); // v128.const
          for (uint32_t i = 0; i < 16; ++i) {
            FE.Body.push_back(0);
          }
          PushU64Const(FE.Body, SrcHeader->C<IR::IROp_Constant>()->Constant);
          PushVec(FE.Body, {0xFD, 0x1E, 0x00}); // i64x2.replace_lane 0
          PushLocalSet(FE.Body, Dest);
          break;
        }
      }
      uint32_t Src;
      if (!ResolveArg(IR, Op->Src, FE, &Src)) {
        return false;
      }
      if (ElemBits == 64 && !FE.LocalIsI64(Src)) {
        // The 64-bit element source must be an i64 local (a memory-loaded
        // m64 or the typed bit-pattern constant) — a 32-bit local cannot be
        // widened (the value would already be truncated).
        LogMan::Msg::EFmt("WasmBackend: VCastFromGPR 64-bit with a non-i64 source local");
        return false;
      }
      UsesFPR = true;
      PushVec(FE.Body, {0xFD, 0x0C}); // v128.const 0
      for (uint32_t i = 0; i < 16; ++i) {
        FE.Body.push_back(0);
      }
      PushLocalGet(FE.Body, Src);
      if (ElemBits == 32) {
        PushVec(FE.Body, {0xFD, 0x1C, 0x00}); // i32x4.replace_lane 0
      } else if (ElemBits == 64) {
        PushVec(FE.Body, {0xFD, 0x1E, 0x00}); // i64x2.replace_lane 0
      } else {
        LogMan::Msg::EFmt("WasmBackend: VCastFromGPR with unsupported element size {}", ElemBits);
        return false;
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_VMOV: {
      // Copy the source vector, truncating to the op's effective size and
      // zero-extending (the movd/movq partial-load shape: LoadXMMRegister's
      // "extract the subregister" path). extract-lane + splat of the matching
      // width is exactly that zero-extension. The size rides the header's
      // Size field (the Arm64 DEF_OP(VMov) reads the same).
      const uint32_t ElemBits = IR::OpSizeAsBits(IROp->Size);
      uint32_t Src;
      if (!ResolveArg(IR, IROp->C<IR::IROp_VMov>()->Source, FE, &Src)) {
        return false;
      }
      UsesFPR = true;
      if (ElemBits == 128) {
        PushLocalGet(FE.Body, Src);
      } else {
        uint8_t Extract, Splat;
        switch (ElemBits) {
        case 8:  Extract = 0x16; Splat = 0x0F; break; // i8x16.extract_lane_u / i8x16.splat
        case 16: Extract = 0x19; Splat = 0x10; break; // i16x8.extract_lane_u / i16x8.splat
        case 32: Extract = 0x1B; Splat = 0x11; break; // i32x4.extract_lane / i32x4.splat
        case 64: Extract = 0x1D; Splat = 0x12; break; // i64x2.extract_lane / i64x2.splat
        default:
          LogMan::Msg::EFmt("WasmBackend: VMov with unsupported element size {}", ElemBits);
          return false;
        }
        PushLocalGet(FE.Body, Src);
        PushVec(FE.Body, {0xFD, Extract, 0x00}); // lane 0
        PushVec(FE.Body, {0xFD, Splat});
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_VEXTRACTTOGPR: {
      // movd/movq xmm -> GPR: raw bits of lane Index.
      const auto* Op = IROp->C<IR::IROp_VExtractToGPR>();
      uint32_t Src;
      if (!ResolveArg(IR, Op->Vector, FE, &Src)) {
        return false;
      }
      UsesFPR = true;
      const uint32_t ElemBits = IR::OpSizeAsBits(IROp->ElementSize);
      PushLocalGet(FE.Body, Src);
      if (ElemBits == 32) {
        PushVec(FE.Body, {0xFD, 0x1B}); // i32x4.extract_lane
        PushULEB(FE.Body, Op->Index);
      } else if (ElemBits == 64) {
        PushVec(FE.Body, {0xFD, 0x1D}); // i64x2.extract_lane
        PushULEB(FE.Body, Op->Index);
      } else {
        LogMan::Msg::EFmt("WasmBackend: VExtractToGPR with unsupported element size {}", ElemBits);
        return false;
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_VINSELEMENT: {
      // DestVector = DestVector with lane DestIdx replaced by SrcVector's
      // lane SrcIdx (the movss/movsd reg,reg shape among others).
      const auto* Op = IROp->C<IR::IROp_VInsElement>();
      uint32_t D, S;
      if (!ResolveArg(IR, Op->DestVector, FE, &D) || !ResolveArg(IR, Op->SrcVector, FE, &S)) {
        return false;
      }
      UsesFPR = true;
      const uint32_t ElemBits = IR::OpSizeAsBits(IROp->ElementSize);
      PushLocalGet(FE.Body, D);
      PushLocalGet(FE.Body, S);
      if (ElemBits == 32) {
        PushVec(FE.Body, {0xFD, 0x1B}); // i32x4.extract_lane
        PushULEB(FE.Body, Op->SrcIdx);
        PushVec(FE.Body, {0xFD, 0x1C}); // i32x4.replace_lane
        PushULEB(FE.Body, Op->DestIdx);
      } else if (ElemBits == 64) {
        PushVec(FE.Body, {0xFD, 0x1D}); // i64x2.extract_lane
        PushULEB(FE.Body, Op->SrcIdx);
        PushVec(FE.Body, {0xFD, 0x1E}); // i64x2.replace_lane
        PushULEB(FE.Body, Op->DestIdx);
      } else {
        LogMan::Msg::EFmt("WasmBackend: VInsElement with unsupported element size {}", ElemBits);
        return false;
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_VDUPELEMENT: {
      // Broadcast one element across the whole register (the pshufd imm-0
      // splat among others): extract_lane + splat is wasm's literal
      // spelling of the AArch64 dup ground truth (JIT/VectorOps.cpp
      // DEF_OP(VDupElement)) — the same two primitives OP_VMOV's
      // zero-extending copy rides. The FSbb2 wall: the d3d9-l6v5u5 write
      // loop's `pshufd $0x0,%xmm0,%xmm0` (rip 0x4027ce) lifted to
      // VDupElement and the emit refused (unsupported IR op 217, err -2).
      const auto* Op = IROp->C<IR::IROp_VDupElement>();
      uint32_t Src;
      if (!ResolveArg(IR, Op->Vector, FE, &Src)) {
        return false;
      }
      UsesFPR = true;
      const uint32_t RegBits = IR::OpSizeAsBits(IROp->Size);
      if (RegBits != 128) {
        // The 128-bit form is the only one the SSE frontend emits
        // (Single128Bit4ByteVectorShuffle's broadcast shapes); a smaller
        // register size would need the splat truncated — loud refusal over
        // silently-wrong wasm (the Andn/Orlshr named-refusal rule).
        LogMan::Msg::EFmt("WasmBackend: VDupElement with unsupported register size {}", RegBits);
        return false;
      }
      const uint32_t ElemBits = IR::OpSizeAsBits(IROp->ElementSize);
      uint8_t Extract, Splat;
      switch (ElemBits) {
        case 8:  Extract = 0x16; Splat = 0x0F; break; // i8x16.extract_lane_u / i8x16.splat
        case 16: Extract = 0x19; Splat = 0x10; break; // i16x8.extract_lane_u / i16x8.splat
        case 32: Extract = 0x1B; Splat = 0x11; break; // i32x4.extract_lane / i32x4.splat
        case 64: Extract = 0x1D; Splat = 0x12; break; // i64x2.extract_lane / i64x2.splat
        default:
          LogMan::Msg::EFmt("WasmBackend: VDupElement with unsupported element size {}", ElemBits);
          return false;
      }
      PushLocalGet(FE.Body, Src);
      PushVec(FE.Body, {0xFD, Extract});
      PushULEB(FE.Body, Op->Index);
      PushVec(FE.Body, {0xFD, Splat});
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_VZIP: {
      // DestVector = the low-half interleave (the punpckl* family): the
      // dest's even lanes keep VectorLower's, the odd lanes take
      // VectorUpper's. The wasm lane extract/replace pair covers every
      // element size the frontend emits (the P2.12 tier's movd/movq ride
      // the same primitives); the trailing half-lanes keep the dest's.
      const auto* Op = IROp->C<IR::IROp_VZip>();
      uint32_t D, S;
      if (!ResolveArg(IR, Op->VectorLower, FE, &D) || !ResolveArg(IR, Op->VectorUpper, FE, &S)) {
        return false;
      }
      UsesFPR = true;
      const uint32_t ElemBits = IR::OpSizeAsBits(IROp->ElementSize);
      const uint32_t RegBits = IR::OpSizeAsBits(IROp->Size);
      const uint32_t Pairs = (RegBits / ElemBits) / 2;
      uint8_t ExtractOp, ReplaceOp;
      switch (ElemBits) {
        case 8:  ExtractOp = 0x16; ReplaceOp = 0x17; break; // i8x16.extract_lane_u / replace_lane
        case 16: ExtractOp = 0x19; ReplaceOp = 0x1A; break; // i16x8.extract_lane_u / replace_lane
        case 32: ExtractOp = 0x1B; ReplaceOp = 0x1C; break; // i32x4.extract_lane / replace_lane
        case 64: ExtractOp = 0x1D; ReplaceOp = 0x1E; break; // i64x2.extract_lane / replace_lane
        default:
          LogMan::Msg::EFmt("WasmBackend: VZip with unsupported element size {}", ElemBits);
          return false;
      }
      if (Pairs == 0 || Pairs > 8) {
        LogMan::Msg::EFmt("WasmBackend: VZip with unsupported lane count {}", Pairs * 2);
        return false;
      }
      // The accumulate runs on the operand stack: VectorLower is pushed
      // ONCE, then each pair replaces its odd lane in place
      // (replace_lane pops the current accumulate). The trailing set
      // stores the complete interleave — W1's defect was re-reading the
      // unmodified VectorLower per pair, which dangled the intermediates
      // and left Dest stale (the d3d9-clear punpckldq movq store landed
      // {632, 0}). zip1's result depends only on the two source vectors,
      // never on Dest's prior value (Dest may be fresh — the shuffle
      // builders' _VZip(Src, Src) shapes).
      PushLocalGet(FE.Body, D);
      for (uint32_t k = 0; k < Pairs; ++k) {
        PushLocalGet(FE.Body, S);
        PushVec(FE.Body, {0xFD, ExtractOp});
        PushULEB(FE.Body, k);
        PushVec(FE.Body, {0xFD, ReplaceOp});
        PushULEB(FE.Body, 2 * k + 1);
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_FCMP: {
      // Arm-FCMP-shaped NZCV from a scalar compare (the UCOMIS path; the
      // x86 form comes from the following AXFlag).
      const auto* Op = IROp->C<IR::IROp_FCmp>();
      uint32_t A, B;
      if (!ResolveArg(IR, Op->Scalar1, FE, &A) || !ResolveArg(IR, Op->Scalar2, FE, &B)) {
        return false;
      }
      UsesFPR = true;
      EmitFloatCompare(FE, A, B, IR::OpSizeAsBits(Op->ElementSize), Dest, false);
      break;
    }
    case IR::IROps::OP_AXFLAG: {
      // Arm NZCV -> the x86 flag values in the canonical stored form
      // (docs/spikes/p2-x87-sse.md): newC = C_arm & !V (the inverted x86 CF),
      // newZ = Z | V, N = V = 0 — the AXFLAG contract's stored-C column.
      const uint32_t NewC = FE.AllocScratch();
      const uint32_t NewZ = FE.AllocScratch();
      // newC = C & !V
      PushLocalGet(FE.Body, FE.NZCV());
      PushU32Const(FE.Body, 29);
      FE.Body.push_back(0x76); // i32.shr_u
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x71); // i32.and
      PushLocalGet(FE.Body, FE.NZCV());
      PushU32Const(FE.Body, 28);
      FE.Body.push_back(0x76); // i32.shr_u
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x71); // i32.and
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x73); // i32.xor -> !V
      FE.Body.push_back(0x71); // i32.and
      PushLocalSet(FE.Body, NewC);
      // newZ = Z | V
      PushLocalGet(FE.Body, FE.NZCV());
      PushU32Const(FE.Body, 30);
      FE.Body.push_back(0x76);
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x71);
      PushLocalGet(FE.Body, FE.NZCV());
      PushU32Const(FE.Body, 28);
      FE.Body.push_back(0x76);
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x71);
      FE.Body.push_back(0x72); // i32.or
      PushLocalSet(FE.Body, NewZ);
      // nzcv = newC<<29 | newZ<<30 (N and V cleared by construction)
      PushLocalGet(FE.Body, NewC);
      PushU32Const(FE.Body, 29);
      FE.Body.push_back(0x74);
      PushLocalGet(FE.Body, NewZ);
      PushU32Const(FE.Body, 30);
      FE.Body.push_back(0x74);
      FE.Body.push_back(0x72); // i32.or
      PushLocalSet(FE.Body, FE.NZCV());
      break;
    }
    case IR::IROps::OP_VFADD:
    case IR::IROps::OP_VFSUB:
    case IR::IROps::OP_VFMUL:
    case IR::IROps::OP_VFDIV: {
      // Packed float lanes (addps/addpd and friends).
      uint32_t A, B;
      if (!ResolveArg(IR, IROp->Args[0], FE, &A) || !ResolveArg(IR, IROp->Args[1], FE, &B)) {
        return false;
      }
      UsesFPR = true;
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      FE.Body.push_back(0xFD);
      const uint8_t VIdx = IROp->Op == IR::IROps::OP_VFADD ? 0
        : IROp->Op == IR::IROps::OP_VFSUB ? 1
        : IROp->Op == IR::IROps::OP_VFMUL ? 2
        : 3;
      // The SIMD opcode is a multi-byte LEB: f32x4 add/sub/mul/div = 0xE4-E7,
      // f64x2 = 0xF0-F3, all with the 0x01 continuation byte.
      FE.Body.push_back((IR::OpSizeAsBits(IROp->ElementSize) == 32 ? 0xE4 : 0xF0) + VIdx);
      FE.Body.push_back(0x01);
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_VFADDSCALARINSERT:
    case IR::IROps::OP_VFSUBSCALARINSERT:
    case IR::IROps::OP_VFMULSCALARINSERT:
    case IR::IROps::OP_VFDIVSCALARINSERT: {
      // Scalar op inserted into lane 0 of Vector1, upper lanes preserved
      // (the addss/subss/mulss/divss shape).
      const auto* Op = IROp->C<IR::IROp_VFAddScalarInsert>();
      uint32_t V1, V2;
      if (!ResolveArg(IR, Op->Vector1, FE, &V1) || !ResolveArg(IR, Op->Vector2, FE, &V2)) {
        return false;
      }
      UsesFPR = true;
      const uint32_t ElemBits = IR::OpSizeAsBits(IROp->ElementSize);
      PushLocalGet(FE.Body, V1);
      PushLocalGet(FE.Body, V1); // the scalar operand is Vector1's lane 0
      if (ElemBits == 32) {
        PushVec(FE.Body, {0xFD, 0x1F, 0x00}); // f32x4.extract_lane 0
      } else {
        PushVec(FE.Body, {0xFD, 0x21, 0x00}); // f64x2.extract_lane 0
      }
      PushLocalGet(FE.Body, V2);
      if (ElemBits == 32) {
        PushVec(FE.Body, {0xFD, 0x1F, 0x00});
      } else {
        PushVec(FE.Body, {0xFD, 0x21, 0x00});
      }
      FE.Body.push_back(FloatBinOp(IROp->Op, IR::OpSizeAsBits(IROp->ElementSize)));
      if (ElemBits == 32) {
        PushVec(FE.Body, {0xFD, 0x20, 0x00}); // f32x4.replace_lane 0
      } else {
        PushVec(FE.Body, {0xFD, 0x22, 0x00}); // f64x2.replace_lane 0
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_VFSQRTSCALARINSERT: {
      // Scalar sqrt inserted into lane 0 of Vector1, upper lanes preserved
      // (the sqrtsd/ss shape). wasm f64.sqrt's NaN/negative semantics match
      // x86's (invalid-operation QNaN), so a plain extract-sqrt-replace is
      // faithful.
      const auto* Op = IROp->C<IR::IROp_VFSqrtScalarInsert>();
      uint32_t V1, V2;
      if (!ResolveArg(IR, Op->Vector1, FE, &V1) || !ResolveArg(IR, Op->Vector2, FE, &V2)) {
        return false;
      }
      UsesFPR = true;
      PushLocalGet(FE.Body, V1);
      PushLocalGet(FE.Body, V2);
      if (IR::OpSizeAsBits(IROp->ElementSize) == 32) {
        PushVec(FE.Body, {0xFD, 0x1F, 0x00}); // f32x4.extract_lane 0
        FE.Body.push_back(0x91);              // f32.sqrt
        PushVec(FE.Body, {0xFD, 0x20, 0x00}); // f32x4.replace_lane 0
      } else {
        PushVec(FE.Body, {0xFD, 0x21, 0x00}); // f64x2.extract_lane 0
        FE.Body.push_back(0x9F);              // f64.sqrt
        PushVec(FE.Body, {0xFD, 0x22, 0x00}); // f64x2.replace_lane 0
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_VFMINSCALARINSERT:
    case IR::IROps::OP_VFMAXSCALARINSERT: {
      // Scalar min/max inserted into lane 0 of Vector1, upper lanes
      // preserved (the minsd/maxsd shape). IR.json's x86 contract: NaN and
      // signed-zero pairs return the SECOND operand — so wasm's f64.min/max
      // (NaN-propagating) must not be used; a select on (Src OP Dest) is
      // exact: SRC > DEST false on NaN and on equal (incl. ±0 pairs) -> SRC.
      const bool IsMin = IROp->Op == IR::IROps::OP_VFMINSCALARINSERT;
      const auto* Op = IROp->C<IR::IROp_VFMinScalarInsert>();
      uint32_t V1, V2;
      if (!ResolveArg(IR, Op->Vector1, FE, &V1) || !ResolveArg(IR, Op->Vector2, FE, &V2)) {
        return false;
      }
      UsesFPR = true;
      const bool Is32 = IR::OpSizeAsBits(IROp->ElementSize) == 32;
      PushLocalGet(FE.Body, V1); // replace_lane's v128 base (upper lanes from DEST)
      PushLocalGet(FE.Body, V1); // select val1: DEST's lane 0
      PushVec(FE.Body, Is32 ? std::initializer_list<uint8_t>{0xFD, 0x1F, 0x00} : std::initializer_list<uint8_t>{0xFD, 0x21, 0x00});
      PushLocalGet(FE.Body, V2);
      PushVec(FE.Body, Is32 ? std::initializer_list<uint8_t>{0xFD, 0x1F, 0x00} : std::initializer_list<uint8_t>{0xFD, 0x21, 0x00});
      // cond = SRC OP DEST: f32.gt 0x5E / f32.lt 0x5D; f64.gt 0x64 / f64.lt 0x63.
      PushLocalGet(FE.Body, V2);
      PushVec(FE.Body, Is32 ? std::initializer_list<uint8_t>{0xFD, 0x1F, 0x00} : std::initializer_list<uint8_t>{0xFD, 0x21, 0x00});
      PushLocalGet(FE.Body, V1);
      PushVec(FE.Body, Is32 ? std::initializer_list<uint8_t>{0xFD, 0x1F, 0x00} : std::initializer_list<uint8_t>{0xFD, 0x21, 0x00});
      FE.Body.push_back(Is32 ? (IsMin ? 0x5E : 0x5D) : (IsMin ? 0x64 : 0x63));
      FE.Body.push_back(0x1B); // i32.select (false -> SRC, the x86 NaN/zero rule)
      PushVec(FE.Body, Is32 ? std::initializer_list<uint8_t>{0xFD, 0x20, 0x00} : std::initializer_list<uint8_t>{0xFD, 0x22, 0x00});
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_PARITY: {
      // x86 PF: even parity of the deferred raw PF value's low byte (the
      // Arm lowering's xor-fold cascade). bit0 = parity(Raw & 0xFF) ^ Invert;
      // the jp/jnp dispatch inverts. The upper bits of a Mask-less Parity are
      // garbage by contract (consumers Sbfe bit 0) — we always normalize to
      // 0/1, which is faithful for every consumer.
      const auto* Op = IROp->C<IR::IROp_Parity>();
      uint32_t Raw;
      if (!ResolveArg(IR, Op->Raw, FE, &Raw)) {
        return false;
      }
      const uint32_t T = FE.AllocScratch();
      PushLocalGet(FE.Body, Raw);
      PushLocalGet(FE.Body, Raw);
      PushU32Const(FE.Body, 4);
      FE.Body.push_back(0x74); // i32.shr_u
      FE.Body.push_back(0x73); // i32.xor
      PushLocalSet(FE.Body, T);
      PushLocalGet(FE.Body, T);
      PushLocalGet(FE.Body, T);
      PushU32Const(FE.Body, 2);
      FE.Body.push_back(0x74); // i32.shr_u
      FE.Body.push_back(0x73); // i32.xor
      PushLocalSet(FE.Body, T);
      PushLocalGet(FE.Body, T);
      PushLocalGet(FE.Body, T);
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x74); // i32.shr_u
      FE.Body.push_back(0x73); // i32.xor
      PushU32Const(FE.Body, 1);
      FE.Body.push_back(0x71); // i32.and -> bit0 = low-byte parity
      if (Op->Invert) {
        PushU32Const(FE.Body, 1);
        FE.Body.push_back(0x73); // i32.xor (the jp inversion)
      }
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_ORLSHL: {
      // Src1 | (Src2 << BitShift) — the FSW reconstruction's bit assembly.
      const auto* Op = IROp->C<IR::IROp_Orlshl>();
      uint32_t A, B;
      if (!ResolveArg(IR, Op->Src1, FE, &A) || !ResolveArg(IR, Op->Src2, FE, &B)) {
        return false;
      }
      PushLocalGet(FE.Body, A);
      PushLocalGet(FE.Body, B);
      if (Op->BitShift != 0) {
        PushU32Const(FE.Body, Op->BitShift);
        FE.Body.push_back(0x74); // i32.shl
      }
      FE.Body.push_back(0x72); // i32.or
      PushLocalSet(FE.Body, Dest);
      break;
    }
    case IR::IROps::OP_LOADMEMTSO:
    case IR::IROps::OP_LOADMEM: {
      // D3a subset: GPR class, <=32-bit, zero-translation (guest VA == wasm
      // offset, doc 01 §5). 8/16-bit GPR loads zero-extend (x86 mov semantics).
      // TSO and non-TSO GPR loads lower identically for v0 (no fences).
      // P2.12: FPR-class loads feed the v128 xmm model (sub-vector loads
      // zero the upper lanes, matching the zero-extend contract).
      const bool IsTSO = IROp->Op == IR::IROps::OP_LOADMEMTSO;
      if (IsTSO) {
        const auto* LM = IROp->C<IR::IROp_LoadMemTSO>();
        if (LM->Class == IR::RegClass::FPR) {
          UsesFPR = true;
          if (!EmitFPRLoad(IR, LM, IROp->Size, FE, Dest)) {
            return false;
          }
          break;
        }
      } else {
        const auto* LM = IROp->C<IR::IROp_LoadMem>();
        if (LM->Class == IR::RegClass::FPR) {
          UsesFPR = true;
          if (!EmitFPRLoad(IR, LM, IROp->Size, FE, Dest)) {
            return false;
          }
          break;
        }
      }
      if (IsTSO ? !EmitGPRLoad(IR, IROp->C<IR::IROp_LoadMemTSO>(), IROp->Size, FE, Dest)
                : !EmitGPRLoad(IR, IROp->C<IR::IROp_LoadMem>(), IROp->Size, FE, Dest)) {
        return false;
      }
      break;
    }
    case IR::IROps::OP_STOREMEMTSO:
    case IR::IROps::OP_STOREMEM: {
      // TSO and non-TSO GPR stores lower identically for v0: plain wasm
      // i32.store (no ordering fences; documented as a naive-backend limit).
      // P2.12: FPR stores extract lane 0 for sub-vector sizes.
      const bool IsTSO = IROp->Op == IR::IROps::OP_STOREMEMTSO;
      if (IsTSO) {
        const auto* SM = IROp->C<IR::IROp_StoreMemTSO>();
        if (SM->Class == IR::RegClass::FPR) {
          uint32_t Value;
          if (!ResolveArg(IR, SM->Value, FE, &Value)) {
            return false;
          }
          UsesFPR = true;
          if (!EmitFPRStore(IR, SM, IROp->Size, FE, Value)) {
            return false;
          }
          break;
        }
      } else {
        const auto* SM = IROp->C<IR::IROp_StoreMem>();
        if (SM->Class == IR::RegClass::FPR) {
          uint32_t Value;
          if (!ResolveArg(IR, SM->Value, FE, &Value)) {
            return false;
          }
          UsesFPR = true;
          if (!EmitFPRStore(IR, SM, IROp->Size, FE, Value)) {
            return false;
          }
          break;
        }
      }
      if (IsTSO ? !EmitGPRStore(IR, IROp->C<IR::IROp_StoreMemTSO>(), IROp->Size, FE)
                : !EmitGPRStore(IR, IROp->C<IR::IROp_StoreMem>(), IROp->Size, FE)) {
        return false;
      }
      break;
    }
    case IR::IROps::OP_ATOMICFETCHADD:
    case IR::IROps::OP_ATOMICFETCHSUB:
    case IR::IROps::OP_ATOMICFETCHAND:
    case IR::IROps::OP_ATOMICFETCHOR:
    case IR::IROps::OP_ATOMICFETCHXOR:
    case IR::IROps::OP_ATOMICSWAP: {
      // P3.1: the LOCK-prefixed read-modify-write class (lock add/sub/and/or/
      // xor, xadd, xchg mem) → wasm atomic RMW. The guest linear memory is
      // unshared, which the wasm threads proposal allows atomics on; every
      // emitted op is seq-cst, matching the LOCK contract. AtomicFetchCLR
      // (Arm-only `&= ~v`) and AtomicFetchNeg have no wasm RMW form and keep
      // the default loud-fail (pinned by negative corpus cases).
      AtomicRmw Kind;
      switch (IROp->Op) {
      case IR::IROps::OP_ATOMICFETCHADD: Kind = AtomicRmw::Add; break;
      case IR::IROps::OP_ATOMICFETCHSUB: Kind = AtomicRmw::Sub; break;
      case IR::IROps::OP_ATOMICFETCHAND: Kind = AtomicRmw::And; break;
      case IR::IROps::OP_ATOMICFETCHOR: Kind = AtomicRmw::Or; break;
      case IR::IROps::OP_ATOMICFETCHXOR: Kind = AtomicRmw::Xor; break;
      default: Kind = AtomicRmw::Xchg; break; // OP_ATOMICSWAP
      }
      bool ok;
      switch (IROp->Op) {
      case IR::IROps::OP_ATOMICSWAP:
        ok = EmitAtomicFetch(IR, IROp->C<IR::IROp_AtomicSwap>(), Kind, IROp->Size, FE, Dest);
        break;
      case IR::IROps::OP_ATOMICFETCHADD:
        ok = EmitAtomicFetch(IR, IROp->C<IR::IROp_AtomicFetchAdd>(), Kind, IROp->Size, FE, Dest);
        break;
      case IR::IROps::OP_ATOMICFETCHSUB:
        ok = EmitAtomicFetch(IR, IROp->C<IR::IROp_AtomicFetchSub>(), Kind, IROp->Size, FE, Dest);
        break;
      case IR::IROps::OP_ATOMICFETCHAND:
        ok = EmitAtomicFetch(IR, IROp->C<IR::IROp_AtomicFetchAnd>(), Kind, IROp->Size, FE, Dest);
        break;
      case IR::IROps::OP_ATOMICFETCHOR:
        ok = EmitAtomicFetch(IR, IROp->C<IR::IROp_AtomicFetchOr>(), Kind, IROp->Size, FE, Dest);
        break;
      default: // OP_ATOMICFETCHXOR
        ok = EmitAtomicFetch(IR, IROp->C<IR::IROp_AtomicFetchXor>(), Kind, IROp->Size, FE, Dest);
        break;
      }
      if (!ok) {
        return false;
      }
      break;
    }
    case IR::IROps::OP_CAS: {
      // P3.1: lock cmpxchg. IR `CAS`'s Dest is the pre-op memory value in
      // both the success and failure cases (IR.json's contract), which is
      // exactly what wasm cmpxchg returns; the frontend derives ZF/CF from
      // it with a plain SubWithFlags.
      if (!EmitCAS(IR, IROp->C<IR::IROp_CAS>(), IROp->Size, FE, Dest)) {
        return false;
      }
      break;
    }
    case IR::IROps::OP_FENCE: {
      // P3.1: mfence/lfence/sfence lower to nothing. The generated module
      // runs on one thread against one linear memory — there is no other
      // agent to order against, and the atomic RMW ops the fence would
      // guard are already sequentially consistent. FenceType::Inst is the
      // same story: wasm module compilation is atomic per block function.
      break;
    }
    case IR::IROps::OP_PUSHTWO: {
      // Post-RA fusion of two consecutive pushes (Arm64: stp pre-index).
      // new = Addr - 2*ValueSize; [new] = Value1; [new + ValueSize] = Value2;
      // the Addr register is updated in place. The fused op has no result
      // node, so the write-back goes to the Addr arg's location directly.
      const auto* Op = IROp->C<IR::IROp_PushTwo>();
      const uint32_t ValueSize = IR::OpSizeToSize(Op->ValueSize);
      if (ValueSize != 4) {
        LogMan::Msg::EFmt("WasmBackend: PushTwo with unsupported ValueSize {}", ValueSize);
        return false;
      }
      uint32_t Value1, Value2;
      if (!ResolveArg(IR, Op->Value1, FE, &Value1) || !ResolveArg(IR, Op->Value2, FE, &Value2)) {
        return false;
      }
      if (!Op->Addr.IsImmediate()) {
        LogMan::Msg::EFmt("WasmBackend: PushTwo Addr must be a static register");
        return false;
      }
      const IR::PhysicalRegister PhyReg(Op->Addr);
      if (PhyReg.AsRegClass() != IR::RegClass::GPRFixed || PhyReg.Reg >= NUM_STATIC_SLOTS) {
        LogMan::Msg::EFmt("WasmBackend: PushTwo Addr register {} unsupported", PhyReg.Reg);
        return false;
      }
      const uint32_t AddrLocal = FE.StaticLocal(PhyReg.Reg);
      PushLocalGet(FE.Body, AddrLocal);
      PushU32Const(FE.Body, 2 * ValueSize);
      FE.Body.push_back(0x6B); // i32.sub
      const uint32_t NewAddr = FE.AllocScratch();
      PushLocalSet(FE.Body, NewAddr);
      PushLocalGet(FE.Body, NewAddr);
      PushLocalGet(FE.Body, Value1);
      if (!PushGPRStore(FE.Body, ValueSize, 0)) {
        return false;
      }
      PushLocalGet(FE.Body, NewAddr);
      PushU32Const(FE.Body, ValueSize);
      FE.Body.push_back(0x6A); // i32.add
      PushLocalGet(FE.Body, Value2);
      if (!PushGPRStore(FE.Body, ValueSize, 0)) {
        return false;
      }
      PushLocalGet(FE.Body, NewAddr);
      PushLocalSet(FE.Body, AddrLocal);
      break;
    }
    case IR::IROps::OP_EXITFUNCTION: {
      // A block's terminal: write the epilogue (context stores + RIP) and
      // return from the wasm function. Mid-block exits are rejected up front
      // (the terminal-op check).
      uint64_t InlineRIP = 0;
      if (!EmitExit(IR, EntryRIP, IROp->C<IR::IROp_ExitFunction>()->NewRIP, FE, ContextBase, &InlineRIP)) {
        return false;
      }
      FE.Body.push_back(0x0F); // return
      // The dispatcher recognizes IP-in-thunk-window at block lookup (doc 04
      // §3.2); when this block exits into the window its module also carries
      // the registered host imports (extra-imports hook).
      AnyExitTargetsThunk |= ThunkSize != 0 && InlineRIP >= ThunkBase && InlineRIP < ThunkBase + ThunkSize;
      if (IsLast) {
        return true;
      }
      LogMan::Msg::EFmt("WasmBackend: ExitFunction is not the last op of its block");
      return false;
    }
    case IR::IROps::OP_CONDJUMP: {
      // Terminal: <cond> -> br_if $false ; <TrueBlock>; end ; <FalseBlock>.
      // The condition is evaluated inside the false-path block so its value
      // is on that frame's stack for br_if.
      const auto* Op = IROp->C<IR::IROp_CondJump>();
      if (!IsLast) {
        LogMan::Msg::EFmt("WasmBackend: CondJump is not the last op of its block");
        return false;
      }
      uint32_t TrueIdx = 0, FalseIdx = 0;
      if (!ResolveBlockIndex(IR, Op->TrueBlock, &TrueIdx) || !ResolveBlockIndex(IR, Op->FalseBlock, &FalseIdx)) {
        return false;
      }
      // The condition evaluates inside the false-path block (br_if consumes
      // it from that frame's stack); br_if branches when NOT taken.
      FE.Body.push_back(0x02); // block
      PushULEB(FE.Body, 0x40); // void type
      if (Op->FromNZCV) {
        bool ok = true;
        EmitNZCVCondition(FE.Body, FE.NZCV(), FE.CondScratch(), Op->Cond, &ok);
        if (!ok) {
          return false;
        }
      } else {
        if (Op->CompareSize != IR::OpSize::i32Bit && Op->CompareSize != IR::OpSize::iInvalid) {
          LogMan::Msg::EFmt("WasmBackend: CondJump with unsupported compare size");
          return false;
        }
        uint32_t C1, C2;
        if (!ResolveArg(IR, Op->Cmp1, FE, &C1) || !ResolveArg(IR, Op->Cmp2, FE, &C2)) {
          return false;
        }
        if (!EmitValueCondition(FE.Body, Op->Cond, C1, C2)) {
          return false;
        }
      }
      FE.Body.push_back(0x45); // i32.eqz
      FE.Body.push_back(0x0D); // br_if
      PushULEB(FE.Body, 0);
      if (!EmitCodeBlock(IR, EntryRIP, ContextBase, ThunkBase, ThunkSize, FE, TrueIdx)) {
        return false;
      }
      FE.Body.push_back(0x0B); // end of the false-path block
      if (!EmitCodeBlock(IR, EntryRIP, ContextBase, ThunkBase, ThunkSize, FE, FalseIdx)) {
        return false;
      }
      return true;
    }
    case IR::IROps::OP_VLOADVECTORELEMENT: {
      // Named loud-fail (the honest-failure contract): outside the emitter
      // subset. In practice only the m80 x87 load shape rides it
      // (fld tbyte: LoadMem -> VLoadVectorElement -> F80CVT -> PushStack
      // LoadSize=10), which the named PushStack check would otherwise
      // shadow — the vector op is refused first.
      LogMan::Msg::EFmt("WasmBackend: VLoadVectorElement is outside the emitter subset (the m80 load shape rides it; no SoftFloat path)");
      return false;
    }
    case IR::IROps::OP_ATOMICFETCHNEG: {
      // Named loud-fail: lock neg has no wasm RMW form (no neg in the
      // atomic surface — the LOCK class covers add/sub/and/or/xor/xchg/
      // cmpxchg/btr only).
      LogMan::Msg::EFmt("WasmBackend: AtomicFetchNeg has no wasm RMW form (no neg in the atomic surface)");
      return false;
    }
    case IR::IROps::OP_JUMP: {
      LogMan::Msg::EFmt("WasmBackend: Jump lowering unsupported (no relooper)");
      return false;
    }
    default:
      LogMan::Msg::EFmt("WasmBackend: unsupported IR op {} ({})", static_cast<uint32_t>(IROp->Op), IR::GetName(IROp->Op));
      return false;
    }

    RecordDef(FE, IR, CodeNode, NodeID);

    // Nodes annotated to a static location write it (this is how the mov into
    // the PF/AF pseudo-regs materializes — the value simply lives there).
    // Ops that already wrote a static location themselves are skipped: their
    // own RA annotation is that destination, not a new value.
    if (!WroteStaticThisNode) {
      NoteStaticDef(FE, IR::PhysicalRegister(CodeNode), Dest);
    }
    WroteStaticThisNode = false;
  }

  LogMan::Msg::EFmt("WasmBackend: IR block {} ended without a terminal op", BlockIdx);
  return false;
}

bool ModuleBuilder::Emit(const IR::IRListView* IR, uint64_t EntryRIP, uint64_t ContextBase, uint64_t ThunkBase, uint64_t ThunkSize,
                         const fextl::vector<WasmHostImport>& HostImports) {
  // Collect the IRList's code blocks so branch targets resolve. The frontend
  // creates both CondJump targets when multiblock is off, so a jcc-ending
  // guest block lifts to three blocks.
  BlockNodes.clear();
  for (auto [Node, Header] : IR->GetBlocks()) {
    BlockNodes.push_back(Node);
  }

  // Sparse node IDs post-RA (push-fusion allocates late IDs) and scratch
  // locals (one per Pop/PushTwo, two per PopTwo) are sized across ALL blocks.
  // Div/UDiv need i64 scratch for the 64-bit quotient/remainder; the float
  // compares need boolean scratch (P2.12).
  //
  // Node locals are typed: FPR-annotated nodes hold v128 (the xmm file);
  // GPR nodes hold i32 except the 64-bit integer producers (fild m64's
  // load/extend chain), which hold i64.
  using LType = FunctionEmitter::LType;
  fextl::vector<LType> NodeTypes;
  uint32_t ScratchCount = 0;
  uint32_t MaxNodeID = 0;
  I64ScratchCount = 0;
  I64HiByLocal.clear();
  // Post-RA def tracking for the VCastFromGPR-64 constant typing: the
  // dynamic register a node's annotation assigns, and the constants by ID.
  uint32_t RegDefNode[NUM_DYNAMIC_REGS] = {};
  std::unordered_map<uint32_t, bool> ConstantByID;
  for (auto* BlockNode : BlockNodes) {
    for (auto [CodeNode, IROp] : IR->GetCode(BlockNode)) {
      const auto Op = IROp->Op;
      const uint32_t ID = IR->GetID(CodeNode).Value;
      MaxNodeID = std::max(MaxNodeID, ID);
      if (NodeTypes.size() <= ID) {
        NodeTypes.resize(ID + 1, LType::I32);
      }
      const auto PhyReg = IR::PhysicalRegister(CodeNode);
      if ((PhyReg.AsRegClass() == IR::RegClass::FPR || PhyReg.AsRegClass() == IR::RegClass::FPRFixed) && !PhyReg.IsInvalid()) {
        NodeTypes[ID] = LType::V128;
        UsesFPR = true; // the xmm static slots must sync in prologue/epilogue
      }
      if (!PhyReg.IsInvalid() && PhyReg.AsRegClass() == IR::RegClass::GPR && PhyReg.Reg < NUM_DYNAMIC_REGS) {
        RegDefNode[PhyReg.Reg] = ID;
      }
      if (Op == IR::IROps::OP_CONSTANT) {
        ConstantByID[ID] = true;
      }
      switch (Op) {
      case IR::IROps::OP_VCASTFROMGPR: {
        // FLDF64_Const's bit-pattern constant (an i64 that cannot live in
        // an i32 local): type the defining constant I64 so the const emits
        // as i64.const and the replace_lane type-checks. The source is
        // followed through the post-RA dest annotations (constants ride
        // DynamicLastDef like any other GPR value). NOTE: a constant's
        // i64-ness is decided by its consumers, not its value — a >u32
        // constant consumed by a 32-bit op truncates by design.
        const auto* VC = IROp->C<IR::IROp_VCastFromGPR>();
        if (IR::OpSizeAsBits(VC->Header.ElementSize) == 64) {
          uint32_t SrcNodeID = 0;
          if (!VC->Src.IsImmediate()) {
            SrcNodeID = IR->GetID(IR->GetNode(VC->Src)).Value;
          } else {
            const IR::PhysicalRegister SrcReg(VC->Src);
            if (SrcReg.AsRegClass() == IR::RegClass::GPR && SrcReg.Reg < NUM_DYNAMIC_REGS && RegDefNode[SrcReg.Reg] != 0) {
              SrcNodeID = RegDefNode[SrcReg.Reg];
            }
          }
          if (SrcNodeID != 0 && ConstantByID.count(SrcNodeID)) {
            NodeTypes[SrcNodeID] = LType::I64;
          }
        }
        // movd/movq write the xmm file: the static slots must sync in
        // prologue/epilogue (the P2.12 UsesFPR classification, folded into
        // this case — it had been a second label in this same switch).
        UsesFPR = true;
        break;
      }
      case IR::IROps::OP_POP:
      case IR::IROps::OP_PUSHTWO:
        ++ScratchCount;
        break;
      case IR::IROps::OP_POPTWO:
        ScratchCount += 2;
        break;
      case IR::IROps::OP_DIV:
      case IR::IROps::OP_UDIV:
        ScratchCount += 2;     // quotient/remainder routing scratch
        I64ScratchCount += 3;  // divisor, quotient, remainder i64 temps
        break;
      case IR::IROps::OP_SUBNZCV:
      case IR::IROps::OP_SHIFTFLAGS:
      case IR::IROps::OP_CONDSUBNZCV:
        ScratchCount += 2; // the ccmp then-leg temp and the masked shift count
        break;
      case IR::IROps::OP_UMUL:
      case IR::IROps::OP_SMULL:
        ++ScratchCount; // the 64-bit product's high half
        break;
      case IR::IROps::OP_MEMCPY:
        // The direction-aware element loop: remaining count, original count,
        // the per-iteration offset, and the total bytes for result routing —
        // plus one scratch per dynamic result register (the final
        // destination/source addresses route through them) and one for a
        // materialized inline-constant direction (the F3d finding: a budget
        // of 4 spilled the results into the v128 FPR locals).
        ScratchCount += 7;
        break;
      case IR::IROps::OP_MEMSET:
        // The same loop plus the masked element value: five i32 temps, the
        // final address routes through the node's own local, and a
        // materialized inline-constant direction is the sixth.
        ScratchCount += 6;
        break;
      case IR::IROps::OP_F80CMPVALUE:
      case IR::IROps::OP_F80CMPSTACK:
        ScratchCount += 3; // LT/EQ/UNORDERED booleans
        UsesFPR = true;
        break;
      case IR::IROps::OP_FLOAT_TOGPR_ZS:
      case IR::IROps::OP_FLOAT_TOGPR_S:
      case IR::IROps::OP_VECTOR_FTOZS:
        ScratchCount += 1; // the in-range condition, consumed by two selects
        UsesFPR = true;
        break;
      case IR::IROps::OP_AXFLAG:
        ScratchCount += 2; // new-C/new-Z
        break;
      case IR::IROps::OP_PARITY:
        ++ScratchCount; // the xor-fold temp
        break;
      case IR::IROps::OP_FCMP:
        ScratchCount += 3; // LT/EQ/UNORDERED booleans
        UsesFPR = true;
        break;
      case IR::IROps::OP_LOADMEM:
      case IR::IROps::OP_LOADMEMTSO:
      case IR::IROps::OP_STOREMEM:
      case IR::IROps::OP_STOREMEMTSO:
      case IR::IROps::OP_SBFE:
      case IR::IROps::OP_VEXTRACTTOGPR: {
        // Per-op type/flag classification (the ternaries must stay within
        // one struct type: layout-compatible pairs only).
        if (Op == IR::IROps::OP_LOADMEM || Op == IR::IROps::OP_LOADMEMTSO) {
          if (Op == IR::IROps::OP_LOADMEM) {
            const auto* LM = IROp->C<IR::IROp_LoadMem>();
            if (LM->Class == IR::RegClass::FPR) {
              UsesFPR = true;
            } else if (LM->Class == IR::RegClass::GPR && IR::OpSizeToSize(IROp->Size) == 8) {
              NodeTypes[ID] = LType::I64; // fild m64's 64-bit integer load
            }
          } else {
            const auto* LM = IROp->C<IR::IROp_LoadMemTSO>();
            if (LM->Class == IR::RegClass::FPR) {
              UsesFPR = true;
            } else if (LM->Class == IR::RegClass::GPR && IR::OpSizeToSize(IROp->Size) == 8) {
              NodeTypes[ID] = LType::I64;
            }
          }
        } else if (Op == IR::IROps::OP_STOREMEM || Op == IR::IROps::OP_STOREMEMTSO) {
          if (Op == IR::IROps::OP_STOREMEM) {
            if (IROp->C<IR::IROp_StoreMem>()->Class == IR::RegClass::FPR) {
              UsesFPR = true;
            }
          } else {
            if (IROp->C<IR::IROp_StoreMemTSO>()->Class == IR::RegClass::FPR) {
              UsesFPR = true;
            }
          }
        } else if (Op == IR::IROps::OP_SBFE) {
          const auto* S = IROp->C<IR::IROp_Sbfe>();
          if (S->Width == 64 && S->lsb == 0) {
            NodeTypes[ID] = LType::I64; // fild m16's sign-extend to 64
          }
        } else { // OP_VEXTRACTTOGPR
          if (IR::OpSizeAsBits(IROp->ElementSize) == 64) {
            NodeTypes[ID] = LType::I64; // movq xmm -> r64 bitcast
          }
          UsesFPR = true;
        }
        break;
      }
      case IR::IROps::OP_F80ADDSTACK:
      case IR::IROps::OP_F80SUBSTACK:
      case IR::IROps::OP_F80MULSTACK:
      case IR::IROps::OP_F80DIVSTACK:
      case IR::IROps::OP_F80ADDVALUE:
      case IR::IROps::OP_F80SUBVALUE:
      case IR::IROps::OP_F80MULVALUE:
      case IR::IROps::OP_F80DIVVALUE:
      case IR::IROps::OP_F80SUBRVALUE:
      case IR::IROps::OP_F80DIVRVALUE:
        I64ScratchCount += 2; // the operand bit patterns for the IE-flag check
        [[fallthrough]];
      case IR::IROps::OP_PUSHSTACK:
      case IR::IROps::OP_COPYPUSHSTACK:
      case IR::IROps::OP_READSTACKVALUE:
      case IR::IROps::OP_STORESTACKMEM:
      case IR::IROps::OP_FLOAT_FTOF:
      case IR::IROps::OP_FLOAT_FROMGPR_S:
      case IR::IROps::OP_VINSELEMENT:
      case IR::IROps::OP_VFADD:
      case IR::IROps::OP_VFSUB:
      case IR::IROps::OP_VFMUL:
      case IR::IROps::OP_VFDIV:
      case IR::IROps::OP_VFADDSCALARINSERT:
      case IR::IROps::OP_VFSUBSCALARINSERT:
      case IR::IROps::OP_VFMULSCALARINSERT:
      case IR::IROps::OP_VFDIVSCALARINSERT:
      case IR::IROps::OP_VFSQRTSCALARINSERT:
      case IR::IROps::OP_VFMINSCALARINSERT:
      case IR::IROps::OP_VFMAXSCALARINSERT:
        UsesFPR = true; // the xmm static slots must sync in prologue/epilogue
        break;
      default:
        break;
      }
    }
  }

  FunctionEmitter FE(MaxNodeID + 1, NodeTypes, ScratchCount, I64ScratchCount);
  EmitPrologue(FE, ContextBase);

  VisitedBlocks.assign(BlockNodes.size(), false);
  AnyExitTargetsThunk = false;
  if (!EmitCodeBlock(IR, EntryRIP, ContextBase, ThunkBase, ThunkSize, FE, 0)) {
    return false;
  }
  // Every block must have been reached; an unreachable block means control
  // flow the structured lowering cannot express (e.g. shared targets).
  for (uint32_t I = 0; I < BlockNodes.size(); ++I) {
    if (!VisitedBlocks[I]) {
      LogMan::Msg::EFmt("WasmBackend: IR block {} is unreachable from the entry block", I);
      return false;
    }
  }

  // Assemble the module:
  //   (module
  //     (import "env" "memory" (memory 0))
  //     (import "<mod>" "<name>" (func $i (param i32...) (result i32)))  ; if the
  //        block exits into the thunk window (extra-imports hook)
  //     (func (export "block") ...locals... )
  //   )
  auto Section = [&Module = Module](uint8_t ID, fextl::vector<uint8_t> Payload) {
    Module.push_back(ID);
    PushULEB(Module, static_cast<uint32_t>(Payload.size()));
    Module.insert(Module.end(), Payload.begin(), Payload.end());
  };

  // Host imports are only injected into modules whose exit lands in the thunk
  // window; every other block keeps the D2 import surface (env.memory only).
  const fextl::vector<WasmHostImport> NoImports;
  const fextl::vector<WasmHostImport>& Imports = AnyExitTargetsThunk ? HostImports : NoImports;
  const uint32_t BlockFuncIndex = static_cast<uint32_t>(Imports.size());

  PushVec(Module, {0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00}); // magic + version

  {
    fextl::vector<uint8_t> P;
    PushULEB(P, 1 + static_cast<uint32_t>(Imports.size())); // types
    PushVec(P, {0x60, 0x00, 0x00});                         // type 0: func () -> ()
    for (const auto& Import : Imports) {
      PushVec(P, {0x60}); // func (param i32 x N) -> (i32)
      PushULEB(P, Import.ParamCount);
      for (uint32_t I = 0; I < Import.ParamCount; ++I) {
        P.push_back(0x7F);
      }
      PushVec(P, {0x01, 0x7F});
    }
    Section(1, std::move(P));
  }
  {
    fextl::vector<uint8_t> P;
    PushULEB(P, 1 + static_cast<uint32_t>(Imports.size())); // imports
    PushULEB(P, 3);
    PushVec(P, {'e', 'n', 'v'});
    PushULEB(P, 6);
    PushVec(P, {'m', 'e', 'm', 'o', 'r', 'y'});
    PushVec(P, {0x02, 0x00, 0x00}); // memory desc: kind, limits flags=0, min=0
    for (uint32_t I = 0; I < Imports.size(); ++I) {
      const auto& Import = Imports[I];
      const auto ModLen = static_cast<uint32_t>(strlen(Import.Module));
      const auto NameLen = static_cast<uint32_t>(strlen(Import.Name));
      PushULEB(P, ModLen);
      P.insert(P.end(), Import.Module, Import.Module + ModLen);
      PushULEB(P, NameLen);
      P.insert(P.end(), Import.Name, Import.Name + NameLen);
      PushVec(P, {0x00});     // import kind: func
      PushULEB(P, 1 + I);     // type index (type 0 is the block's () -> ())
    }
    Section(2, std::move(P));
  }
  {
    fextl::vector<uint8_t> P;
    PushVec(P, {0x01, 0x00}); // 1 function, type 0
    Section(3, std::move(P));
  }
  {
    fextl::vector<uint8_t> P;
    PushVec(P, {0x01}); // 1 export
    PushULEB(P, 5);
    PushVec(P, {'b', 'l', 'o', 'c', 'k'});
    PushVec(P, {0x00}); // func kind
    PushULEB(P, BlockFuncIndex);
    Section(7, std::move(P));
  }
  {
    fextl::vector<uint8_t> Body;
    // Three local groups: i32 (state/scratch/GPR nodes), i64 (64-bit integer
    // nodes + division temps), v128 (FPR nodes + the xmm static slots).
    PushVec(Body, {0x03});
    PushULEB(Body, FE.I32Locals());
    Body.push_back(0x7F); // i32
    PushULEB(Body, FE.I64Locals());
    Body.push_back(0x7E); // i64
    PushULEB(Body, FE.V128Locals());
    Body.push_back(0x7B); // v128
    Body.insert(Body.end(), FE.Body.begin(), FE.Body.end());
    Body.push_back(0x0B); // end

    fextl::vector<uint8_t> P;
    PushVec(P, {0x01}); // 1 function body
    PushULEB(P, static_cast<uint32_t>(Body.size()));
    P.insert(P.end(), Body.begin(), Body.end());
    Section(10, std::move(P));
  }
  return true;
}

} // namespace

WasmBackend::WasmBackend(Context::ContextImpl*, Core::InternalThreadState* Thread)
  : CPUBackend(*static_cast<Context::ContextImpl*>(Thread->CTX), Thread) {
  // Report a generous virtual register file to the RA pass (the Arm64 JIT
  // ctor does the same). Plenty of registers means the RA pass never inserts
  // spill/fill ops, so the IR keeps the exact shape the frontend produced.
  auto* RAPass = Thread->PassManager->GetPass<IR::RegisterAllocationPass>("RA");
  RAPass->AddRegisters(IR::RegClass::GPR, 31);
  RAPass->AddRegisters(IR::RegClass::GPRFixed, NUM_STATIC_SLOTS);
  RAPass->AddRegisters(IR::RegClass::FPR, 31);
  RAPass->AddRegisters(IR::RegClass::FPRFixed, 16);
  RAPass->SetNumPairRegs(0);
}

CPUBackend::CompiledCode WasmBackend::CompileCode(uint64_t Entry, uint64_t, bool, const IR::IRListView* IR,
                                                  FEXCore::Core::DebugData*, bool) {
  CompiledCode Code {};

  ModuleBuilder Builder;
  if (!Builder.Emit(IR, Entry, ContextBase, ThunkWindowBase, ThunkWindowSize, HostImports)) {
    return Code;
  }

  ModuleBytes = Builder.Bytes();
  LastEntryRIP = Entry;

  // BlockBegin points at the wasm module bytes, *not* executable host code —
  // the D2 two-phase pipeline hands these bytes to the host runtime to
  // instantiate (docs/spikes/p1-d2-wasm-backend.md).
  Code.BlockBegin = const_cast<uint8_t*>(ModuleBytes.data());
  Code.Size = ModuleBytes.size();
  Code.EntryPoints = {{Entry, Code.BlockBegin}};
  Code.HostCodeOffset = 0;
  return Code;
}

fextl::unique_ptr<CPUBackend> CreateWasmBackend(Context::ContextImpl* ctx, Core::InternalThreadState* Thread) {
  return fextl::make_unique<WasmBackend>(ctx, Thread);
}

} // namespace FEXCore::CPU
