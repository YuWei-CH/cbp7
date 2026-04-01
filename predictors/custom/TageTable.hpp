#pragma once

#include "../../harcom.hpp"
#include "../common.hpp"

using namespace hcm;

// Constexpr helper (kept for backward compatibility with Tage.hpp)
constexpr size_t clog2(std::uint64_t n) {
  int r = 0;
  std::uint64_t v = 1;
  while (v < n) {
    v <<= 1;
    ++r;
  }
  return r;
}

// Empty type for conditional member elimination
struct EmptyMember {
  EmptyMember() = default;
  EmptyMember(const char *) {}
};

// Default u-bit reset functor for FF mode
// Mode 0: reset to zero
// Mode 1: right shift by 1
// Mode 2: saturating decrement by 1
struct DefaultResetFn {
  static constexpr u64 MODE_BITS = 2;

  template <u64 W> static val<W> apply(val<W> u, val<MODE_BITS> mode) {
    auto shifted = u >> hard<1>{};
    auto decremented =
        select(u == hard<0>{}, val<W>{0}, val<W>{u - 1});
    return select(mode == hard<0>{}, val<W>{0},
                  select(mode == hard<1>{}, shifted, decremented));
  }
};

// ============================================================================
// TageTable: parameterized TAGE tagged table — storage-only container
//
// One instance per table in the TAGE hierarchy. The predictor (TageImpl)
// handles all read/write logic inline for zero val-copy overhead.
//
// Storage is split into 4 separate RAMs per table:
//   tag_ram  — plain ram (tag bits)
//   pred_ram — plain ram (prediction counter)
//   hyst_ram — rwram (hysteresis, supports same-cycle read+write via banking)
//   u_ram    — rwram (usefulness bits) or u_ff (flip-flops)
//
// CTR_WIDTH and HYST_WIDTH are independent parameters:
//   CTR_WIDTH controls the prediction counter width (MSB = direction)
//   HYST_WIDTH controls the hysteresis counter width (separate RAM)
//
// All RAMs and helpers are public — the predictor accesses them directly.
// ============================================================================

template <
    u64 TABLE_SIZE = 1024,   // entries per bank (power of 2, >= 2)
    u64 TABLE_HIST = 64,     // history length (not used by table; predictor
                             //   uses for fold computation)
    u64 TAG_WIDTH = 11,      // tag width in bits
    u64 CTR_WIDTH = 1,       // prediction counter width (stored in pred_ram)
    u64 HYST_WIDTH = 2,      // hysteresis counter width (stored in hyst_ram, rwram)
    u64 U_WIDTH = 1,         // useful counter width
    u64 N = 4,               // max branches per cycle = predictions per block
    u64 NUM_BANKS = 1,       // branch-slot banks; BPB = N / NUM_BANKS
    bool USE_AHEAD = false,  // 1-ahead pipelining; doubles physical banks
    bool SHARED_TAG = true,  // share one tag across branch-slot banks
    bool SHARED_U = true,    // share one u-bit across branch-slot banks
    bool SHARED_HYS = true,  // share one hysteresis across BPB slots in a bank
    bool U_STOR_FF = true,   // true = u-bits in flip-flops, false = u-bits in SRAM (rwram)
    u64 DECAY_CTR = 0,       // probabilistic decay LFSR width (0=disabled, use epoch reset)
    typename ResetFn = DefaultResetFn, // u-bit reset functor (U_STOR_FF=true only)
    bool USE_FF_CACHE = false // cache SRAM reads in FFs for block reuse
    >
class TageTable {

public:
  // ======== Exported Constants ========

  static constexpr u64 bpb = N / NUM_BANKS;

  static constexpr u64 table_size = TABLE_SIZE;
  static constexpr u64 table_hist = TABLE_HIST;
  static constexpr u64 tag_width = TAG_WIDTH;
  static constexpr u64 ctr_width = CTR_WIDTH;
  static constexpr u64 hyst_width = HYST_WIDTH;
  static constexpr u64 u_width = U_WIDTH;
  static constexpr u64 n_branches = N;
  static constexpr u64 num_banks = NUM_BANKS;
  static constexpr bool use_ahead = USE_AHEAD;
  static constexpr bool shared_tag = SHARED_TAG;
  static constexpr bool shared_u = SHARED_U;
  static constexpr bool shared_hys = SHARED_HYS;
  static constexpr bool u_stor_ff = U_STOR_FF;
  static constexpr bool use_ff_cache = USE_FF_CACHE;

  // ======== Computed Constants ========

  static constexpr u64 BPB = bpb;
  static constexpr u64 AHEAD_FACTOR = USE_AHEAD ? 2 : 1;
  static constexpr u64 PHYS_BANKS = NUM_BANKS * AHEAD_FACTOR;
  static constexpr u64 IDX_BITS = clog2(TABLE_SIZE);

  // Internal banking factor for rwram instances (matches reference)
  static constexpr u64 RWRAM_BANKS = 4;

  // Prediction counter bits per entry (per-bank)
  static constexpr u64 PRED_ENTRY_BITS = BPB * CTR_WIDTH;

  // Hysteresis bits per rwram entry
  static constexpr u64 HYST_ENTRY_BITS = SHARED_HYS ? HYST_WIDTH : BPB * HYST_WIDTH;

  // RAM instance counts (controlled by sharing params)
  static constexpr u64 TAG_RAM_COUNT = SHARED_TAG ? AHEAD_FACTOR : PHYS_BANKS;
  static constexpr u64 HYST_RAM_COUNT = PHYS_BANKS; // always per-bank
  static constexpr u64 U_STORAGE_COUNT = SHARED_U ? AHEAD_FACTOR : PHYS_BANKS;

  // Result register counts (depends on sharing mode)
  static constexpr u64 TAG_REG_COUNT = SHARED_TAG ? 1 : NUM_BANKS;
  static constexpr u64 U_REG_COUNT = SHARED_U ? 1 : NUM_BANKS;
  static constexpr u64 HIT_REG_COUNT = SHARED_TAG ? 1 : NUM_BANKS;

  // ======== Static Constraints ========

  static_assert(TABLE_SIZE >= 2,
                "TABLE_SIZE must be at least 2");
  static_assert(std::has_single_bit(TABLE_SIZE),
                "TABLE_SIZE must be power of 2");
  static_assert(N > 0,
                "Must predict at least one branch");
  static_assert(NUM_BANKS > 0,
                "Must have at least one bank");
  static_assert(N % NUM_BANKS == 0,
                "N must be divisible by NUM_BANKS");
  static_assert(!SHARED_TAG || SHARED_U,
                "Per-bank U (SHARED_U=false) requires per-bank tags "
                "(SHARED_TAG=false)");
  static_assert(!USE_FF_CACHE || BPB > 1,
                "FF caching requires BPB > 1 (multiple branches per bank)");
  static_assert(CTR_WIDTH > 0,
                "Counter width must be positive");
  static_assert(TAG_WIDTH > 0,
                "Tag width must be positive");
  static_assert(U_WIDTH > 0,
                "U-bit width must be positive");
  // rwram requires TABLE_SIZE / RWRAM_BANKS > 1, i.e. TABLE_SIZE >= 2*RWRAM_BANKS
  static_assert(HYST_WIDTH == 0 || TABLE_SIZE >= 2 * RWRAM_BANKS,
                "TABLE_SIZE must be >= 2*RWRAM_BANKS for rwram hysteresis");
  static_assert(U_STOR_FF || TABLE_SIZE >= 2 * RWRAM_BANKS,
                "TABLE_SIZE must be >= 2*RWRAM_BANKS for rwram u-bits");

  // ======== SRAM Storage ========

  // Tag: plain ram
  hcm::ram<val<TAG_WIDTH>, TABLE_SIZE> tag_ram[TAG_RAM_COUNT]{"tag"};

  // Prediction counter: plain ram, always per-bank
  hcm::ram<val<PRED_ENTRY_BITS>, TABLE_SIZE> pred_ram[PHYS_BANKS]{"pred"};

  // Hysteresis: rwram (banked, same-cycle R+W via internal banking)
  // Conditionally eliminated when HYST_WIDTH=0
  std::conditional_t<(HYST_WIDTH > 0),
      rwram<HYST_ENTRY_BITS, TABLE_SIZE, RWRAM_BANKS>,
      EmptyMember> hyst_ram[HYST_RAM_COUNT]{"hyst"};

  // U-bits: either rwram (SRAM) or flip-flops
  std::conditional_t<U_STOR_FF,
      EmptyMember,
      rwram<U_WIDTH, TABLE_SIZE, RWRAM_BANKS>> u_ram[U_STORAGE_COUNT]{"u"};

  // ======== Flip-Flop Storage ========

  std::conditional_t<U_STOR_FF,
      arr<reg<U_WIDTH>, TABLE_SIZE>,
      EmptyMember> u_ff[U_STORAGE_COUNT];

  // ======== Staging / Cache Registers (USE_FF_CACHE=true only) ========

  struct StagingRegs {
    reg<IDX_BITS> idx_reg;
    reg<TAG_WIDTH> tag_reg[TAG_REG_COUNT];
    reg<1> hit_reg[HIT_REG_COUNT];
    reg<PRED_ENTRY_BITS> pred_reg;
    reg<std::max(u64(1), HYST_ENTRY_BITS)> hyst_reg; // min 1 to avoid zero-width reg
    reg<U_WIDTH> u_reg[U_REG_COUNT];
  };

  std::conditional_t<USE_FF_CACHE, StagingRegs, EmptyMember> cache_;

  // ======== Helpers ========

  TageTable() = default;
  ~TageTable() = default;

  // Compute tag RAM index from stage and optionally bank
  u64 tag_ram_idx(u64 stage, [[maybe_unused]] u64 bank = 0) const {
    if constexpr (SHARED_TAG) {
      return stage;
    } else {
      return stage * NUM_BANKS + bank;
    }
  }

  // Compute u storage index from stage and optionally bank
  u64 u_storage_idx(u64 stage, [[maybe_unused]] u64 bank = 0) const {
    if constexpr (SHARED_U) {
      return stage;
    } else {
      return stage * NUM_BANKS + bank;
    }
  }

  // Write a single entry in a u-bit FF array at a dynamic val index.
  void write_u_ff_arr(u64 arr_idx, val<IDX_BITS> &index, val<U_WIDTH> &u) {
    for (u64 i = 0; i < TABLE_SIZE; i++) {
      execute_if(index == val<IDX_BITS>{static_cast<unsigned>(i)},
                 [&]() { u_ff[arr_idx][i] = u; });
    }
  }

  // ======== Reuse Read (FF Cache) ========

  // Read cached results from staging registers into caller's output regs.
  // Only valid when USE_FF_CACHE=true and a prior read was done.
  template <typename OutTag, typename OutPred, typename OutHyst, typename OutU>
  void reuseRead(u64 bank,
                 OutTag &out_tag, OutPred &out_pred,
                 OutHyst &out_hyst, OutU &out_u) {
    static_assert(USE_FF_CACHE, "reuseRead requires USE_FF_CACHE=true");
    if constexpr (SHARED_TAG) {
      out_tag = val<TAG_WIDTH>{cache_.tag_reg[0]};
    } else {
      out_tag = val<TAG_WIDTH>{cache_.tag_reg[bank]};
    }
    out_pred = val<PRED_ENTRY_BITS>{cache_.pred_reg};
    if constexpr (HYST_WIDTH > 0) {
      out_hyst = val<HYST_ENTRY_BITS>{cache_.hyst_reg};
    }
    if constexpr (SHARED_U) {
      out_u = val<U_WIDTH>{cache_.u_reg[0]};
    } else {
      out_u = val<U_WIDTH>{cache_.u_reg[bank]};
    }
  }

  // ======== U-bit Reset ========

  // Reset u-bits in rwram (SRAM mode)
  void reset_u_ram() {
    static_assert(!U_STOR_FF, "reset_u_ram requires U_STOR_FF=false");
    for (u64 i = 0; i < U_STORAGE_COUNT; i++) {
      u_ram[i].reset();
    }
  }

  // Apply ResetFn to all u-bit flip-flops with the given mode (FF mode only).
  void reset_u(val<ResetFn::MODE_BITS> &mode) {
    static_assert(U_STOR_FF, "reset_u requires U_STOR_FF=true");
    for (u64 a = 0; a < U_STORAGE_COUNT; a++) {
      for (u64 i = 0; i < TABLE_SIZE; i++) {
        u_ff[a][i] = ResetFn::template apply<U_WIDTH>(u_ff[a][i], mode);
      }
    }
  }
};
