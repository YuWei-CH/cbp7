# TageOverrider — Pluggable Prediction Override

## Overview

The TAGE predictor supports an optional overrider component that can override P2 predictions. The overrider runs its own table lookup **in parallel** with TAGE's RAM reads, then makes a final override decision after TAGE signals are available. When no overrider is used (`NoOverrider`), all override code compiles away — zero cost, bit-exact with non-overrider code.

## Quick Start

```cpp
// Default: no override
using MyPredictor = Tage<>;

// With loop predictor (last template parameter):
using MyPredictor = Tage<DefaultTableConfig, DefaultAllocConfig,
    16, 4096, 1, 1, false, true, true, true, false, 0, 0,
    DefaultResetFn, false, true, 16384, 6, true, 4, 2, 0, false, 27, 6,
    LoopPredictor<64>>;
```

## Architecture

```
predict2 timeline:

  t=0  ─── PARALLEL ───────────────────────────
  │                                            │
  │  TAGE: compute indexes                     │  Overrider: lookup(pc, bindex)
  │  TAGE: read tag/pred/hyst/u RAMs           │    - read own tables
  │  TAGE: tag compare → match1, match2        │    - unpack, tag match
  │  TAGE: meta-prediction → tage_p2           │    - compute prediction
  │  TAGE: compute tage_confidence             │    - store results in regs
  │                                            │
  t≈560ps ─────────────────────────────────────
  │
  │  override_predict(tage_confidence, tage_p2, pc)
  │    - ONE select mux: gate on tage_confidence (~20ps)
  │
  │  TageImpl mux: p2 = select(ovr_valid, ovr_pred, tage_p2)
  │
  t≈580ps ── return prediction
```

## Writing a New Overrider

Create a struct with these members and methods:

```cpp
#include "TageOverrider.hpp"

template <u64 TABLE_SIZE = 64, /* your params */>
struct MyOverrider {
    static constexpr bool ENABLED = true;

    // --- Storage (HARCOM types) ---
    // Your RAMs, regs, etc.

    // --- Phase 1: lookup() ---
    // Called at START of predict2, parallel with TAGE reads.
    // Do ALL heavy work here: RAM reads, unpack, compare, prediction.
    // Store results in regs for override_predict() to read.
    void lookup(val<64> &inst_pc, reg<BINDEX_BITS> &bindex);

    // --- Phase 2: override_predict() ---
    // Called AFTER TAGE computes confidence + prediction.
    // Should be MINIMAL: one select/AND gate using tage_confidence
    // to decide whether to override. Heavy computation belongs in lookup().
    void override_predict(arr<val<2>, FETCH_WIDTH> &tage_confidence,
                           val<FETCH_WIDTH> &tage_p2,
                           val<64> &inst_pc);

    // --- Accessors (called by TageImpl to build the final mux) ---
    val<1> get_override_valid(u64 offset);   // should override at this offset?
    val<1> get_override_pred(u64 offset);    // overrider's prediction
    val<1> get_skip_tage_alloc(u64 offset);  // skip TAGE allocation?

    // --- Phase 3: train() ---
    // Called in update_cycle with outcome information.
    void train(arr<val<1>, FETCH_WIDTH> &branch_taken,
               arr<val<1>, FETCH_WIDTH> &is_branch,
               val<1> &mispredict,
               val<1> &correct_pred,
               arr<reg<1>, FETCH_WIDTH> &override_active,
               reg<FETCH_WIDTH> &p2_before_override);
};
```

## Critical Rules

### 1. Pass HARCOM types by reference, never by value
```cpp
void lookup(val<64> &pc, reg<8> &idx);     // ✓ reference
void lookup(val<64> pc, reg<8> idx);        // ✗ val copies add latency, reg copies CRASH
```

### 2. The overrider must NOT write to TageImpl's regs
The overrider returns `val<1>` from accessors. TageImpl owns the final `p2` mux:
```cpp
// TageImpl does this — NOT the overrider:
p2 = select(overrider.get_override_valid(offset),
            overrider.get_override_pred(offset),
            tage_p2_split[offset]);
```

### 3. Keep override_predict() minimal
All heavy computation (RAM reads, tag compare, prediction logic) belongs in `lookup()` which runs in parallel with TAGE. `override_predict()` should only gate on `tage_confidence` — typically one AND/select:
```cpp
void override_predict(...) {
    val<1> tage_weak = (tage_confidence[0] < hard<3>{});
    ovr_valid[0] = val<1>{precomputed_candidate} & tage_weak;  // ~20ps
}
```

### 4. Avoid reg write→read in the same function
Writing to a reg then reading it back in the same cycle adds latency:
```cpp
reg<8> temp;
temp = some_val;            // write
val<8> x = val<8>{temp};    // read back — adds reg write delay!
```
Use local vals for intermediate computation, regs only for final results.

### 5. Use rwram for tables that need same-cycle read+write
`ram<T,N>` enforces one access per cycle. If `lookup()` reads and `train()` writes in the same HARCOM cycle:
```cpp
rwram<ENTRY_BITS, NUM_SETS, 2> my_table{"mytable"};  // 2-bank rwram
// write signature: my_table.write(addr, data, noconflict_signal);
```

### 6. Use static_loop for compile-time iteration
When indexing RAMs or using `hard<w>{}` with a way/bank index:
```cpp
static_loop<ASSOC>([&]<u64 w>() {
    loop_ram[w].read(...);           // ✓ w is compile-time
    execute_if(hit_way == hard<w>{}, ...);  // ✓
});
```

### 7. val<> cannot be reassigned
```cpp
val<1> x = val<1>{0};
x = select(cond, val<1>{1}, x);  // ✗ move assignment is private

// Use arrays + fold instead:
arr<val<1>, N> results = [&](u64 i) { return ...; };
val<1> x = results.fo1().fold_or();  // ✓
```

## TAGE Confidence Encoding

The `tage_confidence` array (per offset, 2 bits) encodes:

| Value | Meaning |
|-------|---------|
| 0 | Newly allocated entry (weak, cold) |
| 1 | Primary ≠ alt prediction (disagreement) |
| 2 | Primary == alt, normal agreement |
| 3 | Primary == alt, bimodal provider (highest confidence) |

## Files

- `predictors/custom/TageOverrider.hpp` — `NoOverrider` + concept docs
- `predictors/custom/LoopPredictor.hpp` — Example: loop predictor implementation
- `predictors/custom/Tage.hpp` — Scaffolding call sites (search for `Overrider::ENABLED`)
