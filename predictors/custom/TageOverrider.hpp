#pragma once
// TageOverrider — pluggable prediction override interface for TAGE.
//
// After TAGE computes its P2 prediction, an optional overrider can examine
// the prediction + confidence signals and decide to flip it. The overrider
// runs its own table lookup in PARALLEL with TAGE's RAM reads, then makes
// its override decision after TAGE signals are available.
//
// When Overrider = NoOverrider (default), all override code compiles away
// via if constexpr (Overrider::ENABLED). Zero cost — bit-exact with
// non-overrider code.
//
// === Overrider Concept ===
//
// An overrider type must provide:
//
//   static constexpr bool ENABLED = true;
//
//   // Phase 1: Parallel lookup (called at start of predict2, concurrent with TAGE reads)
//   // Overrider reads its own tables here.
//   void lookup(val<64> &inst_pc, reg<BINDEX_BITS> &bindex);
//
//   // Phase 2: Override decision (called after TAGE computes confidence + p2)
//   // Sets internal ovr_valid/ovr_pred registers.
//   void override_predict(arr<val<2>, FETCH_WIDTH> &tage_confidence,
//                          reg<FETCH_WIDTH> &p2, val<64> &inst_pc);
//
//   // Per-offset accessors for the override decision
//   val<1> get_override_valid(u64 offset);  // should override TAGE?
//   val<1> get_override_pred(u64 offset);   // overrider's prediction
//
//   // Signal to TageImpl: skip TAGE allocation when loop is confident provider
//   val<1> get_skip_tage_alloc(u64 offset);
//
//   // Phase 3: Training (called in update_cycle with outcome signals)
//   void train(arr<val<1>, FETCH_WIDTH> &branch_taken,
//              arr<val<1>, FETCH_WIDTH> &is_branch,
//              val<1> &mispredict,
//              val<1> &correct_pred,
//              arr<reg<1>, FETCH_WIDTH> &override_active,
//              reg<FETCH_WIDTH> &p2_before_override,
//              val<IDX_BITS> tage_idx,   // temporary return val — by value OK
//              reg<BINDEX_BITS> &bindex);
//
// === CRITICAL: Pass HARCOM types by REFERENCE ===
//
// All val<>, reg<>, arr<> arguments MUST be passed by reference (&), never by value:
//   - reg<N> by value CRASHES (copy creates/destroys HARCOM storage)
//   - val<N> by value adds ~50% latency overhead (FO2 inverter from copy)
//   - arr<T,N> by value copies all elements (multiplied overhead)
//
// Exception: temporary return values (e.g. from tidx<I>()) are fine by value.
// HARCOM get() is not const-qualified, so use non-const & (not const &).

struct NoOverrider {
  static constexpr bool ENABLED = false;
  // No methods needed — all call sites are behind if constexpr (Overrider::ENABLED)
};
