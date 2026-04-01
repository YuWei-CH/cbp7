#pragma once
// LoopPredictor — detects repeating branch patterns and overrides TAGE at loop exits.
//
// Implements the TageOverrider concept. Plugs into TageImpl via template parameter.
// Design follows TAGE-SC-L literature (Seznec CBP2025, Ros deep dive).
//
// The loop predictor detects branches that are taken N times then not-taken once
// (or vice versa). When confident, it overrides TAGE's prediction at the loop
// exit boundary where TAGE is typically wrong.

#include "../../harcom.hpp"
#include "../common.hpp"
#include "TageOverrider.hpp"

using namespace hcm;

template <u64 TABLE_SIZE = 64,      // total entries
          u64 ASSOC = 4,            // ways per set
          u64 TAG_BITS = 10,        // PC tag width
          u64 ITER_BITS = 10,       // iteration counter width
          u64 CONF_BITS = 3,        // confidence counter width (0..7)
          u64 FETCH_WIDTH = 16,     // from parent TAGE
          u64 CACHE_ENTRIES = 2>    // entry cache size (0=disabled, N=N-entry cache)
struct LoopPredictor {

  static constexpr bool ENABLED = true;

  // ======== Derived constants ========
  static constexpr u64 NUM_SETS = TABLE_SIZE / ASSOC;
  static constexpr u64 SET_BITS = clog2(NUM_SETS);
  static constexpr u64 AGE_BITS = 3;
  static constexpr u64 CONF_THRESHOLD = 0;  // override when conf > 0 AND nb_iter >= 1

  // Entry layout (LSB-first for split<>):
  //   [TAG_BITS] [ITER_BITS] [ITER_BITS] [CONF_BITS] [1 dir] [AGE_BITS]
  static constexpr u64 ENTRY_BITS =
      TAG_BITS + ITER_BITS + ITER_BITS + CONF_BITS + 1 + AGE_BITS;

  static constexpr u64 RWRAM_BANKS = 4;  // banking for same-cycle read+write
#ifdef CHEATING_MODE
  static constexpr bool USE_CACHE = (CACHE_ENTRIES > 0);
#else
  static constexpr bool USE_CACHE = false;
#endif
  // Cache key: set_idx + way (to distinguish entries in the same set)
  static constexpr u64 WAY_BITS = (ASSOC > 1) ? clog2(ASSOC) : 1;
  static constexpr u64 CACHE_KEY_BITS = SET_BITS + WAY_BITS;

#ifdef TAGE_MONITOR
  // SW counters for monitoring (zero HARCOM cost)
  struct LoopStats {
    u64 lookups = 0;
    u64 hits = 0;           // tag match
    u64 overrides = 0;      // confident override fired
    u64 alloc_writes = 0;   // new entry allocations
    u64 update_writes = 0;  // existing entry updates
    u64 cache_hits = 0;     // entry cache hits
    u64 cache_misses = 0;   // entry cache misses (fall through to RAM)
    u64 cache_writes = 0;   // entry cache updates
    u64 conf_nonzero = 0;  // lookups where conf > 0
    u64 nb_iter_nonzero = 0; // lookups where nb_iter > 0
  } loop_stats;
#endif

  static_assert(NUM_SETS >= 2, "Need at least 2 sets");
  static_assert(ASSOC >= 1, "Need at least 1 way");
  static_assert(TABLE_SIZE == NUM_SETS * ASSOC, "TABLE_SIZE must be NUM_SETS * ASSOC");

  // ======== HARCOM Storage ========

  rwram<ENTRY_BITS, NUM_SETS, RWRAM_BANKS> loop_ram[ASSOC]{"loop"};

  // ======== Entry Cache ========
  // Small reg-based cache in front of rwram. Hot loop entries stay in cache,
  // avoiding repeated rwram reads to the same bank. Writes go to both cache
  // and rwram (rwram write may be delayed by banking, but cache is instant).

  // Entry cache: pure SW shadow (plain C++ types, zero HARCOM cost).
  // Tracks raw entry bits alongside the rwram. On cache hit, we construct
  // a val<ENTRY_BITS> from the cached raw bits instead of using the RAM result.
  // This avoids the rwram bank conflict problem for hot loop entries.
  struct EntryCache {
    u64 data[CACHE_ENTRIES] = {};       // raw entry bits
    u64 keys[CACHE_ENTRIES] = {};       // set_idx ++ way
    bool valids[CACHE_ENTRIES] = {};
    u64 rp = 0;                         // FIFO replacement pointer

    std::pair<bool, u64> lookup(u64 set_val, u64 way_val) const {
      u64 target = (way_val << SET_BITS) | set_val;
      for (u64 i = 0; i < CACHE_ENTRIES; i++) {
        if (valids[i] && keys[i] == target) return {true, data[i]};
      }
      return {false, 0};
    }

    void update(u64 set_val, u64 way_val, u64 entry_bits) {
      u64 target = (way_val << SET_BITS) | set_val;
      for (u64 i = 0; i < CACHE_ENTRIES; i++) {
        if (valids[i] && keys[i] == target) {
          data[i] = entry_bits;
          return;
        }
      }
      // Miss — evict at rp (FIFO)
      data[rp] = entry_bits;
      keys[rp] = target;
      valids[rp] = true;
      rp = (rp + 1) % CACHE_ENTRIES;
    }
  };

  std::conditional_t<USE_CACHE, EntryCache, EmptyMember> ecache;

  // Lookup results (set in lookup, consumed in override_predict and train)
  arr<reg<ENTRY_BITS>, ASSOC> read_entries;
  reg<SET_BITS> lookup_set;
  reg<TAG_BITS> lookup_tag;
  reg<1> hit;             // any way matched
  reg<std::max(u64(1), clog2(ASSOC))> hit_way;  // which way matched
  reg<1> loop_has_candidate;  // confident match found (pre-computed in lookup)

  // Lookup result (returned as vals — no regs on critical path)
  // TageImpl applies the TAGE-confidence gate on top.
  struct LookupResult {
    val<1> candidate;  // confident tag match found
    val<1> pred;       // loop prediction (valid when candidate=1)
  };

  // ======== lookup() — called in predict1 ========
  // Does RAM read (or cache hit) + unpack + tag match + confidence + prediction.
  // Returns LookupResult (vals) — TageImpl applies the TAGE gate later.

  // SW shadow of last lookup set index (for cache keying, no HARCOM cost)
  u64 lookup_set_sw = 0;

  LookupResult lookup(val<64> &inst_pc, [[maybe_unused]] auto &bindex,
                      u64 raw_pc = 0) {
    // Hash PC
    val<SET_BITS> set_idx = val<SET_BITS>{inst_pc >> 2};
    lookup_set = set_idx;
    val<TAG_BITS> tag = val<TAG_BITS>{inst_pc >> (2 + SET_BITS)};
    tag.fanout(hard<ASSOC + 1>{});
    lookup_tag = tag;
    if constexpr (ASSOC > 1) {
      set_idx.fanout(hard<ASSOC>{});
    }

    // SW cache key derived from raw PC (no HARCOM cost)
    lookup_set_sw = (raw_pc >> 2) & ((u64(1) << SET_BITS) - 1);

    // Per-way: read entry (from cache or RAM), compute tag/conf/pred
    // Confidence uses Ros-style: conf > 0 AND nb_iter >= (1 << CONF_THRESHOLD).
    arr<val<3>, ASSOC> way_results = [&](u64 w) -> val<3> {
      // Always do RAM read (HARCOM timing model)
      val<ENTRY_BITS> ram_entry = loop_ram[w].read(set_idx);

      // Check SW cache — on hit, override with cached raw bits
      val<ENTRY_BITS> entry = [&]() -> val<ENTRY_BITS> {
        if constexpr (USE_CACHE) {
          auto [ch, raw] = ecache.lookup(lookup_set_sw, w);
          if (ch) {
#ifdef TAGE_MONITOR
            loop_stats.cache_hits++;
#endif
            return val<ENTRY_BITS>{raw};
          }
#ifdef TAGE_MONITOR
          loop_stats.cache_misses++;
#endif
        }
        return ram_entry;
      }();

      read_entries[w] = entry;
      auto [e_tag, e_nb_iter, e_cur_iter, e_conf, e_dir, e_age] =
          split<TAG_BITS, ITER_BITS, ITER_BITS, CONF_BITS, 1, AGE_BITS>(entry);
      val<1> tag_match = (val<TAG_BITS>{e_tag} == tag);
      val<1> conf_nz = (val<CONF_BITS>{e_conf} != hard<0>{});
      val<1> iter_significant =
          (val<ITER_BITS>{e_nb_iter} >= hard<(u64(1) << CONF_THRESHOLD)>{});
      val<1> confident = conf_nz & iter_significant;
#ifdef TAGE_MONITOR
      if (static_cast<u64>(tag_match)) {
        if (static_cast<u64>(conf_nz)) loop_stats.conf_nonzero++;
        if (static_cast<u64>(e_nb_iter)) loop_stats.nb_iter_nonzero++;
      }
#endif
      val<ITER_BITS> next_iter =
          val<ITER_BITS>{val<ITER_BITS>{e_cur_iter} + hard<1>{}};
      val<1> at_exit = (next_iter == val<ITER_BITS>{e_nb_iter});
      val<1> pred = select(at_exit, ~val<1>{e_dir}, val<1>{e_dir});
      return concat(tag_match, confident, pred);
    };

    // Compute candidate + pred — IIFE to avoid val reassignment
    auto result = [&]() -> LookupResult {
      if constexpr (ASSOC == 1) {
        // Direct-mapped: no priority encode, minimal gate chain
        way_results.fanout(hard<3>{});
        val<1> tag_match = val<1>{way_results[0]};
        val<1> candidate = tag_match & val<1>{way_results[0] >> 1};
        val<1> pred = val<1>{way_results[0] >> 2};
        hit = tag_match;
        hit_way = val<std::max(u64(1), clog2(ASSOC))>{0};
#ifdef TAGE_MONITOR
        loop_stats.lookups++;
        if (static_cast<u64>(tag_match)) loop_stats.hits++;
        if (static_cast<u64>(candidate)) loop_stats.overrides++;
#endif
        return {candidate, pred};
      } else {
        // Multi-way: extract + priority encode
        way_results.fanout(hard<3>{});
        arr<val<1>, ASSOC> way_tag_match = [&](u64 w) -> val<1> {
          return val<1>{way_results[w]};
        };
        arr<val<1>, ASSOC> way_candidate = [&](u64 w) -> val<1> {
          return val<1>{way_results[w]} & val<1>{way_results[w] >> 1};
        };
        arr<val<1>, ASSOC> way_pred = [&](u64 w) -> val<1> {
          return val<1>{way_results[w] >> 2};
        };

        val<ASSOC> hit_mask = way_tag_match.fo1().concat();
        val<ASSOC> candidate_mask = way_candidate.fo1().concat();
        val<ASSOC> first_candidate = candidate_mask.one_hot();
        val<ASSOC> first_hit = hit_mask.one_hot();

        hit = (hit_mask != hard<0>{});
        if constexpr (ASSOC <= 2) {
          hit_way = val<std::max(u64(1), clog2(ASSOC))>{first_hit >> 1};
        } else {
          hit_way = encode(first_hit);
        }
#ifdef TAGE_MONITOR
        loop_stats.lookups++;
        if (static_cast<u64>(hit_mask) != 0) loop_stats.hits++;
        if (static_cast<u64>(candidate_mask) != 0) loop_stats.overrides++;
#endif
        return {(candidate_mask != hard<0>{}),
                (first_candidate & way_pred.fo1().concat()) != hard<0>{}};
      }
    }();

    return result;
  }

  // ======== Phase 3: train() ========

  void train(arr<val<1>, FETCH_WIDTH> &branch_taken,
             arr<val<1>, FETCH_WIDTH> &is_branch,
             val<1> &mispredict,
             val<1> &correct_pred,
             arr<reg<1>, FETCH_WIDTH> &override_active,
             [[maybe_unused]] reg<FETCH_WIDTH> &p2_before_override,
             val<1> &extra_cycle,
             [[maybe_unused]] u64 num_branch = 1) {
    // Train on offset 0 — matches the block PC used for lookup keying
    val<1> actual = branch_taken[0];
    val<1> is_br = is_branch[0];

    // Re-read the matching entry from saved lookup results
    // (read_entries was set during lookup, still valid)
    read_entries.fanout(hard<2>{});
    lookup_set.fanout(hard<ASSOC + 1>{});  // one per way write + one for alloc
    lookup_tag.fanout(hard<ASSOC + 1>{});
    hit.fanout(hard<ASSOC + 2>{});
    if constexpr (ASSOC > 1) {
      hit_way.fanout(hard<ASSOC>{});
    }

    // Combined update + allocation in a single loop (one RAM write per way max)
    val<1> should_alloc = ~val<1>{hit} & mispredict & is_br &
        (val<4>{static_cast<u64>(std::rand())} == hard<0>{});  // 1/16 probability

    static_loop<ASSOC>([&]<u64 w>() {
      auto [e_tag, e_nb_iter, e_cur_iter, e_conf, e_dir, e_age] =
          split<TAG_BITS, ITER_BITS, ITER_BITS, CONF_BITS, 1, AGE_BITS>(
              read_entries[w]);

      val<1> is_hit_way = val<1>{hit} &
          (val<std::max(u64(1), clog2(ASSOC))>{hit_way} == hard<w>{});

      // --- Update path (hit way) ---
      val<1> dir_flip = val<1>{e_dir} != actual;
      val<ITER_BITS> incremented_iter =
          val<ITER_BITS>{val<ITER_BITS>{e_cur_iter} + hard<1>{}};
      val<ITER_BITS> new_cur_iter =
          select(dir_flip, val<ITER_BITS>{0}, incremented_iter);
      val<1> iter_match =
          (val<ITER_BITS>{e_cur_iter} == val<ITER_BITS>{e_nb_iter});
      // Confidence update:
      //   dir_flip + iter_match (correct exit): increment
      //   dir_flip + !iter_match (wrong exit): decrement
      //   !dir_flip + nb_iter known: increment (correct iteration)
      //   !dir_flip + nb_iter==0 (learning): unchanged
      val<1> nb_known = (val<ITER_BITS>{e_nb_iter} != hard<0>{});
      val<CONF_BITS> new_conf = select(dir_flip,
          select(iter_match,
                 update_ctr(val<CONF_BITS>{e_conf}, val<1>{1}),
                 update_ctr(val<CONF_BITS>{e_conf}, val<1>{0})),
          select(nb_known,
                 update_ctr(val<CONF_BITS>{e_conf}, val<1>{1}),
                 val<CONF_BITS>{e_conf}));
      val<ITER_BITS> new_nb_iter = select(
          dir_flip & (val<ITER_BITS>{e_nb_iter} == hard<0>{}),
          val<ITER_BITS>{e_cur_iter},
          val<ITER_BITS>{e_nb_iter});
      val<1> free_entry = dir_flip & ~iter_match &
          (val<ITER_BITS>{e_nb_iter} != hard<0>{});
      val<AGE_BITS> new_age = select(
          val<1>{override_active[0]} & correct_pred,
          update_ctr(val<AGE_BITS>{e_age}, val<1>{1}),
          val<AGE_BITS>{e_age});
      val<ENTRY_BITS> updated_entry = select(free_entry,
          val<ENTRY_BITS>{0},
          concat(val<TAG_BITS>{e_tag}, new_nb_iter, new_cur_iter,
                 new_conf, select(dir_flip, actual, val<1>{e_dir}),
                 new_age));

      // --- Allocation path (miss way) ---
      val<1> age_zero = (val<AGE_BITS>{e_age} == hard<0>{});
      val<1> do_alloc = should_alloc & age_zero & ~is_hit_way;
      val<ENTRY_BITS> alloc_entry = concat(
          val<TAG_BITS>{lookup_tag},
          val<ITER_BITS>{0}, val<ITER_BITS>{0}, val<CONF_BITS>{0},
          ~actual, val<AGE_BITS>{0});

      // Single write: update if hit, allocate if miss, never both
      val<ENTRY_BITS> write_entry = select(is_hit_way,
          updated_entry, alloc_entry);
      val<1> do_write = (is_hit_way | do_alloc) & is_br;

      // Write to rwram (may be delayed by banking)
      execute_if(do_write, [&]() {
        loop_ram[w].write(val<SET_BITS>{lookup_set}, write_entry, extra_cycle);
      });

#ifdef CHEATING_MODE
      // Write to entry cache (instant — pure SW, only active in CHEATING_MODE)
      if constexpr (USE_CACHE) {
        if (static_cast<u64>(do_write)) {
          ecache.update(lookup_set_sw, w, static_cast<u64>(write_entry));
#ifdef TAGE_MONITOR
          loop_stats.cache_writes++;
#endif
        }
      }
#endif

#ifdef TAGE_MONITOR
      if (static_cast<u64>(do_alloc) && static_cast<u64>(is_br))
        loop_stats.alloc_writes++;
      if (static_cast<u64>(is_hit_way) && static_cast<u64>(is_br))
        loop_stats.update_writes++;
#endif
    });
  }
};
