#pragma once
#include <atomic>
#include <cassert>
#include <cstdint>
#include <immintrin.h>

// PageState - shared lock + optimistic version counter packed into a single atomic<u64>.
//
// Extracted from integrated_vmcache.cpp. No OSv kernel dependencies.
//
// Layout (64 bits):
//   [63:56]  state  - Unlocked(0) / shared-count(1-252) / Locked(253) / Marked(254) / Evicted(255)
//   [55:0]   version counter - bumped via bumpVersion() or exclusive unlock (see commented-out lockX)

namespace osv_duckdb {

using u64 = uint64_t;
using u32 = uint32_t;

struct PageState {
    std::atomic<u64> stateAndVersion;

    static const u64 Unlocked = 0;
    static const u64 MaxShared = 252;
    static const u64 Locked = 253;
    static const u64 Marked = 254;
    static const u64 Evicted = 255;

    // Sentinel that always fails validateRead() - state bits = Evicted (255).
    static constexpr u64 NO_VERSION = ~0ULL;

    PageState() {}

    static inline u64 sameVersion(u64 v, u64 newState) { return ((v<<8)>>8) | newState<<56; }
    static inline u64 nextVersion(u64 v, u64 newState) { return (((v<<8)>>8)+1) | newState<<56; }

    // Increment the version counter in-place, keeping the current state.
    // Call this when page contents change (e.g. eviction) to invalidate optimistic readers.
    void bumpVersion() {
        while (true) {
            u64 v = stateAndVersion.load();
            if (stateAndVersion.compare_exchange_strong(v, nextVersion(v, getState(v)))) return;
        }
    }

    // Exclusive lock: CAS current value v to Locked, preserving the version counter.
    // Caller decides which source state is acceptable by checking before calling.
    bool tryLockX(u64 v)        { return stateAndVersion.compare_exchange_strong(v, sameVersion(v, Locked)); }
    // Unlock after eviction: bump version so readers detect the change. v is the current
    // value, which the caller must be holding X. Returns the stored value.
    u64 unlockXNextVersion(u64 v) {
        u64 new_v = nextVersion(v, Unlocked);
        stateAndVersion.store(new_v, std::memory_order_release);
        return new_v;
    }
    // Unlock without bumping version: use when no data changed (e.g. all eviction
    // candidates were already in-flight or Uncached, so no frame was actually freed).
    void unlockXSameVersion(u64 locked_v) { stateAndVersion.store(sameVersion(locked_v, Unlocked), std::memory_order_release); }
    void unlockXEvicted()       { stateAndVersion.store(nextVersion(stateAndVersion.load(), Evicted),  std::memory_order_release); }
    void downgradeLock()        { stateAndVersion.store(nextVersion(stateAndVersion.load(), 1),        std::memory_order_release); }

    bool tryLockS(u64 v) {
        u64 s = getState(v);
        if (s < MaxShared) return stateAndVersion.compare_exchange_strong(v, sameVersion(v, s+1));
        if (s == Marked) return stateAndVersion.compare_exchange_strong(v, sameVersion(v, 1));
        return false;
    }

    void lockS() {
        for (;;) {
            u64 v = stateAndVersion.load();
            if (tryLockS(v)) return;
            _mm_pause();
        }
    }

    void unlockS() {
        while (true) {
            u64 v = stateAndVersion.load();
            u64 s = getState(v);
            assert(s > 0 && s <= MaxShared);
            if (stateAndVersion.compare_exchange_strong(v, sameVersion(v, s-1))) return;
        }
    }

    // Returns a version counter for optimistic reads (no lock taken).
    u64 beginRead() {
        for (;;) {
            u64 v = stateAndVersion.load();
            u64 s = getState(v);
            if (s != Locked && s != Evicted) return v;
            _mm_pause();
        }
    }

    // Returns true if the version counter hasn't changed since beginRead().
    bool validateRead(u64 v) {
        u64 cur = stateAndVersion.load();
        if (cur == v) return true;
        if ((cur << 8) != (v << 8)) return false;
        u64 s = getState(cur);
        return s <= MaxShared || s == Marked;
    }

    bool tryMark(u64 v) { return stateAndVersion.compare_exchange_strong(v, sameVersion(v, Marked)); }

    static u64 getState(u64 v)   { return v >> 56; }
    static u64 getVersion(u64 v) { return v & ((1ULL << 56) - 1); }
    u64 getState() { return getState(stateAndVersion.load()); }
    u64 load()     { return stateAndVersion.load(); }

    void operator=(PageState &) = delete;
};

} // namespace osv_duckdb
