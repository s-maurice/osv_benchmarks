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
//   [63:56]  state  - Unlocked(0) / shared-count(1-124) / Locked(253) / Marked(254) / Evicted(255)
//                     ReAccess flag (0x80) on Unlocked / shared-count: the page was re-accessed.
//   [55:0]   version counter - bumped via bumpVersion() or exclusive unlock (see commented-out lockX)

namespace osv_duckdb {

using u64 = uint64_t;
using u32 = uint32_t;

struct PageState {
    std::atomic<u64> stateAndVersion;

    static const u64 Unlocked = 0;
    static const u64 MaxShared = 124;
    static const u64 ReAccess = 0x80;
    static const u64 UnlockedPostReAccess = Unlocked | ReAccess;
    static const u64 Locked = 253;
    static const u64 Marked = 254;
    static const u64 Evicted = 255;

    // Sentinel version that never matches a real version, forcing a TLB flush on first use.
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
    bool tryLockX(u64 v) { return stateAndVersion.compare_exchange_strong(v, sameVersion(v, Locked)); }
    // Bumps the version so that accessors know that their TLB entry is invalid.
    // Use this after we have modified any PTEs for frames on this page. Returns the new value.
    // Caller must already hold an exclusive lock.
    u64 unlockXEvicted(u64 v) {
        u64 new_v = nextVersion(v, Evicted);
        stateAndVersion.store(new_v, std::memory_order_release);
        return new_v;
    }
    // Use this when we unlock a page without unmapping it in the page table
    // (we did not invalidate any TLBs). Caller must already hold an exclusive lock.
    void unlockXSameVersion(u64 locked_v) { stateAndVersion.store(sameVersion(locked_v, Unlocked), std::memory_order_release); }
    // Unlock into Evicted without bumping the version.
    // Caller must already hold an exclusive lock.
    void unlockXEvictedSameVersion(u64 locked_v) { stateAndVersion.store(sameVersion(locked_v, Evicted), std::memory_order_release); }

    // Only Unlocked -> shared sets ReAccess, joining a held page does not.
    bool tryLockS(u64 v) {
        u64 s = getState(v);
        if (s == Unlocked) return stateAndVersion.compare_exchange_strong(v, sameVersion(v, ReAccess | 1));
        if (s < Locked && count(s) < MaxShared) return stateAndVersion.compare_exchange_strong(v, sameVersion(v, s+1));
        if (s == Marked || s == Evicted) return stateAndVersion.compare_exchange_strong(v, sameVersion(v, 1));
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
            assert(isShared(s));
            if (stateAndVersion.compare_exchange_strong(v, sameVersion(v, s-1))) return;
        }
    }

    bool tryMark(u64 v) { return stateAndVersion.compare_exchange_strong(v, sameVersion(v, Marked)); }
    bool tryClearReAccess(u64 v) { return stateAndVersion.compare_exchange_strong(v, sameVersion(v, Unlocked)); }

    static u64 count(u64 s)     { return s & ~ReAccess; }
    static bool isShared(u64 s) { return s < Locked && count(s) > 0; }

    static u64 getState(u64 v)   { return v >> 56; }
    static u64 getVersion(u64 v) { return v & ((1ULL << 56) - 1); }
    u64 getState() { return getState(stateAndVersion.load()); }
    u64 load()     { return stateAndVersion.load(); }

    void operator=(PageState &) = delete;
};

} // namespace osv_duckdb
