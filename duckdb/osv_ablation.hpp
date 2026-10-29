#pragma once

// Ablation switches. A build is one step; each step adds one concept to the steps before it.
//   1 base      DuckDB's parquet reader on plain uCache: frame eviction, TLB shootdown by IPI
//   2 prefetch  column chunks are prefetched before they are read
//   3 pages     readers lock PageEntries, eviction picks pages (sweep over dir->pages)
//   4 queues    eviction pops pages from the page resident set (S/M FIFO)
//   5 noipi     no TLB shootdown IPIs, a CPU flushes by page version on its next lock
//   6 full      pages are read and decompressed straight from the VMA, without a copy
//
// The step is set by the justfile through osv_ablation_variant.hpp (generated, not in git).
#if __has_include("osv_ablation_variant.hpp")
#include "osv_ablation_variant.hpp"
#endif

#ifndef DUCKDB_ABLATION_STEP
#define DUCKDB_ABLATION_STEP 6
#endif

#if DUCKDB_ABLATION_STEP < 1 || DUCKDB_ABLATION_STEP > 6
#error "DUCKDB_ABLATION_STEP must be 1..6"
#endif

// Each switch can also be set by itself in osv_ablation_variant.hpp.
#ifndef DUCKDB_PREFETCH
#define DUCKDB_PREFETCH (DUCKDB_ABLATION_STEP >= 2)
#endif
#ifndef DUCKDB_PAGE_LOCKS
#define DUCKDB_PAGE_LOCKS (DUCKDB_ABLATION_STEP >= 3)
#endif
#ifndef DUCKDB_PAGE_EVICT
#define DUCKDB_PAGE_EVICT (DUCKDB_ABLATION_STEP >= 3)
#endif
// 0 = sweep over dir->pages, 1 = page resident set.
#ifndef DUCKDB_PAGE_RS
#define DUCKDB_PAGE_RS (DUCKDB_ABLATION_STEP >= 4)
#endif
#ifndef DUCKDB_NO_TLB_SHOOTDOWN
#define DUCKDB_NO_TLB_SHOOTDOWN (DUCKDB_ABLATION_STEP >= 5)
#endif
#ifndef DUCKDB_ZERO_COPY
#define DUCKDB_ZERO_COPY (DUCKDB_ABLATION_STEP >= 6)
#endif
// Not an ablation step: the number of leading frames of each page that eviction keeps resident,
// starting with the frame holding its header. A frame shared with the next page stays resident
// too. 0 evicts every frame.
#ifndef DUCKDB_KEEP_HEADER_FRAMES
#define DUCKDB_KEEP_HEADER_FRAMES 0
#endif

#if DUCKDB_PAGE_EVICT && !DUCKDB_PAGE_LOCKS
#error "page eviction X-locks its victims, it needs DUCKDB_PAGE_LOCKS"
#endif
#if DUCKDB_KEEP_HEADER_FRAMES && !DUCKDB_PAGE_EVICT
#error "keeping header frames needs DUCKDB_PAGE_EVICT"
#endif
#if DUCKDB_PAGE_RS && !DUCKDB_PAGE_EVICT
#error "the page resident set needs DUCKDB_PAGE_EVICT"
#endif
// Note: with DUCKDB_PREFETCH and without DUCKDB_PAGE_EVICT (step 2), a prefetched frame that is
// never read stays in the cache, as uCache's own policy does not evict it.

// Without page eviction nothing bumps the page versions the lazy flush relies on.
#if DUCKDB_NO_TLB_SHOOTDOWN && !DUCKDB_PAGE_EVICT
#error "skipping the TLB shootdown needs DUCKDB_PAGE_EVICT"
#endif
