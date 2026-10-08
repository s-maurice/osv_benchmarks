#pragma once
#include <cstddef>
#include <cstdint>

#include "osv_ablation.hpp"

namespace osv_duckdb {

struct PageDirectory;

// Simplified S3-FIFO over the resident PageEntries of all page directories: a small FIFO S for
// pages accessed once, a main FIFO M for re-accessed pages. Eviction pops its victims from here,
// so its cost depends on how much it evicts, not on the data size.
namespace page_rs {

struct Entry {
    uint32_t dir_id;
    uint32_t page_idx;
};

enum Queue { Small = 0, Main = 1 };

// Adds a directory and returns its dir_id. The caller serializes registrations.
uint32_t Register(PageDirectory *dir, uint64_t num_pages, uint64_t page_bytes, uint64_t cache_bytes);
PageDirectory *Directory(uint32_t dir_id);

// Pushes a page that just became resident to S.
void Insert(Entry e);

// Pops up to max entries from the head of q.
size_t PopBatch(Queue q, Entry *out, size_t max);
// Appends to the tail of q.
void PushBatch(Queue q, const Entry *in, size_t n);
// Returns popped entries to the head of q, in their original order.
void UnpopBatch(Queue q, const Entry *in, size_t n);
size_t Size(Queue q);
// Number of entries S may hold before eviction takes from S instead of M.
size_t SmallTarget();
// Average page size over all registered directories.
uint64_t AvgPageBytes();

} // namespace page_rs
} // namespace osv_duckdb
