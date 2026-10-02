#include "osv_page_resident_set.hpp"

#include <boost/circular_buffer.hpp>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>

namespace osv_duckdb {
namespace page_rs {

// Share of the cache, in pages, that S may hold.
static constexpr uint64_t kSmallPct = 10;
static constexpr uint32_t kMaxDirs = 4096;

// A FIFO ring behind a test-and-test-and-set spinlock.
struct alignas(64) PageFifo {
    std::atomic<bool> locked{false};
    // Copy of ring.size() for reads without the lock.
    std::atomic<size_t> size{0};
    boost::circular_buffer<Entry> ring;

    void lock() {
        while (locked.exchange(true, std::memory_order_acquire)) {
            while (locked.load(std::memory_order_relaxed)) _mm_pause();
        }
    }
    void unlock() { locked.store(false, std::memory_order_release); }

    // The ring holds every registered page, so a push never overwrites an entry.
    void check_room(size_t n) {
        if (ring.size() + n > ring.capacity()) {
            std::printf("[page_rs] queue overflow\n");
            std::abort();
        }
    }
};

static PageFifo g_fifo[2];
static std::atomic<PageDirectory *> g_dirs[kMaxDirs];
static uint32_t g_num_dirs = 0;
static uint64_t g_total_pages = 0;
static uint64_t g_total_bytes = 0;
static std::atomic<size_t> g_small_target{0};
static std::atomic<uint64_t> g_avg_page_bytes{1};

uint32_t Register(PageDirectory *dir, uint64_t num_pages, uint64_t page_bytes, uint64_t cache_bytes) {
    if (g_num_dirs == kMaxDirs) {
        std::printf("[page_rs] too many page directories\n");
        std::abort();
    }
    uint32_t id = g_num_dirs++;
    g_dirs[id].store(dir, std::memory_order_release);
    g_total_pages += num_pages;
    g_total_bytes += page_bytes;

    for (PageFifo &f : g_fifo) {
        f.lock();
        if (f.ring.capacity() < g_total_pages) f.ring.set_capacity(g_total_pages);
        f.unlock();
    }

    if (g_total_bytes > 0) {
        uint64_t avg_page_bytes = std::max<uint64_t>(1, g_total_bytes / g_total_pages);
        g_avg_page_bytes.store(avg_page_bytes, std::memory_order_relaxed);
        g_small_target.store(kSmallPct * (cache_bytes / avg_page_bytes) / 100, std::memory_order_relaxed);
    }
    return id;
}

PageDirectory *Directory(uint32_t dir_id) {
    return g_dirs[dir_id].load(std::memory_order_acquire);
}

void Insert(Entry e) {
    PushBatch(Small, &e, 1);
}

size_t PopBatch(Queue q, Entry *out, size_t max) {
    PageFifo &f = g_fifo[q];
    f.lock();
    size_t n = std::min(max, f.ring.size());
    auto one = f.ring.array_one();
    size_t n1 = std::min(n, one.second);
    std::copy(one.first, one.first + n1, out);
    if (n > n1) {
        auto two = f.ring.array_two();
        std::copy(two.first, two.first + (n - n1), out + n1);
    }
    f.ring.erase_begin(n);
    f.size.store(f.ring.size(), std::memory_order_relaxed);
    f.unlock();
    return n;
}

void PushBatch(Queue q, const Entry *in, size_t n) {
    if (n == 0) return;
    PageFifo &f = g_fifo[q];
    f.lock();
    f.check_room(n);
    f.ring.insert(f.ring.end(), in, in + n);
    f.size.store(f.ring.size(), std::memory_order_relaxed);
    f.unlock();
}

void UnpopBatch(Queue q, const Entry *in, size_t n) {
    if (n == 0) return;
    PageFifo &f = g_fifo[q];
    f.lock();
    f.check_room(n);
    for (size_t i = n; i-- > 0;) f.ring.push_front(in[i]);
    f.size.store(f.ring.size(), std::memory_order_relaxed);
    f.unlock();
}

size_t Size(Queue q) {
    return g_fifo[q].size.load(std::memory_order_relaxed);
}

size_t SmallTarget() {
    return g_small_target.load(std::memory_order_relaxed);
}

uint64_t AvgPageBytes() {
    return g_avg_page_bytes.load(std::memory_order_relaxed);
}

} // namespace page_rs
} // namespace osv_duckdb
