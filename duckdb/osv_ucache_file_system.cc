#include "osv_ucache_file_system.hpp"
#include "osv_parquet_file_handle.hpp"
#include "osv_page_resident_set.hpp"

#include <duckdb/common/exception.hpp>
#include <duckdb/common/file_open_flags.hpp>

#include <osv/mmu.hh>
#include <osv/sched.hh>

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <array>
#include <mutex>
#include <numeric>

namespace osv_duckdb {

// ─────────────────────────────────────────────────────────────────────────────
// enqueue_prefetch - called after readers lock pages (AcquireChunkLocks, VmaPagePin::Prefetch),
// and for footer/metadata reads (OsvCachingFileHandle::RegisterPrefetch).
//
// Immediately issues async NVMe IO for all pages in [pos, pos+len) that are
// not yet in the cache.  When the page fault arrives later, checkPipeline()
// in uCache::handleFault() finds the buffer in Reading/ReadyToInsert state
// and simply polls for completion rather than starting a new synchronous read.
// ─────────────────────────────────────────────────────────────────────────────
void enqueue_prefetch(ucache::VMA *vma, duckdb::idx_t pos, duckdb::idx_t len) {
#if !DUCKDB_PREFETCH
    (void)vma; (void)pos; (void)len;
#else
    if (len == 0 || ucache::uCacheManager == nullptr) return;

    // Count total in-flight prefetch IOs across all CPUs to avoid saturating the
    // NVMe submission queue.  Only issue up to (prefetch_batch - total_inflight)
    // new IOs per call.
    //
    // A request leaves the count when its queue sees the completion, so collect the
    // completions of this CPU's queue first.
    vma->file->try_poll_on_cpu(sched::cpu::current()->id);
    u64 total_inflight = 0;
    for (size_t i = 0; i < sched::cpus.size(); i++) {
        int v = ucache::uCacheManager->per_cpu_inflight_count[i].load(
                    std::memory_order_relaxed);
        if (v > 0) total_inflight += (u64)v;
    }

    u64 batch_cap = ucache::uCacheManager->prefetch_batch;
    if (total_inflight >= batch_cap) return;  // queue already busy, skip this hint
    u64 max_new = batch_cap - total_inflight;

    duckdb::idx_t first = pos / vma->pageSize;
    duckdb::idx_t last = (pos + len - 1) / vma->pageSize;

    std::vector<ucache::Buffer *> pl;
    for (duckdb::idx_t i = first; i <= last && i < (duckdb::idx_t)vma->buffers.size(); i++) {
        if (pl.size() >= max_new) break;
        ucache::Buffer *buf = vma->buffers[i];
        ucache::BufferSnapshot bs(vma->nbPages);
        buf->updateSnapshot(&bs);
        if (bs.state == ucache::BufferState::Uncached)
            pl.push_back(buf);
    }
    if (!pl.empty()) {
        ucache::uCacheManager->ensureFreePages(vma->pageSize * pl.size());
        ucache::uCacheManager->prefetch(vma, pl);
    }
#endif
}


// ─────────────────────────────────────────────────────────────────────────────
// Eviction callbacks for DuckDB VMA
// ─────────────────────────────────────────────────────────────────────────────

#if DUCKDB_PAGE_EVICT

// Frames that could not be evicted when their page was evicted, are retried at each eviction.
// Some reasons for this could be: I/O was still in-flight, or the frame was a straddle-frame,
// where the other page(s) were locked.
// Their pages stay Evicted, so only the graveyard evicts them; dropped if a reader relocks a page.
struct GraveyardEntry { ucache::VMA* vma; u64 frame; };
static std::mutex g_eviction_graveyard_mu;
static std::vector<GraveyardEntry> g_eviction_graveyard;

// Vector containing the currently held page locks for eviction.
// flushed_all: this eviction submits every frame of the page, so the evicting CPU flushes all
// of the page's TLB entries and can skip that flush on its next lock of the page.
struct PendingUnlock { PageDirectory* dir; u32 page_idx; bool flushed_all; };
static thread_local std::vector<PendingUnlock> t_pending_unlock;

enum class FramePages { AllHeld, Resident, Busy };

// X-locks every page overlapping frame, except those starting in [skip_start, skip_end), and
// appends them to held. All or nothing, and only from Evicted: a frame's PTE may be cleared only
// while every page on it is X-locked, and each of them needs a version bump afterwards.
// Resident: a page is resident and still covers the frame. Busy: another eviction holds a page.
static FramePages lock_frame_pages(PageDirectory* dir, ucache::VMA* vma, u64 frame,
                                   u64 skip_start, u64 skip_end, std::vector<u32>& held) {
    const size_t held_before = held.size();
    u64 f_start = frame * vma->pageSize;
    u64 f_end = f_start + vma->pageSize;
    FramePages result = FramePages::AllHeld;

    // Step 1: lock the overlapping pages in ascending offset order, stop at the first failure.
    auto it = std::lower_bound(dir->offset_index.begin(), dir->offset_index.end(), f_start,
        [&](u32 idx, u64 val) {
            return dir->pages[idx].offset + (u64)dir->pages[idx].page_size() <= val;
        });
    for (; it != dir->offset_index.end() && dir->pages[*it].offset < f_end; ++it) {
        PageEntry& pe = dir->pages[*it];
        if (pe.offset >= skip_start && pe.offset < skip_end) continue;
        u64 v = pe.lock.load();
        u64 s = PageState::getState(v);
        if (s == PageState::Evicted && pe.lock.tryLockX(v)) {
            held.push_back(*it);
            continue;
        }
        // Locked, or we lost the race for an Evicted page.
        result = (s == PageState::Locked || s == PageState::Evicted) ? FramePages::Busy : FramePages::Resident;
        break;
    }

    // Step 2: on failure release what this call locked.
    if (result != FramePages::AllHeld) {
        for (size_t i = held_before; i < held.size(); i++) {
            PageState& lock = dir->pages[held[i]].lock;
            lock.unlockXEvictedSameVersion(lock.load());
        }
        held.resize(held_before);
    }
    return result;
}

#if !DUCKDB_KEEP_HEADER_FRAMES
// Submits the frames of run: X-locked pages in ascending, byte-adjacent order. A frame shared
// with pages outside the run is submitted only if those are all Evicted and can be locked, so a
// shared frame is freed by whichever of its pages is evicted last. Every page of the run ends
// Evicted: here if none of its frames is submitted, else in post_EvictedBatch.
static void evict_frames_in_run(PageDirectory* dir, ucache::VMA* vma, const std::vector<u32>& run,
                                 ucache::EvictList el) {
    u64 start = dir->pages[run.front()].offset;
    u64 end = dir->pages[run.back()].offset + (u64)dir->pages[run.back()].page_size();
    u64 first = start / vma->pageSize;
    u64 last = (end - 1) / vma->pageSize;

    std::vector<u64> evicted; // ascending
    std::vector<u64> to_graveyard;
    // Pages outside the run, X-locked for a submitted edge frame.
    std::vector<u32> neighbours;

    // Step 1: submit the run's frames.
    for (u64 f = first; f <= last && f < vma->buffers.size(); f++) {
        ucache::Buffer* buf = vma->buffers[f];
        // Not exception safe: if we throw, we would not properly clean up `bs`.
        auto* bs = new ucache::BufferSnapshot(vma->nbPages);
        buf->updateSnapshot(bs);

        // validate: is this even possible?
        if (bs->state == ucache::BufferState::Uncached) {
            delete bs;
            continue;
        }

        // An edge frame is shared with pages outside the run, lock those first.
        bool edge = (f == first && start % vma->pageSize != 0) || (f == last && end % vma->pageSize != 0);
        const size_t neighbours_before = neighbours.size();
        if (edge) {
            FramePages others = lock_frame_pages(dir, vma, f, start, end, neighbours);
            if (others == FramePages::Resident) {
                delete bs;
                continue;
            }
            if (others == FramePages::Busy) {
                // both evictors would fail here, so push to graveyard to retry later
                delete bs;
                to_graveyard.push_back(f);
                continue;
            }
            // The frame may have changed before its other pages were locked.
            buf->updateSnapshot(bs);
        }

        // Submit the frame. On success uCache owns bs.
        bool uncached = bs->state == ucache::BufferState::Uncached;
        if (!uncached && vma->addEvictionCandidateAllowUnaccessed(buf, bs, el)) {
            evicted.push_back(f);
            continue;
        }

        // Not submitted, release the pages locked for it.
        delete bs;
        for (size_t i = neighbours_before; i < neighbours.size(); i++) {
            PageState& lock = dir->pages[neighbours[i]].lock;
            lock.unlockXEvictedSameVersion(lock.load());
        }
        neighbours.resize(neighbours_before);
        // Prefetch I/O still pending: the device may still write into the frame, so retry later.
        if (!uncached) to_graveyard.push_back(f);
    }

    // Step 2: queue the locked pages outside the run for unlock in post_EvictedBatch. This CPU
    // flushes only their edge frame, so flushed_all is false.
    for (u32 k : neighbours) t_pending_unlock.push_back({dir, k, false});

    // Step 3: unlock the run's pages. A page with a submitted frame is unlocked in
    // post_EvictedBatch. A page with none is unlocked here, without incrementing its version.
    for (u32 k : run) {
        PageEntry& pe = dir->pages[k];
        u64 pf_first = pe.offset / vma->pageSize;
        u64 pf_end = (pe.offset + (u64)pe.page_size() - 1) / vma->pageSize + 1;
        u64 submitted = 0;
        for (u64 f = pf_first; f < pf_end; f++) {
            if (std::binary_search(evicted.begin(), evicted.end(), f)) submitted++;
        }

        if (submitted > 0) {
            t_pending_unlock.push_back({dir, k, submitted == pf_end - pf_first});
        } else {
            // No frame of this page is being evicted: each is Uncached, left to a resident page
            // that shares it, or in the graveyard. No PTE changed, so the version stays.
            pe.lock.unlockXEvictedSameVersion(pe.lock.load());
        }
    }

    // Step 4: add graveyarded frames into the graveyard.
    if (!to_graveyard.empty()) {
        std::scoped_lock lk(g_eviction_graveyard_mu);
        for (u64 f : to_graveyard) g_eviction_graveyard.push_back({vma, f});
    }
}

#else
// Like evict_frames_in_run, but submits only the inner frames of each page: those after its
// first DUCKDB_KEEP_HEADER_FRAMES frames, which hold its header. Frames which are shared with
// other pages are not evicted.
static void evict_inner_frames_in_run(PageDirectory* dir, ucache::VMA* vma, const std::vector<u32>& run,
                                       ucache::EvictList el) {
    std::vector<u64> to_graveyard;

    for (u32 k : run) {
        PageEntry& pe = dir->pages[k];
        u64 inner_first = pe.offset / vma->pageSize + DUCKDB_KEEP_HEADER_FRAMES;
        u64 inner_end = std::min<u64>((pe.offset + (u64)pe.page_size()) / vma->pageSize, vma->buffers.size());

        // Step 1: submit the page's inner frames.
        u64 submitted = 0;
        for (u64 f = inner_first; f < inner_end; f++) {
            ucache::Buffer* buf = vma->buffers[f];
            // Not exception safe: if we throw, we would not properly clean up `bs`.
            auto* bs = new ucache::BufferSnapshot(vma->nbPages);
            buf->updateSnapshot(bs);
            if (bs->state == ucache::BufferState::Uncached) {
                delete bs;
                continue;
            }

            // On success uCache owns bs.
            if (vma->addEvictionCandidateAllowUnaccessed(buf, bs, el)) {
                submitted++;
                continue;
            }
            delete bs;
            // Prefetch I/O still pending: the device may still write into the frame, so retry later.
            to_graveyard.push_back(f);
        }

        // Step 2: unlock the page, in post_EvictedBatch if a frame was submitted. The PTEs of the
        // kept frames never change, so submitting every inner frame flushes all of the page.
        if (submitted > 0) {
            t_pending_unlock.push_back({dir, k, submitted == inner_end - inner_first});
        } else {
            pe.lock.unlockXEvictedSameVersion(pe.lock.load());
        }
    }

    // Step 3: add graveyarded frames into the graveyard.
    if (!to_graveyard.empty()) {
        std::scoped_lock lk(g_eviction_graveyard_mu);
        for (u64 f : to_graveyard) g_eviction_graveyard.push_back({vma, f});
    }
}
#endif

// Submits the frames of a run of victim pages, through the variant this build uses.
static void evict_run(PageDirectory* dir, ucache::VMA* vma, const std::vector<u32>& run,
                      ucache::EvictList el) {
#if DUCKDB_KEEP_HEADER_FRAMES
    evict_inner_frames_in_run(dir, vma, run, el);
#else
    evict_frames_in_run(dir, vma, run, el);
#endif
}

// Retries frames left in g_eviction_graveyard by an earlier evict_frames_in_run call. Global
// and VMA-independent: any CPU's eviction call, for any file, can drain it.
static void drain_eviction_graveyard(ucache::EvictList el) {
    std::vector<GraveyardEntry> to_retry;
    {
        // Swap, don't drain in place: a frame that fails again is re-queued below and belongs
        // to the next drain, not this one.
        std::scoped_lock lk(g_eviction_graveyard_mu);
        to_retry.swap(g_eviction_graveyard);
    }

    // Step 1: drop duplicates. Two evictions can send the same frame.
    std::sort(to_retry.begin(), to_retry.end(), [](const GraveyardEntry& a, const GraveyardEntry& b) {
        return a.vma != b.vma ? a.vma < b.vma : a.frame < b.frame;
    });
    to_retry.erase(std::unique(to_retry.begin(), to_retry.end(), [](const GraveyardEntry& a, const GraveyardEntry& b) {
        return a.vma == b.vma && a.frame == b.frame;
    }), to_retry.end());

    std::vector<u32> held;
    std::vector<GraveyardEntry> requeue;

    // Step 2: retry each frame.
    for (const GraveyardEntry& e : to_retry) {
        ucache::VMA* vma = e.vma;
        auto* dir = static_cast<PageDirectory*>(vma->options.user_data);
        if (!dir || e.frame >= vma->buffers.size()) continue;

        // Lock every page on the frame.
        held.clear();
        FramePages pages = lock_frame_pages(dir, vma, e.frame, 0, 0, held);
        // A reader relocked a page: drop the entry, that page covers the frame again.
        if (pages == FramePages::Resident) continue;
        // Another eviction holds a page, which will end up Evicted: retry later.
        if (pages == FramePages::Busy) {
            requeue.push_back(e);
            continue;
        }

        // Submit the frame. On success uCache owns bs.
        ucache::Buffer* buf = vma->buffers[e.frame];
        // Not exception safe; if we throw, we would not properly clean up `bs`.
        auto* bs = new ucache::BufferSnapshot(vma->nbPages);
        buf->updateSnapshot(bs);
        // An Uncached frame is already freed, so the entry is dropped.
        bool uncached = bs->state == ucache::BufferState::Uncached;
        if (!uncached && vma->addEvictionCandidateAllowUnaccessed(buf, bs, el)) {
            for (u32 pi : held) t_pending_unlock.push_back({dir, pi, false});
            continue;
        }

        // Not submitted, release the locks without incrementing the versions.
        delete bs;
        for (u32 pi : held)
            dir->pages[pi].lock.unlockXEvictedSameVersion(dir->pages[pi].lock.load());
        if (!uncached) requeue.push_back(e);
    }

    // Step 3: keep the frames that are still not evictable for the next drain.
    if (!requeue.empty()) {
        std::scoped_lock lk(g_eviction_graveyard_mu);
        g_eviction_graveyard.insert(g_eviction_graveyard.end(), requeue.begin(), requeue.end());
    }
}

#if !DUCKDB_PAGE_RS
static void duckdb_evict_policy(ucache::VMA* vma, u64 nbToEvict, ucache::EvictList el) {
    drain_eviction_graveyard(el);

    auto* dir = static_cast<PageDirectory*>(vma->options.user_data);
    if (!dir) return;
    const size_t n = dir->pages.size();
    if (n == 0) return;

    // returns wether we marked any pages for second chance.
    auto sweep = [&]() -> bool {
        bool marked = false;
        bool in_run = false;
        size_t run_lo = 0;
        size_t run_hi = 0;
        std::vector<u32> run;

        auto end_run = [&]() {
            if (in_run) {
                run.clear();
                for (size_t k = run_lo; k <= run_hi; k++) run.push_back((u32)k);
                evict_run(dir, vma, run, el);
            }
            in_run = false;
        };

        size_t i = dir->evict_cursor.load(std::memory_order_relaxed) % n;
        for (size_t scanned = 0; scanned < n; scanned++, i = (i + 1 == n) ? 0 : i + 1) {
            if ((u64)el.size() >= nbToEvict) { end_run(); break; }
            PageEntry& pe = dir->pages[i];

            u64 v = pe.lock.load();
            u64 s = PageState::getState(v);

            // check if we are pinned by a reader
            if (PageState::isShared(s)) {
                end_run();
                continue;
            }
            if (s == PageState::Locked || s == PageState::Evicted) {
                end_run();
                continue;
            }

            // second chance policy, mark the PageEntry
            if (s == PageState::Unlocked || s == PageState::UnlockedPostReAccess) {
                pe.lock.tryMark(v);
                marked = true;
                end_run();
                continue;
            }

            assert(s == PageState::Marked);

            // check if we have a byte gap between our run and the current PageEntry
            if (in_run && dir->pages[run_hi].offset + (u64) dir->pages[run_hi].page_size() != pe.offset) {
                end_run();
            }

            // if we lost the race, skip
            if (!pe.lock.tryLockX(v)) { end_run(); continue; }

            if (!in_run) {
                run_lo = i;
                in_run = true;
            }
            run_hi = i;

            // check if we have evicted enough frames
            u64 run_bytes = pe.offset + (u64)pe.page_size() - dir->pages[run_lo].offset;
            if ((u64)el.size() + run_bytes / vma->pageSize >= nbToEvict) end_run();
        }

        end_run();
        dir->evict_cursor.store((u32)i, std::memory_order_relaxed);
        return marked;
    };

    // if we sweep marked but didn't evict anything, try again.
    const size_t start_size = el.size();
    if (sweep() && el.size() == start_size) {
        sweep();
    }
}

#else
// Pops victims from the page resident set. Victim selection is global across files, so the vma
// uCache passes in is ignored.
static void page_rs_evict_policy(ucache::VMA* /*vma*/, u64 nbToEvict, ucache::EvictList el) {
    using namespace page_rs;
    static constexpr size_t kMaxPop = 64;

    drain_eviction_graveyard(el);

    struct Victim { Entry entry; Queue origin; };
    Entry popped[kMaxPop];
    // Popped entries that stay resident, per destination queue.
    std::array<std::vector<Entry>, 2> requeue;
    std::vector<Victim> victims;
    std::vector<u32> run;

    // Bounds the work when most resident pages are held by readers.
    int64_t budget = 2 * (int64_t)(Size(Small) + Size(Main));

    while ((u64)el.size() < nbToEvict && budget > 0) {
        // Pop about as many pages as cover the frames still needed.
        u64 want_frames = nbToEvict - el.size();
        size_t to_pop = std::clamp<u64>(want_frames * mmu::page_size / AvgPageBytes() + 1, 1, kMaxPop);

        Queue queue = (Size(Small) > SmallTarget() || Size(Main) == 0) ? Small : Main;
        size_t n = PopBatch(queue, popped, to_pop);
        if (n == 0) {
            queue = (queue == Small) ? Main : Small;
            n = PopBatch(queue, popped, to_pop);
            if (n == 0) break;
        }
        for (size_t i = 0; i < n; i++)
            __builtin_prefetch(&Directory(popped[i].dir_id)->pages[popped[i].page_idx]);

        // Take victims until they cover the frames still needed.
        u64 victim_frames = 0;
        size_t i = 0;
        for (; i < n && victim_frames < want_frames; i++, budget--) {
            Entry entry = popped[i];
            PageDirectory* dir = Directory(entry.dir_id);
            PageEntry& pe = dir->pages[entry.page_idx];
            u64 v = pe.lock.load();
            u64 s = PageState::getState(v);
            // We own the page's only entry, so it is resident and no other evictor holds it.
            ucache::assert_crash(s != PageState::Locked && s != PageState::Marked && s != PageState::Evicted);
            if (s == PageState::UnlockedPostReAccess && pe.lock.tryClearReAccess(v)) {
                // From S: promotion, from M: reinsertion.
                requeue[Main].push_back(entry);
            } else if (s == PageState::Unlocked && pe.lock.tryLockX(v)) {
                victims.push_back({entry, queue});
                victim_frames += (pe.page_size() + dir->vma->pageSize - 1) / dir->vma->pageSize;
            } else {
                // Held by a reader, or we lost the race to one.
                requeue[queue].push_back(entry);
            }
        }
        UnpopBatch(queue, popped + i, n - i);
        for (Queue q : {Small, Main}) {
            PushBatch(q, requeue[q].data(), requeue[q].size());
            requeue[q].clear();
        }

        // sorting globally allows for file-adjacent regions to remain adjacent, allowing them to form runs
        std::sort(victims.begin(), victims.end(), [](const Victim& a, const Victim& b) {
            if (a.entry.dir_id != b.entry.dir_id) return a.entry.dir_id < b.entry.dir_id;
            return Directory(a.entry.dir_id)->pages[a.entry.page_idx].offset <
                   Directory(b.entry.dir_id)->pages[b.entry.page_idx].offset;
        });

        // submit evictions in runs
        for (size_t run_start = 0; run_start < victims.size();) {
            const u32 dir_id = victims[run_start].entry.dir_id;
            PageDirectory* dir = Directory(dir_id);

            // extend the current run while we are in the same file and the next page is adjacent
            run.clear();
            run.push_back(victims[run_start].entry.page_idx);
            size_t run_end = run_start + 1;
            for (; run_end < victims.size() && victims[run_end].entry.dir_id == dir_id; run_end++) {
                PageEntry& prev = dir->pages[run.back()];
                if (prev.offset + (u64)prev.page_size() != dir->pages[victims[run_end].entry.page_idx].offset) break;
                run.push_back(victims[run_end].entry.page_idx);
            }

            evict_run(dir, dir->vma, run, el);
            run_start = run_end;
        }
        victims.clear();
    }
}
#endif

// Eviction cannot fail.
// We already hold an exclusive lock, the PTE cannot have changed since we snapshotted it.
static bool duckdb_canBeEvicted(ucache::Buffer* /*buf*/) {
    return true;
}

static void duckdb_postEvictedBatch(ucache::Buffer* const* buffers, size_t count) {
    // all the buffers belong to a single vma
    auto* dir = static_cast<PageDirectory*>(buffers[0]->vma->options.user_data);
    if (!dir) return;

    // release all the locks
    size_t keep = 0;
    for (const PendingUnlock& p : t_pending_unlock) {
        // belongs to another vma, skip
        if (p.dir != dir) {
            t_pending_unlock[keep++] = p;
            continue;
        }
        PageEntry& pe = dir->pages[p.page_idx];
        u64 v = pe.lock.load();

        // we are the lock holder
        ucache::assert_crash(PageState::getState(v) == PageState::Locked);
        u64 new_v = pe.lock.unlockXEvicted(v);

        // Optimisation: skip the redundant version check and TLB flush that
        // lockSWithFlush/validateFlushTlb would otherwise fire on this CPU's
        // next read of the page. Only valid if we actually flushed every frame on this page.
        if (p.flushed_all) dir->thread_versions[GetThreadSlot()][p.page_idx] = new_v;
    }
    t_pending_unlock.resize(keep);

    // Old approach: we previously calculated the mapping back from frame -> page, relying on
    // buffers[] being provided already sorted by base address by uCache::evict.
    // This was error prone: for example, we need to make sure we do not unlock a page twice.
    // The following is true (and must be true): If a frame is evicted, we hold the lock for ALL pages which contain the frame.
    //
    // size_t next = 0;
    // for (size_t i = 0; i < count; i++) {
    //     u64 buf_start = (u64)buffers[i]->baseVirt - (u64)buffers[i]->vma->start;
    //     u64 buf_end = buf_start + buffers[i]->vma->pageSize;
    //
    //     auto first = std::lower_bound(dir->offset_index.begin(), dir->offset_index.end(), buf_start,
    //         [&](u32 idx, u64 val) {
    //             return dir->pages[idx].offset + (u64)dir->pages[idx].page_size() <= val;
    //         });
    //     size_t k = std::max(next, (size_t)(first - dir->offset_index.begin()));
    //
    //     for (; k < dir->offset_index.size() && dir->pages[dir->offset_index[k]].offset < buf_end; k++) {
    //         u32 pi = dir->offset_index[k];
    //         PageEntry& pe = dir->pages[pi];
    //         u64 v = pe.lock.load();
    //         ucache::assert_crash(PageState::getState(v) == PageState::Locked);
    //         dir->thread_versions[GetThreadSlot()][pi] = pe.lock.unlockXNextVersion(v);
    //     }
    //     next = k;
    // }
}
#endif // DUCKDB_PAGE_EVICT

// ─────────────────────────────────────────────────────────────────────────────
// Page directory persistence
// ─────────────────────────────────────────────────────────────────────────────

void PageResidentSetInsert(PageDirectory *dir, u32 page_idx) {
#if DUCKDB_PAGE_RS
    page_rs::Insert({dir->dir_id, page_idx});
#else
    (void)dir;
    (void)page_idx;
#endif
}

size_t GetThreadSlot() {
    return static_cast<size_t>(sched::cpu::current()->id);
}

// We need to make sure we don't migrate to another core (with an unknown TLB state) after
// we have validated our TLB.
void DisableMigration() { sched::migrate_disable(); }
void EnableMigration()  { sched::migrate_enable(); }

u64 lockSWithFlush(PageState &ps, u64 &local_version, const char *page_addr, u64 page_size) {
    std::vector<void *> flush_pages;
    auto make_flush_pages = [&]() -> std::vector<void *> & {
        if (flush_pages.empty()) {
            static constexpr u64 k = 4096;
            uintptr_t start = reinterpret_cast<uintptr_t>(page_addr) & ~(k - 1);
            uintptr_t end = reinterpret_cast<uintptr_t>(page_addr) + page_size;
            flush_pages.reserve((end - start + k - 1) / k);
            for (uintptr_t p = start; p < end; p += k)
                flush_pages.push_back(reinterpret_cast<void *>(p));
        }
        return flush_pages;
    };

    for (;;) {
        u64 v = ps.load();
        u64 s = PageState::getState(v);
        if (s == PageState::Locked) {
            _mm_pause();
            continue;
        }
        ucache::assert_crash(!PageState::isShared(s) || PageState::count(s) < PageState::MaxShared);
        if (PageState::getVersion(v) != PageState::getVersion(local_version)) {
            auto &pages = make_flush_pages();
            mmu::invlpg_tlb_local(pages.data(), pages.size());
            local_version = v;
        }
        if (ps.tryLockS(v)) {
            local_version = v;
            return s;
        }
        _mm_pause();
    }
}

void validateFlushTlb(PageState &ps, u64 &local_version, const char *page_addr, u64 page_size) {
    u64 cur = ps.load();
    if (PageState::getVersion(cur) == PageState::getVersion(local_version)) return;
    static constexpr u64 k = 4096;
    uintptr_t start = reinterpret_cast<uintptr_t>(page_addr) & ~(k - 1);
    uintptr_t end = reinterpret_cast<uintptr_t>(page_addr) + page_size;
    std::vector<void *> pages;
    pages.reserve((end - start + k - 1) / k);
    for (uintptr_t p = start; p < end; p += k)
        pages.push_back(reinterpret_cast<void *>(p));
    mmu::invlpg_tlb_local(pages.data(), pages.size());
    local_version = cur;
}

PageDirectory *OsvUCacheFileSystem::StorePageDirectory(const duckdb::string &path,
                                                        PageDirectory dir) {
    std::lock_guard<std::mutex> lk(vma_mu_);
    auto it = page_dirs_.find(path);
    if (it != page_dirs_.end())
        return &it->second;
    size_t num_cpus = sched::cpus.size();
    size_t num_pages = dir.pages.size();
    dir.thread_versions.assign(num_cpus, std::vector<u64>(num_pages, PageState::NO_VERSION));

    dir.offset_index.resize(num_pages);
    std::iota(dir.offset_index.begin(), dir.offset_index.end(), 0u);
    std::sort(dir.offset_index.begin(), dir.offset_index.end(),
        [&](u32 a, u32 b) { return dir.pages[a].offset < dir.pages[b].offset; });

    auto [ins, _] = page_dirs_.emplace(path, std::move(dir));
    PageDirectory &dir_ref = ins->second;

    ucache::VMA *vma = ucache::uCacheManager->mmap(path.c_str(), 0, mmu::page_size);
    vma->options.user_data = &dir_ref;
    dir_ref.vma = vma;

    // Records the ablation variant in the log.
    if (page_dirs_.size() == 1) {
        std::printf("[ablation] step=%d page_locks=%d page_evict=%d page_rs=%d prefetch=%d "
                    "no_tlb_shootdown=%d zero_copy=%d keep_header_frames=%d\n",
                    DUCKDB_ABLATION_STEP, DUCKDB_PAGE_LOCKS, DUCKDB_PAGE_EVICT, DUCKDB_PAGE_RS,
                    DUCKDB_PREFETCH, DUCKDB_NO_TLB_SHOOTDOWN, DUCKDB_ZERO_COPY,
                    DUCKDB_KEEP_HEADER_FRAMES);
    }

#if DUCKDB_PAGE_EVICT
    // We can disable TLB shootdowns because we handle signaling PTE updates with PageEntries.
    vma->options.skipTLBShootdown = DUCKDB_NO_TLB_SHOOTDOWN;

#if DUCKDB_PAGE_RS
    u64 page_bytes = 0;
    for (const PageEntry &pe : dir_ref.pages) page_bytes += pe.page_size();
    dir_ref.dir_id = page_rs::Register(&dir_ref, num_pages, page_bytes,
                                       ucache::uCacheManager->totalPhysSize);

    ucache::uCacheManager->setEvictionPolicy(vma, page_rs_evict_policy);
#else
    ucache::uCacheManager->setEvictionPolicy(vma, duckdb_evict_policy);
#endif

    // We choose victims among PageEntries ourselves, so uCache's resident set is never read.
    static ucache::NullResidentSet null_rs;
    vma->residentSet = &null_rs;
    vma->callback_implems.canBeEvicted_implem = duckdb_canBeEvicted;
    vma->callback_implems.post_EvictedBatch_callback_implem = duckdb_postEvictedBatch;
#endif
    // Without DUCKDB_PAGE_EVICT the vma keeps uCache's own frame eviction and TLB shootdown.

    return &dir_ref;
}

PageDirectory *OsvUCacheFileSystem::GetPageDirectory(const duckdb::string &path) {
    std::lock_guard<std::mutex> lk(vma_mu_);
    auto it = page_dirs_.find(path);
    return it != page_dirs_.end() ? &it->second : nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

OsvUCacheFileHandle &OsvUCacheFileSystem::Cast(duckdb::FileHandle &handle) {
    return handle.Cast<OsvUCacheFileHandle>();
}

ucache::VMA *OsvUCacheFileSystem::GetOrCreateVMA(
    const duckdb::string &path,
    duckdb::FileOpenFlags flags,
    duckdb::optional_ptr<duckdb::FileOpener> opener)
{
    // uCacheManager->mmap() is idempotent: it returns the existing VMA if one
    // already exists for this path.  It also creates a local_ufile (LBA table +
    // direct NVMe) and calls switch_to_poll_mode() on first call.
    //
    // We pass the file_size as req_size so mmap() doesn't need to guess.
    // The inner_fs_ stat is cheap because ext4 is only used for metadata.
    auto inner_handle = inner_fs_->OpenFile(path, flags, opener);
    u64 file_size = static_cast<u64>(inner_fs_->GetFileSize(*inner_handle));

    ucache::VMA *vma = ucache::uCacheManager->mmap(
        path.c_str(), file_size, mmu::page_size, nullptr);
    // We don't set skipTLBShootdown here, instead we set it later when we set up the PageDirectory.
    return vma;
}


// ─────────────────────────────────────────────────────────────────────────────
// OsvUCacheFileSystem - file open
// ─────────────────────────────────────────────────────────────────────────────

duckdb::unique_ptr<duckdb::FileHandle>
OsvUCacheFileSystem::OpenFile(const duckdb::string &path,
                              duckdb::FileOpenFlags flags,
                              duckdb::optional_ptr<duckdb::FileOpener> opener)
{
    if (ucache::uCacheManager == nullptr || ucache::uCacheManager->totalPhysSize == 0) {
        throw duckdb::IOException(
            "OsvUCacheFileSystem: cache not initialised - call osv_ucache_init() first");
    }

    std::lock_guard<std::mutex> lk(vma_mu_);
    ucache::VMA *vma = GetOrCreateVMA(path, flags, opener);

    // Keep a thin inner handle for metadata (GetLastModifiedTime, etc.).
    auto inner_handle = inner_fs_->OpenFile(path, flags, opener);
    return duckdb::make_uniq<OsvUCacheFileHandle>(
        *this, path, flags, vma, std::move(inner_handle));
}

duckdb::unique_ptr<OsvCachingFileHandle>
OsvUCacheFileSystem::OpenParquetHandle(
    const duckdb::string &path,
    duckdb::FileOpenFlags flags,
    duckdb::optional_ptr<duckdb::FileOpener> opener)
{
    if (ucache::uCacheManager == nullptr || ucache::uCacheManager->totalPhysSize == 0) {
        throw duckdb::IOException(
            "OsvUCacheFileSystem: cache not initialised - call osv_ucache_init() first");
    }

    std::lock_guard<std::mutex> lk(vma_mu_);
    ucache::VMA *vma = GetOrCreateVMA(path, flags, opener);

    auto inner_handle = inner_fs_->OpenFile(path, flags, opener);
    u64 file_size = static_cast<u64>(vma->file->size);

    auto handle = duckdb::make_uniq<OsvCachingFileHandle>(
        vma, static_cast<const char *>(vma->start),
        file_size, path, std::move(inner_handle), *this);

    auto dir_it = page_dirs_.find(path);
    if (dir_it != page_dirs_.end())
        handle->page_dir = &dir_it->second;

    return handle;
}


void OsvUCacheFileSystem::PreOpen(const duckdb::string &path) {
    std::lock_guard<std::mutex> lk(vma_mu_);
    GetOrCreateVMA(path, duckdb::FileOpenFlags::FILE_FLAGS_READ, nullptr);
}


// ── Cached read path ──────────────────────────────────────────────────────────

void OsvUCacheFileSystem::Read(duckdb::FileHandle &handle, void *buffer,
                               int64_t nr_bytes, duckdb::idx_t location)
{
    auto &h = Cast(handle);
    if (location >= h.file_size || nr_bytes <= 0) {
        return;
    }
    auto actual = static_cast<duckdb::idx_t>(
        std::min(static_cast<u64>(nr_bytes), h.file_size - location));

    std::memcpy(buffer,
                static_cast<char *>(h.vma->start) + location,
                actual);

    if (actual < static_cast<duckdb::idx_t>(nr_bytes)) {
        std::memset(static_cast<char *>(buffer) + actual, 0,
                    static_cast<duckdb::idx_t>(nr_bytes) - actual);
    }
}

int64_t OsvUCacheFileSystem::Read(duckdb::FileHandle &handle, void *buffer,
                                  int64_t nr_bytes)
{
    auto &h = Cast(handle);
    if (h.position >= h.file_size || nr_bytes <= 0) {
        return 0;
    }
    auto actual = static_cast<duckdb::idx_t>(
        std::min(static_cast<u64>(nr_bytes), h.file_size - h.position));

    std::memcpy(buffer,
                static_cast<char *>(h.vma->start) + h.position,
                actual);
    h.position += actual;
    return static_cast<int64_t>(actual);
}

// ── Write path (bypasses cache) ───────────────────────────────────────────────

void OsvUCacheFileSystem::Write(duckdb::FileHandle &handle, void *buffer,
                                int64_t nr_bytes, duckdb::idx_t location)
{
    auto wh = inner_fs_->OpenFile(
        handle.GetPath(),
        duckdb::FileOpenFlags::FILE_FLAGS_WRITE |
        duckdb::FileOpenFlags::FILE_FLAGS_FILE_CREATE_NEW);
    inner_fs_->Write(*wh, buffer, nr_bytes, location);
}

int64_t OsvUCacheFileSystem::Write(duckdb::FileHandle &handle, void *buffer,
                                   int64_t nr_bytes)
{
    auto &h = Cast(handle);
    auto wh = inner_fs_->OpenFile(
        handle.GetPath(),
        duckdb::FileOpenFlags::FILE_FLAGS_WRITE |
        duckdb::FileOpenFlags::FILE_FLAGS_FILE_CREATE_NEW);
    inner_fs_->Seek(*wh, h.position);
    auto written = inner_fs_->Write(*wh, buffer, nr_bytes);
    h.position += static_cast<duckdb::idx_t>(written);
    return written;
}

// ── Seek / position ───────────────────────────────────────────────────────────

void OsvUCacheFileSystem::Seek(duckdb::FileHandle &handle, duckdb::idx_t location) {
    Cast(handle).position = location;
}

void OsvUCacheFileSystem::Reset(duckdb::FileHandle &handle) {
    Cast(handle).position = 0;
}

duckdb::idx_t OsvUCacheFileSystem::SeekPosition(duckdb::FileHandle &handle) {
    return Cast(handle).position;
}

// ── Metadata ──────────────────────────────────────────────────────────────────

int64_t OsvUCacheFileSystem::GetFileSize(duckdb::FileHandle &handle) {
    return static_cast<int64_t>(Cast(handle).file_size);
}

duckdb::timestamp_t
OsvUCacheFileSystem::GetLastModifiedTime(duckdb::FileHandle &handle) {
    auto &h = Cast(handle);
    if (h.inner) {
        return inner_fs_->GetLastModifiedTime(*h.inner);
    }
    return duckdb::timestamp_t::ninfinity();
}

duckdb::string OsvUCacheFileSystem::GetVersionTag(duckdb::FileHandle &handle) {
    auto &h = Cast(handle);
    if (h.inner) {
        return inner_fs_->GetVersionTag(*h.inner);
    }
    return {};
}

duckdb::FileType OsvUCacheFileSystem::GetFileType(duckdb::FileHandle &handle) {
    return duckdb::FileType::FILE_TYPE_REGULAR;
}

void OsvUCacheFileSystem::Truncate(duckdb::FileHandle &handle, int64_t new_size) {
    auto wh = inner_fs_->OpenFile(handle.GetPath(),
                                  duckdb::FileOpenFlags::FILE_FLAGS_WRITE);
    inner_fs_->Truncate(*wh, new_size);
}

void OsvUCacheFileSystem::FileSync(duckdb::FileHandle &handle) {
    (void)handle;
}

bool OsvUCacheFileSystem::Trim(duckdb::FileHandle &handle,
                               duckdb::idx_t offset_bytes,
                               duckdb::idx_t length_bytes) {
    (void)handle; (void)offset_bytes; (void)length_bytes;
    return false;
}

// ── Delegated file-system operations ─────────────────────────────────────────

bool OsvUCacheFileSystem::FileExists(const duckdb::string &filename,
                                     duckdb::optional_ptr<duckdb::FileOpener> opener) {
    return inner_fs_->FileExists(filename, opener);
}

void OsvUCacheFileSystem::RemoveFile(const duckdb::string &filename,
                                     duckdb::optional_ptr<duckdb::FileOpener> opener) {
    inner_fs_->RemoveFile(filename, opener);
}

bool OsvUCacheFileSystem::TryRemoveFile(const duckdb::string &filename,
                                        duckdb::optional_ptr<duckdb::FileOpener> opener) {
    return inner_fs_->TryRemoveFile(filename, opener);
}

void OsvUCacheFileSystem::MoveFile(const duckdb::string &source,
                                   const duckdb::string &target,
                                   duckdb::optional_ptr<duckdb::FileOpener> opener) {
    inner_fs_->MoveFile(source, target, opener);
}

bool OsvUCacheFileSystem::DirectoryExists(const duckdb::string &directory,
                                          duckdb::optional_ptr<duckdb::FileOpener> opener) {
    return inner_fs_->DirectoryExists(directory, opener);
}

void OsvUCacheFileSystem::CreateDirectory(const duckdb::string &directory,
                                          duckdb::optional_ptr<duckdb::FileOpener> opener) {
    inner_fs_->CreateDirectory(directory, opener);
}

void OsvUCacheFileSystem::RemoveDirectory(const duckdb::string &directory,
                                          duckdb::optional_ptr<duckdb::FileOpener> opener) {
    inner_fs_->RemoveDirectory(directory, opener);
}

bool OsvUCacheFileSystem::ListFiles(
    const duckdb::string &directory,
    const std::function<void(const duckdb::string &, bool)> &callback,
    duckdb::FileOpener *opener)
{
    return inner_fs_->ListFiles(directory, callback, opener);
}

bool OsvUCacheFileSystem::IsPipe(const duckdb::string &filename,
                                 duckdb::optional_ptr<duckdb::FileOpener> opener) {
    return inner_fs_->IsPipe(filename, opener);
}

duckdb::vector<duckdb::OpenFileInfo>
OsvUCacheFileSystem::Glob(const duckdb::string &path,
                          duckdb::FileOpener *opener) {
    return inner_fs_->Glob(path, opener);
}

duckdb::string OsvUCacheFileSystem::GetHomeDirectory() {
    return inner_fs_->GetHomeDirectory();
}

duckdb::string OsvUCacheFileSystem::ExpandPath(const duckdb::string &path) {
    return inner_fs_->ExpandPath(path);
}

duckdb::string OsvUCacheFileSystem::PathSeparator(const duckdb::string &path) {
    return inner_fs_->PathSeparator(path);
}

bool OsvUCacheFileSystem::IsPathAbsolute(const duckdb::string &path) {
    return inner_fs_->IsPathAbsolute(path);
}

} // namespace osv_duckdb
