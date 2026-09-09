#include "osv_ucache_file_system.hpp"
#include "osv_parquet_file_handle.hpp"

#include <duckdb/common/exception.hpp>
#include <duckdb/common/file_open_flags.hpp>

#include <osv/mmu.hh>

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <numeric>

namespace osv_duckdb {

// ─────────────────────────────────────────────────────────────────────────────
// enqueue_prefetch - called by OsvCachingFileHandle::RegisterPrefetch()
//
// DuckDB already knows which byte ranges it will need next; this function
// immediately issues async NVMe IO for all pages in [pos, pos+len) that are
// not yet in the cache.  When the page fault arrives later, checkPipeline()
// in uCache::handleFault() finds the buffer in Reading/ReadyToInsert state
// and simply polls for completion rather than starting a new synchronous read.
// ─────────────────────────────────────────────────────────────────────────────
void enqueue_prefetch(ucache::VMA *vma, duckdb::idx_t pos, duckdb::idx_t len) {
    if (len == 0 || ucache::uCacheManager == nullptr) return;

    // Count total in-flight prefetch IOs across all CPUs to avoid saturating the
    // NVMe submission queue.  Only issue up to (prefetch_batch - total_inflight)
    // new IOs per call.
    //
    // per_cpu_inflight_count is a signed int.  It can transiently go negative due
    // to the race between UncachedToPrefetching (which sets the prefetcher field
    // in the PTE, making the buffer visible as Reading) and the fetch_add in
    // prefetch() (which increments the counter a few instructions later).  If a
    // page fault resolves the IO between those two instructions the decrement in
    // ReadyToInsertToCached fires before the increment, yielding -1.  Casting a
    // negative int to u64 wraps to ~2^64, so we clamp each per-CPU value to ≥ 0.
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
}


// ─────────────────────────────────────────────────────────────────────────────
// Eviction callbacks for DuckDB VMA
// ─────────────────────────────────────────────────────────────────────────────

// Frames addEvictionCandidate rejected; retried by drain_eviction_graveyard() at the top of
// the next eviction.
struct GraveyardEntry { ucache::VMA* vma; u64 frame; };
static std::mutex g_eviction_graveyard_mu;
static std::vector<GraveyardEntry> g_eviction_graveyard;

// Vector containing the currently held page locks for eviction.
struct PendingUnlock { PageDirectory* dir; u32 page_idx; };
static thread_local std::vector<PendingUnlock> t_pending_unlock;

// Submits every frame covered by the PageEntry range, straddlers between pages[lo..hi]
// included, and unlocks any page in the run that ends up with no evicted frame.
static void evict_frames_in_run(PageDirectory* dir, ucache::VMA* vma, size_t lo, size_t hi,
                                 ucache::EvictList el, u64 nbToEvict) {
    u64 start = dir->pages[lo].offset;
    u64 end = dir->pages[hi].offset + (u64)dir->pages[hi].page_size();
    // Frames fully inside [start, end). A frame straddling the run's edge is shared with a
    // page we don't hold, so it isn't ours to evict.
    u64 first = start / vma->pageSize;
    u64 last = (end - 1) / vma->pageSize;
    u64 fi = (start % vma->pageSize == 0) ? first : first + 1;
    u64 inner_end = (end % vma->pageSize == 0) ? last + 1 : last;   // exclusive

    std::vector<u64> evicted;
    std::vector<u64> rejected;

    for (; fi < inner_end && (u64)el.size() < nbToEvict && fi < vma->buffers.size(); fi++) {
        ucache::Buffer* buf = vma->buffers[fi];
        auto* bs = new ucache::BufferSnapshot(vma->nbPages);
        buf->updateSnapshot(bs);
        // todo: add non-failable addEvictionCandidate
        if (vma->addEvictionCandidate(buf, bs, el)) {
            evicted.push_back(fi);
        } else {
            delete bs;
            rejected.push_back(fi);
        }
    }

    // A page with no evicted frame will never reach post_EvictedBatch (it will not be
    // submitted for eviction), so unlock it here.
    for (size_t k = lo; k <= hi; k++) {
        PageEntry& pe = dir->pages[k];
        u64 pf_first = pe.offset / vma->pageSize;
        u64 pf_end = (pe.offset + (u64)pe.page_size() - 1) / vma->pageSize + 1;
        bool has_frame = false;
        for (u64 f = pf_first; f < pf_end && !has_frame; f++)
            has_frame = std::find(evicted.begin(), evicted.end(), f) != evicted.end();

        // todo: check if we should unlock into marked
        if (has_frame)
            t_pending_unlock.push_back({dir, (u32)k});
        else
            pe.lock.unlockXSameVersion(pe.lock.load());
    }

    if (!rejected.empty()) {
        // Failing to evict a frame should be _very_ rare. One possible reason is as follows:
        // We had I/O registered for the frame, but we never read from the frame (and thus never
        // blocked on it). Then, we release the Page's lock, and the PageEntry is selected for
        // eviction. We cannot release that frame, as the frame is still being used for I/O.
        std::printf("[evict] %zu frame(s) not evicted\n", rejected.size());
        std::scoped_lock lk(g_eviction_graveyard_mu);
        for (u64 f : rejected) g_eviction_graveyard.push_back({vma, f});
    }
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

    std::vector<u32> held;
    std::vector<GraveyardEntry> requeue;

    for (const GraveyardEntry& e : to_retry) {
        ucache::VMA* vma = e.vma;
        auto* dir = static_cast<PageDirectory*>(vma->options.user_data);
        if (!dir || e.frame >= vma->buffers.size()) continue;

        u64 f_start = e.frame * vma->pageSize;
        u64 f_end = f_start + vma->pageSize;

        // Lock every PageEntry overlapping the frame, all-or-nothing, in ascending offset
        // order so concurrent drains cannot deadlock.
        held.clear();
        bool all_held = true;
        auto it = std::lower_bound(dir->offset_index.begin(), dir->offset_index.end(), f_start,
            [&](u32 idx, u64 val) {
                return dir->pages[idx].offset + (u64)dir->pages[idx].page_size() <= val;
            });
        for (; it != dir->offset_index.end() && dir->pages[*it].offset < f_end; ++it) {
            PageEntry& pe = dir->pages[*it];
            u64 v = pe.lock.load();
            u64 s = PageState::getState(v);
            // A shared count means a reader is inside the page, Locked means another evictor
            // owns it. Either way the frame is not ours this round.
            if ((s != PageState::Unlocked && s != PageState::Marked) || !pe.lock.tryLockX(v)) {
                all_held = false;
                break;
            }
            held.push_back(*it);
        }

        bool evicted = false;
        if (all_held) {
            ucache::Buffer* buf = vma->buffers[e.frame];
            auto* bs = new ucache::BufferSnapshot(vma->nbPages);
            buf->updateSnapshot(bs);
            evicted = vma->addEvictionCandidate(buf, bs, el);
            if (!evicted) delete bs;
        }

        if (evicted) {
            for (u32 pi : held)
                t_pending_unlock.push_back({dir, pi});
        } else {
            for (u32 pi : held)
                dir->pages[pi].lock.unlockXSameVersion(dir->pages[pi].lock.load());
            requeue.push_back(e);
        }
    }

    if (!requeue.empty()) {
        std::scoped_lock lk(g_eviction_graveyard_mu);
        g_eviction_graveyard.insert(g_eviction_graveyard.end(), requeue.begin(), requeue.end());
    }
}

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

        auto end_run = [&]() {
            if (in_run) evict_frames_in_run(dir, vma, run_lo, run_hi, el, nbToEvict);
            in_run = false;
        };

        size_t i = dir->evict_cursor.load(std::memory_order_relaxed) % n;
        for (size_t scanned = 0; scanned < n; scanned++, i = (i + 1 == n) ? 0 : i + 1) {
            if ((u64)el.size() >= nbToEvict) { end_run(); break; }
            PageEntry& pe = dir->pages[i];

            u64 v = pe.lock.load();
            u64 s = PageState::getState(v);

            // check if we are pinned by a reader
            if (s >= 1 && s <= PageState::MaxShared) {
                end_run();
                continue;
            }
            if (s == PageState::Locked || s == PageState::Evicted) {
                end_run();
                continue;
            }

            // second chance policy, mark the PageEntry
            if (s == PageState::Unlocked) {
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
        u64 new_v = pe.lock.unlockXNextVersion(v);

        // Optimisation: skip the redundant version check and TLB flush that
        // lockSWithFlush/validateFlushTlb would otherwise fire on this CPU's
        // next read of the page.
        dir->thread_versions[GetThreadSlot()][p.page_idx] = new_v;
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

// ─────────────────────────────────────────────────────────────────────────────
// Page directory persistence
// ─────────────────────────────────────────────────────────────────────────────

size_t GetThreadSlot() {
    return static_cast<size_t>(sched::cpu::current()->id);
}

void lockSWithFlush(PageState &ps, u64 &local_version, const char *page_addr, u64 page_size) {
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
        if (s == PageState::Locked || s == PageState::Evicted) {
            _mm_pause();
            continue;
        }
        if (PageState::getVersion(v) != PageState::getVersion(local_version)) {
            auto &pages = make_flush_pages();
            mmu::invlpg_tlb_local(pages.data(), pages.size());
            local_version = v;
        }
        if (ps.tryLockS(v)) {
            local_version = v;
            return;
        }
        _mm_pause();
    }
}

void validateFlushTlb(PageState &ps, u64 &local_version, const char *page_addr, u64 page_size) {
    if (ps.validateRead(local_version)) return;
    static constexpr u64 k = 4096;
    uintptr_t start = reinterpret_cast<uintptr_t>(page_addr) & ~(k - 1);
    uintptr_t end = reinterpret_cast<uintptr_t>(page_addr) + page_size;
    std::vector<void *> pages;
    pages.reserve((end - start + k - 1) / k);
    for (uintptr_t p = start; p < end; p += k)
        pages.push_back(reinterpret_cast<void *>(p));
    mmu::invlpg_tlb_local(pages.data(), pages.size());
    local_version = ps.beginRead();
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
    ucache::uCacheManager->setEvictionPolicy(vma, duckdb_evict_policy);
    // We sweep dir->pages ourselves, so the resident set is never read.
    static ucache::NullResidentSet null_rs;
    vma->residentSet = &null_rs;
    vma->callback_implems.canBeEvicted_implem = duckdb_canBeEvicted;
    vma->callback_implems.post_EvictedBatch_callback_implem = duckdb_postEvictedBatch;

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
    vma->options.skipTLBShootdown = true;
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
