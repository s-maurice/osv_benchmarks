#include "osv_ucache_file_system.hpp"
#include "osv_parquet_file_handle.hpp"

#include <duckdb/common/exception.hpp>
#include <duckdb/common/file_open_flags.hpp>

#include <osv/mmu.hh>

#include <cstring>
#include <algorithm>
#include <numeric>
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

static void duckdb_evict_policy(ucache::VMA* vma, u64 nbToEvict, ucache::EvictList el) {
    auto* dir = static_cast<PageDirectory*>(vma->options.user_data);
    if (!dir) return;
    const size_t n = dir->pages.size();
    if (n == 0) return;

    size_t i = dir->evict_cursor.load(std::memory_order_relaxed) % n;
    for (size_t scanned = 0; scanned < n; scanned++, i = (i + 1 == n) ? 0 : i + 1) {
        if ((u64)el.size() >= nbToEvict) break;
        PageEntry& pe = dir->pages[i];

        u64 v = pe.lock.load();
        u64 s = PageState::getState(v);

        if (s >= 1 && s <= PageState::MaxShared) continue;         // pinned by a reader
        if (s == PageState::Locked || s == PageState::Evicted) continue;

        if (s == PageState::Unlocked) {
            pe.lock.tryMark(v);                                     // second chance
            continue;
        }

        assert(s == PageState::Marked);

        // Straddling frames belong to two pages. For now, we only evict "inner"
        // (non-straddling) frames.
        u64 page_end = pe.offset + (u64)pe.page_size();
        u64 first = pe.offset / vma->pageSize;
        u64 last = (page_end - 1) / vma->pageSize;
        u64 inner_first = (pe.offset % vma->pageSize == 0) ? first : first + 1;
        u64 inner_end = (page_end % vma->pageSize == 0) ? last + 1 : last;   // exclusive
        if (inner_first >= inner_end) continue;

        if (!pe.lock.tryLockX(v)) continue;                        // lost race - skip

        bool any = false;
        for (u64 i = inner_first; i < inner_end && i < vma->buffers.size(); i++) {
            if ((u64)el.size() >= nbToEvict) break;
            ucache::Buffer* buf = vma->buffers[i];
            auto* bs = new ucache::BufferSnapshot(vma->nbPages);
            buf->updateSnapshot(bs);
            if (vma->addEvictionCandidate(buf, bs, el))
                any = true;
            else
                delete bs;
        }

        if (!any)
            pe.lock.unlockXSameVersion(v);
        // Otherwise post_EvictedBatch releases the lock after the buffers are evicted.
    }
    dir->evict_cursor.store((u32)i, std::memory_order_relaxed);
}

static bool duckdb_canBeEvicted(ucache::Buffer* /*buf*/) {
    return true;
}

static void duckdb_postEvictedBatch(ucache::Buffer* const* buffers, size_t count) {
    auto* dir = static_cast<PageDirectory*>(buffers[0]->vma->options.user_data);
    if (!dir) return;

    u32 last_pi = UINT32_MAX;
    for (size_t i = 0; i < count; i++) {
        u64 buf_start = (u64)buffers[i]->baseVirt - (u64)buffers[i]->vma->start;
        u64 buf_end = buf_start + buffers[i]->vma->pageSize;

        auto it = std::lower_bound(dir->offset_index.begin(), dir->offset_index.end(), buf_start,
            [&](u32 idx, u64 val) {
                return dir->pages[idx].offset + (u64)dir->pages[idx].page_size() <= val;
            });
        if (it == dir->offset_index.end()) continue;
        u32 pi = *it;
        if (dir->pages[pi].offset >= buf_end) continue;
        if (pi == last_pi) continue;
        last_pi = pi;

        PageEntry& pe = dir->pages[pi];
        u64 v = pe.lock.load();
        assert(PageState::getState(v) == PageState::Locked);
        u64 new_v = PageState::nextVersion(v, PageState::Unlocked);
        bool ok = pe.lock.stateAndVersion.compare_exchange_strong(v, new_v);
        assert(ok);
        // Optimisations: skip the redundant version check and TLB flush that
        // lockSWithFlush/validateFlushTlb would otherwise fire on this CPU's
        // next read of the page.
        dir->thread_versions[GetThreadSlot()][pi] = new_v;
        // {
        //     static constexpr uintptr_t k = 4096;
        //     const char* base = static_cast<const char*>(buffers[i]->vma->start);
        //     uintptr_t start  = reinterpret_cast<uintptr_t>(base + pe.offset) & ~(k - 1);
        //     uintptr_t end    = reinterpret_cast<uintptr_t>(base + pe.offset + (u64)pe.page_size());
        //     std::vector<void*> flush_pages;
        //     flush_pages.reserve((end - start + k - 1) / k);
        //     for (uintptr_t p = start; p < end; p += k)
        //         flush_pages.push_back(reinterpret_cast<void*>(p));
        //     mmu::invlpg_tlb_local(flush_pages.data(), flush_pages.size());
        // }
    }
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
    if (duckdb_rs_ == nullptr)
        duckdb_rs_ = new ucache::HashTableResidentSet(
            ucache::uCacheManager->totalPhysSize / mmu::page_size);
    if (vma->residentSet == ucache::uCacheManager->globalResidentSet)
        vma->residentSet = duckdb_rs_;
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
