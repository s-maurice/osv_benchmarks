#pragma once

// OSv Parquet-reader integration with uCache.
//
// This header is included by DuckDB's parquet extension (cmake build) as well
// as by osv_ucache_file_system.cc (OSv kernel build).  It therefore must NOT
// include any OSv kernel headers (<osv/ucache.hh>, <osv/mmu.hh>, etc.).
//
// OSv kernel types are referenced only as opaque pointers via forward
// declarations.  All code that actually dereferences those types lives in
// osv_ucache_file_system.cc, which is compiled as part of the OSv kernel build
// with the full kernel header tree available.
//
#include <duckdb/common/file_system.hpp>
#include <duckdb/common/file_open_flags.hpp>
#include <duckdb/common/optional_ptr.hpp>
#include <duckdb/common/types/timestamp.hpp>
#include <duckdb/storage/buffer/buffer_handle.hpp>

// Thrift transport base + ParquetTransportBase ABC
#include "thrift/transport/TVirtualTransport.h"
#include "thrift_tools.hpp"

#include <cstring>
#include <cstdio>
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include <duckdb/common/atomic.hpp>
#include "resizable_buffer.hpp"
#include "osv_page_state.hpp"
#include "osv_ablation.hpp"

// ── Forward-declare OSv kernel types ──────────────────────────────────────────
// Only used as pointers; no kernel headers pulled in.
namespace ucache {
class VMA;
class Buffer;
} // namespace ucache

namespace osv_duckdb {

// Forward declarations needed by the VmaPagePin constructor.
struct PageDirectory;
struct PageState;
size_t GetThreadSlot();
void validateFlushTlb(PageState &ps, uint64_t &local_version, const char *page_addr, uint64_t page_size);
uint64_t lockSWithFlush(PageState &ps, uint64_t &local_version, const char *page_addr, uint64_t page_size);
void PageResidentSetInsert(PageDirectory *dir, u32 page_idx);
void enqueue_prefetch(ucache::VMA *vma, duckdb::idx_t pos, duckdb::idx_t len);
void DisableMigration();
void EnableMigration();

// Per-file page index. Built once during metadata load.
struct PageEntry {
    duckdb::idx_t  offset;       // byte offset of page header in file
    // duckdb::idx_t  header_size;  // serialised Thrift header size
    duckdb::idx_t  body_size;    // page size: Thrift header + compressed page body
    PageState      lock;

    duckdb::idx_t  page_size() const { return body_size; }

    PageEntry(duckdb::idx_t off, duckdb::idx_t hdr_sz, duckdb::idx_t body_sz)
        : offset(off), body_size(hdr_sz + body_sz)
    {
        lock.stateAndVersion.store(PageState::sameVersion(0, PageState::Evicted),
                                    std::memory_order_relaxed); // Evicted, version 0
    }
    PageEntry(PageEntry &&o) noexcept
        : offset(o.offset), body_size(o.body_size)
    {
        lock.stateAndVersion.store(o.lock.stateAndVersion.load(), std::memory_order_relaxed);
    }
    PageEntry &operator=(PageEntry &&) = delete;
    PageEntry(const PageEntry &) = delete;
    PageEntry &operator=(const PageEntry &) = delete;
};

struct PageDirectory {
    std::vector<PageEntry>           pages;           // flat across all row groups and columns
    std::vector<duckdb::idx_t>       chunk_start;     // [rg * num_cols + col] -> start in pages
    std::vector<duckdb::idx_t>       chunk_page_count;// [rg * num_cols + col] -> page count for that chunk
    duckdb::idx_t                    num_cols = 0;
    // Per-thread last-seen version for each page. [cpu_slot][page_idx], init to NO_VERSION.
    // Allocated in StorePageDirectory once sched::cpus.size() is known.
    std::vector<std::vector<u64>>    thread_versions;
    // Indices into pages[], sorted by pages[i].offset. Built once in StorePageDirectory.
    // Enables O(log N) lookup from a VMA buffer byte range to overlapping PageEntries.
    std::vector<u32>                 offset_index;
    // Resume point for the eviction sweep.
    std::atomic<u32>                 evict_cursor{0};
    // Set in StorePageDirectory: the file's VMA, and its id in the page resident set.
    ucache::VMA                     *vma = nullptr;
    u32                              dir_id = 0;

    // [row_group_idx] -> bloom filter PageEntry index in pages[], or NO_BLOOM_PAGE.
    static constexpr duckdb::idx_t NO_BLOOM_PAGE = ~static_cast<duckdb::idx_t>(0);
    std::vector<duckdb::idx_t> rg_bloom_page;

    PageDirectory() = default;
    // std::atomic cannot be moved, so we enable moving with 0.
    PageDirectory(PageDirectory &&o) noexcept
        : pages(std::move(o.pages)),
          chunk_start(std::move(o.chunk_start)),
          chunk_page_count(std::move(o.chunk_page_count)),
          num_cols(o.num_cols),
          thread_versions(std::move(o.thread_versions)),
          offset_index(std::move(o.offset_index)),
          evict_cursor(0),
          vma(o.vma),
          dir_id(o.dir_id),
          rg_bloom_page(std::move(o.rg_bloom_page)) {}
};

// S-lock pages[idx] with lockSWithFlush. The reader that makes the page resident adds it to the
// resident set.
inline void LockPageS(PageDirectory *dir, duckdb::idx_t idx, const char *vma_base) {
#if DUCKDB_PAGE_LOCKS
    auto &pe = dir->pages[idx];
    u64 prev = lockSWithFlush(pe.lock, dir->thread_versions[GetThreadSlot()][idx],
                              vma_base + pe.offset, static_cast<u64>(pe.page_size()));
    if (prev == PageState::Evicted) PageResidentSetInsert(dir, static_cast<u32>(idx));
#else
    (void)dir; (void)idx; (void)vma_base;
#endif
}

// Releases the S-lock taken by LockPageS.
inline void UnlockPageS(PageDirectory *dir, duckdb::idx_t idx) {
#if DUCKDB_PAGE_LOCKS
    dir->pages[idx].lock.unlockS();
#else
    (void)dir; (void)idx;
#endif
}

// RAII guard for a PageState S-lock.
// Inherits from ResizeableBuffer, meaning we support zero-copy passing into DuckDB for
// uncompressed + unencrypted pages.
//
// Assumes worker threads are pinned to a CPU (DuckDB's pin_threads setting).
// See DisableMigration()/EnableMigration() below if that assumption ever needs enforcing here.
struct VmaPagePin : public duckdb::ResizeableBuffer {
    PageDirectory *dir;       // owning directory
    duckdb::idx_t  page_idx;  // index of the locked page within dir->pages
    bool           owns_lock; // false when adopting an already-held lock (see already_locked)

    // already_locked: lock already held elsewhere; we only re-validate the TLB, never take/release it.
    // vma_ptr: pointer to the pinned byte range (e.g. vma_base_p + offset), not the VMA base.
    VmaPagePin(const char *vma_ptr, uint64_t size,
               PageDirectory *dir_p, duckdb::idx_t page_idx_p,
               const char *vma_base_p, bool already_locked = false)
        : dir(dir_p), page_idx(page_idx_p), owns_lock(!already_locked)
    {
        ptr = reinterpret_cast<duckdb::data_ptr_t>(const_cast<char *>(vma_ptr));
        len = size + 1;  // match AllocateBlock(uncompressed_size + 1); required by ByteStreamSplitDecoder
#if DUCKDB_PAGE_LOCKS
        if (already_locked) {
            // We may have moved threads since the lock was acquired - re-validate the TLB.
            auto &pe = dir->pages[page_idx];
            validateFlushTlb(pe.lock, dir->thread_versions[GetThreadSlot()][page_idx],
                             vma_base_p + pe.offset, static_cast<u64>(pe.page_size()));
        } else {
            LockPageS(dir, page_idx, vma_base_p);
        }
#else
        (void)vma_base_p;
#endif
    }

    ~VmaPagePin() {
        if (owns_lock) UnlockPageS(dir, page_idx);
    }

    void Prefetch(ucache::VMA *vma) {
        auto &pe = dir->pages[page_idx];
        enqueue_prefetch(vma, pe.offset, pe.page_size());
    }

    void resize(duckdb::Allocator &, uint64_t) {
        D_ASSERT(false); // must not resize a VMA-backed buffer
    }
};

// Non-allocating optional<VmaPagePin>, since VmaPagePin has no default constructor.
class OptionalVmaPagePin {
    alignas(VmaPagePin) unsigned char storage_[sizeof(VmaPagePin)];
    bool engaged_ = false;

    VmaPagePin *ptr() { return reinterpret_cast<VmaPagePin *>(storage_); }

public:
    OptionalVmaPagePin() = default;
    OptionalVmaPagePin(const OptionalVmaPagePin &) = delete;
    OptionalVmaPagePin &operator=(const OptionalVmaPagePin &) = delete;
    ~OptionalVmaPagePin() { Reset(); }

    template <typename... Args>
    void Emplace(Args &&...args) {
        Reset();
        new (storage_) VmaPagePin(std::forward<Args>(args)...);
        engaged_ = true;
    }

    void Reset() {
        if (engaged_) {
            ptr()->~VmaPagePin();
            engaged_ = false;
        }
    }

    VmaPagePin *operator->() { return ptr(); }
    explicit operator bool() const { return engaged_; }
};

// Issue async IO for the [pos, pos+len) byte range into the VMA's uCache buffers.
// Only Uncached buffers in the range are prefetched; already-cached pages are skipped.
void enqueue_prefetch(ucache::VMA *vma, duckdb::idx_t pos, duckdb::idx_t len);

// Returns the current CPU id - used to index into PageDirectory::thread_versions.
size_t GetThreadSlot();

// If local_version is still valid, do nothing. Otherwise flush TLB entries for
// all 4 KiB pages covering [page_addr, page_addr+page_size) and update
// local_version to the current stable version.
void validateFlushTlb(PageState &ps, u64 &local_version, const char *page_addr, u64 page_size);

// Acquire a shared lock on ps, flushing TLB entries for the page's VMA region
// if the version changed since local_version. The flush and the tryLockS CAS
// are tied to the same observed version - if the CAS fails, the loop re-checks
// and re-flushes if needed. On return: lock is held and local_version equals
// the version at which the lock was acquired. Returns the state the lock was taken from.
u64 lockSWithFlush(PageState &ps, u64 &local_version, const char *page_addr, u64 page_size);

// Make sure we don't migrate to another core after validating our TLB entries.
void DisableMigration();
void EnableMigration();


// CachingFileHandle backed by a uCache VMA: reads return a pointer into VMA memory,
// no copy and no buffer-manager allocation.
struct OsvCachingFileHandle {
    ucache::VMA                             *vma;
    const char                              *vma_base;   // = static_cast<char*>(vma->start)
    duckdb::idx_t                            file_size;
    duckdb::string                           path;
    duckdb::unique_ptr<duckdb::FileHandle>   inner;       // for metadata queries
    duckdb::FileSystem                      &fs;
    PageDirectory                           *page_dir = nullptr; // non-owning; set after metadata load

    OsvCachingFileHandle(ucache::VMA *vma_p,
                         const char *base_p,
                         duckdb::idx_t size_p,
                         const duckdb::string &path_p,
                         duckdb::unique_ptr<duckdb::FileHandle> inner_p,
                         duckdb::FileSystem &fs_p)
        : vma(vma_p), vma_base(base_p), file_size(size_p),
          path(path_p), inner(std::move(inner_p)), fs(fs_p)
    {}

    // ── Read API ─────────────────────────────────────────────────────────────
    // Sets buffer to point directly into VMA memory (zero-copy).
    duckdb::BufferHandle Read(duckdb::data_ptr_t &buffer,
                              duckdb::idx_t /*nr_bytes*/,
                              duckdb::idx_t location) {
        buffer = reinterpret_cast<duckdb::data_ptr_t>(
            const_cast<char *>(vma_base + location));
        return duckdb::BufferHandle();   // sentinel - VMA owns the memory
    }

    // ── Prefetch API ─────────────────────────────────────────────────────────
    void RegisterPrefetch(duckdb::idx_t pos, duckdb::idx_t len) {
        enqueue_prefetch(vma, pos, len);
    }
    void PrefetchRegistered() {}  // IO was already issued at RegisterPrefetch time
    void ClearPrefetch() {}       // in-flight IOs drain naturally via checkPipeline

    // ── Metadata ─────────────────────────────────────────────────────────────
    duckdb::idx_t       GetFileSize()        { return file_size; }
    duckdb::string      GetPath()  const     { return path; }
    bool                CanSeek()            { return true; }
    bool                IsRemoteFile() const { return false; }
    // OnDiskFile() = false → DuckDB enables prefetch mode for Parquet reads.
    bool                OnDiskFile()         { return false; }
    duckdb::FileHandle &GetFileHandle()      { return *inner; }

    duckdb::timestamp_t GetLastModifiedTime() {
        if (!inner) return duckdb::timestamp_t::ninfinity();
        return inner->file_system.GetLastModifiedTime(*inner);
    }
    duckdb::string GetVersionTag() {
        if (!inner) return {};
        return inner->file_system.GetVersionTag(*inner);
    }
    bool Validate() const { return true; }
    duckdb::idx_t SeekPosition() { return 0; }
    void Seek(duckdb::idx_t /*location*/) {}
};


// ─────────────────────────────────────────────────────────────────────────────
// OsvThriftFileTransport
//
// A Thrift transport backed by OsvCachingFileHandle.  Reads are direct
// memcpy's from VMA memory; prefetch registration populates the uCache
// prefetch queue; actual prefetch drain is handled via parquet_prefetch_pol.
// ─────────────────────────────────────────────────────────────────────────────
class OsvThriftFileTransport
    : public duckdb_apache::thrift::transport::TVirtualTransport<OsvThriftFileTransport>,
      public duckdb::ParquetTransportBase {
public:
    static constexpr uint64_t PREFETCH_FALLBACK_BUFFERSIZE = 1000000;

    OsvThriftFileTransport(OsvCachingFileHandle &handle_p, bool /*prefetch_mode_p*/)
        : handle(handle_p), location(0), size(handle_p.file_size)
    {}

    // ── TVirtualTransport read ────────────────────────────────────────────────
    uint32_t read(uint8_t *buf, uint32_t len) override {
        duckdb::idx_t actual = std::min(static_cast<duckdb::idx_t>(len),
                                        size - location);
        if (actual == 0) return 0;
        std::memcpy(buf, handle.vma_base + location, actual);
        location += actual;
        return static_cast<uint32_t>(actual);
    }

    // ── ParquetTransportBase interface ───────────────────────────────────────
    void SetLocation(duckdb::idx_t loc) override         { location = loc; }
    duckdb::idx_t GetLocation() const override           { return location; }
    duckdb::idx_t GetSize() const override               { return size; }
    void Skip(duckdb::idx_t skip_count) override         { location += skip_count; }

    // VMA reads are always available - no ReadHead buffering needed.
    duckdb::optional_ptr<duckdb::ReadHead> GetReadHead(duckdb::idx_t /*pos*/) override {
        return nullptr;
    }

    bool HasPrefetch() const override { return !prefetch_registered; }

    // can_merge=false means that this column has a filter, and we should avoid prefetching it.
    void RegisterPrefetch(duckdb::idx_t pos, uint64_t len, bool can_merge = true) override {
        if (can_merge) {
            handle.RegisterPrefetch(pos, static_cast<duckdb::idx_t>(len));
        }
    }
    void FinalizeRegistration() override {}
    void Prefetch(duckdb::idx_t pos, uint64_t len) override {
        RegisterPrefetch(pos, len);
        prefetch_registered = true;
    }
    void PrefetchRegistered() override { prefetch_registered = true; }
    void ClearPrefetch() override {
        prefetch_registered = false;
        handle.ClearPrefetch();
    }

private:
    OsvCachingFileHandle &handle;
    duckdb::idx_t         location;
    duckdb::idx_t         size;
    bool                  prefetch_registered = false;
};


// ─────────────────────────────────────────────────────────────────────────────
// OsvFooterTransport
//
// Reads the Parquet footer into a heap buffer through the inner file handle.
// Used instead of the VMA, which would leave the footer's frames permanently
// stuck in the VMA. Prefetch hints extend the buffer.
// ─────────────────────────────────────────────────────────────────────────────
class OsvFooterTransport
    : public duckdb_apache::thrift::transport::TVirtualTransport<OsvFooterTransport>,
      public duckdb::ParquetTransportBase {
public:
    explicit OsvFooterTransport(OsvCachingFileHandle &handle_p)
        : handle(handle_p), location(0), size(handle_p.file_size), buf_start(handle_p.file_size)
    {}

    uint32_t read(uint8_t *dst, uint32_t len) override {
        if (location >= size) return 0;
        duckdb::idx_t actual = std::min(static_cast<duckdb::idx_t>(len), size - location);
        Load(location);
        std::memcpy(dst, buf.data() + (location - buf_start), actual);
        location += actual;
        return static_cast<uint32_t>(actual);
    }

    void SetLocation(duckdb::idx_t loc) override         { location = loc; }
    duckdb::idx_t GetLocation() const override           { return location; }
    duckdb::idx_t GetSize() const override               { return size; }
    void Skip(duckdb::idx_t skip_count) override         { location += skip_count; }

    duckdb::optional_ptr<duckdb::ReadHead> GetReadHead(duckdb::idx_t /*pos*/) override {
        return nullptr;
    }
    bool HasPrefetch() const override { return false; }

    void RegisterPrefetch(duckdb::idx_t pos, uint64_t /*len*/, bool /*can_merge*/ = true) override {
        Load(pos);
    }
    void FinalizeRegistration() override {}
    void Prefetch(duckdb::idx_t pos, uint64_t /*len*/) override { Load(pos); }
    void PrefetchRegistered() override {}
    void ClearPrefetch() override {}

private:
    // buf always ends at size; extend it back to pos in one read.
    void Load(duckdb::idx_t pos) {
        if (pos >= buf_start) return;
        const duckdb::idx_t extra = buf_start - pos;
        buf.insert(buf.begin(), extra, 0);
        // Chunk reads: OSv doesn't support more than 4 MiB in a single read call.
        static constexpr duckdb::idx_t kMaxRead = 2 * 1024 * 1024;
        for (duckdb::idx_t done = 0; done < extra; done += kMaxRead) {
            handle.inner->Read(buf.data() + done, std::min(kMaxRead, extra - done), pos + done);
        }
        buf_start = pos;
    }

    OsvCachingFileHandle &handle;
    duckdb::idx_t         location;
    duckdb::idx_t         size;
    std::vector<uint8_t>  buf;
    duckdb::idx_t         buf_start;  // file offset of buf[0]
};


// ─────────────────────────────────────────────────────────────────────────────
// SimpleFileTransport
//
// Minimal Thrift transport for BuildPageDirectory.  Reads sequentially from a
// plain FileHandle via FileSystem::Read - no VMA, no caching layer.
// Used only during page directory construction (one-time, per file).
// Buffered read, TCompactProtocol reads 1-5 bytes at a time.
// ─────────────────────────────────────────────────────────────────────────────
class SimpleFileTransport
    : public duckdb_apache::thrift::transport::TVirtualTransport<SimpleFileTransport> {
public:
    static constexpr duckdb::idx_t kBufferSize = 4 * 1024;

    SimpleFileTransport(duckdb::FileSystem &fs_p, duckdb::FileHandle &handle_p, duckdb::idx_t size_p)
        : fs(fs_p), handle(handle_p), location(0), size(size_p),
          buf_start(0), buf_len(0)
    {
        buf.resize(kBufferSize);
    }

    uint32_t read(uint8_t *dst, uint32_t len) {
        // Skip() can make location past EOF. location is unsigned.
        if (location >= size) return 0;
        duckdb::idx_t remaining = std::min(static_cast<duckdb::idx_t>(len), size - location);
        duckdb::idx_t copied = 0;

        while (remaining > 0) {
            if (location < buf_start || location >= buf_start + buf_len) {
                Refill();
                // Report the short read rather than spinning.
                if (buf_len == 0) break;
            }
            duckdb::idx_t avail = buf_start + buf_len - location;
            duckdb::idx_t n = std::min(avail, remaining);
            std::memcpy(dst + copied, buf.data() + (location - buf_start), n);
            location += n;
            copied += n;
            remaining -= n;
        }
        return static_cast<uint32_t>(copied);
    }

    void SetLocation(duckdb::idx_t loc) { location = loc; }
    duckdb::idx_t GetLocation() const { return location; }
    void Skip(duckdb::idx_t n) { location += n; }

private:
    void Refill() {
        // std::min not available.
        const duckdb::idx_t avail = location < size ? size - location : 0;
        const duckdb::idx_t want = avail < kBufferSize ? avail : kBufferSize;
        buf_start = location;
        buf_len = 0;
        if (want == 0) return;

        // Reads want bytes or throws, so buf_len = want is safe.
        fs.Read(handle, buf.data(), static_cast<int64_t>(want), buf_start);
        buf_len = want;
    }

    duckdb::FileSystem  &fs;
    duckdb::FileHandle  &handle;
    duckdb::idx_t        location;
    duckdb::idx_t        size;
    std::vector<uint8_t> buf;
    duckdb::idx_t        buf_start;  // file offset of buf[0]
    duckdb::idx_t        buf_len;    // valid bytes in buf
};


// ─────────────────────────────────────────────────────────────────────────────
// OsvParquetFileSystemBase
//
// Thin mixin interface implemented by OsvUCacheFileSystem.
// Included by DuckDB parquet source (no kernel headers needed) so that
// parquet_reader.cpp can dynamic_cast to this type and call OpenParquetHandle
// without depending on the full OsvUCacheFileSystem definition.
// ─────────────────────────────────────────────────────────────────────────────
class OsvParquetFileSystemBase {
public:
    virtual ~OsvParquetFileSystemBase() = default;

    virtual duckdb::unique_ptr<OsvCachingFileHandle> OpenParquetHandle(
        const duckdb::string &path,
        duckdb::FileOpenFlags flags,
        duckdb::optional_ptr<duckdb::FileOpener> opener = nullptr) = 0;

    virtual PageDirectory *StorePageDirectory(const duckdb::string &path, PageDirectory dir) = 0;
    virtual duckdb::FileSystem &InnerFS() = 0;
};

} // namespace osv_duckdb
