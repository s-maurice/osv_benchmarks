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
#include <algorithm>
#include <cstdint>
#include <vector>

#include <duckdb/common/atomic.hpp>
#include "resizable_buffer.hpp"
#include "osv_page_state.hpp"

// ── Forward-declare OSv kernel types ──────────────────────────────────────────
// Only used as pointers; no kernel headers pulled in.
namespace ucache {
class VMA;
class Buffer;
} // namespace ucache

namespace osv_duckdb {

// Forward declarations needed by VmaPinnedBuffer / ScopedPagePin constructors.
struct PageDirectory;
struct PageState;
size_t GetThreadSlot();
void validateFlushTlb(PageState &ps, uint64_t &local_version, const char *page_addr, uint64_t page_size);
void lockSWithFlush(PageState &ps, uint64_t &local_version, const char *page_addr, uint64_t page_size);

// Per-file page index. Built once during metadata load.
struct PageEntry {
    duckdb::idx_t  offset;       // byte offset of page header in file
    duckdb::idx_t  header_size;  // serialised Thrift header size
    duckdb::idx_t  body_size;    // compressed page body size
    PageState      lock;

    duckdb::idx_t  page_size() const { return header_size + body_size; }

    PageEntry(duckdb::idx_t off, duckdb::idx_t hdr_sz, duckdb::idx_t body_sz)
        : offset(off), header_size(hdr_sz), body_size(body_sz)
    {
        lock.stateAndVersion.store(0, std::memory_order_relaxed); // Unlocked, version 0
    }
    PageEntry(PageEntry &&o) noexcept
        : offset(o.offset), header_size(o.header_size), body_size(o.body_size)
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
    duckdb::idx_t                    num_cols = 0;
    // Per-thread last-seen version for each page. [cpu_slot][page_idx], init to NO_VERSION.
    // Allocated in StorePageDirectory once sched::cpus.size() is known.
    std::vector<std::vector<u64>>    thread_versions;
    // Indices into pages[], sorted by pages[i].offset. Built once in StorePageDirectory.
    // Enables O(log N) lookup from a VMA buffer byte range to overlapping PageEntries.
    std::vector<u32>                 offset_index;
};

// A ResizeableBuffer pointing into VMA memory; RAII pins a PageEntry.
struct VmaPinnedBuffer : public duckdb::ResizeableBuffer {
    PageDirectory *dir;
    duckdb::idx_t  page_idx;

    VmaPinnedBuffer(const char *vma_ptr, uint64_t size,
                    PageDirectory *dir_p, duckdb::idx_t idx,
                    const char *vma_base_p)
        : dir(dir_p), page_idx(idx)
    {
        ptr = reinterpret_cast<duckdb::data_ptr_t>(const_cast<char *>(vma_ptr));
        len = size + 1;  // match AllocateBlock(uncompressed_size + 1); required by ByteStreamSplitDecoder
        u64 &local_ver = dir->thread_versions[GetThreadSlot()][page_idx];
        lockSWithFlush(dir->pages[page_idx].lock, local_ver,
                       vma_base_p + dir->pages[page_idx].offset,
                       static_cast<u64>(dir->pages[page_idx].page_size()));
    }

    ~VmaPinnedBuffer() {
        dir->pages[page_idx].lock.unlockS();
    }

    void resize(duckdb::Allocator &, uint64_t) {
        D_ASSERT(false); // must not resize a VMA-backed buffer
    }
};

// RAII pin so eviction cannot unmap the pages during a read.
struct ScopedPagePin {
    PageDirectory *dir;
    duckdb::idx_t  page_idx;
    bool           active = true;

    ScopedPagePin(PageDirectory *dir_p, duckdb::idx_t idx, const char *vma_base_p)
        : dir(dir_p), page_idx(idx)
    {
        u64 &local_ver = dir->thread_versions[GetThreadSlot()][page_idx];
        lockSWithFlush(dir->pages[page_idx].lock, local_ver,
                       vma_base_p + dir->pages[page_idx].offset,
                       static_cast<u64>(dir->pages[page_idx].page_size()));
    }
    ~ScopedPagePin() {
        if (active) dir->pages[page_idx].lock.unlockS();
    }
    void release() {
        if (active) {
            dir->pages[page_idx].lock.unlockS();
            active = false;
        }
    }
    ScopedPagePin(const ScopedPagePin &) = delete;
    ScopedPagePin &operator=(const ScopedPagePin &) = delete;
};

// Thrift transport over a local byte buffer, so header parsing holds no VMA pin.
class MemoryBufferTransport
    : public duckdb_apache::thrift::transport::TVirtualTransport<MemoryBufferTransport> {
public:
    MemoryBufferTransport(const uint8_t *data, uint32_t len)
        : data_(data), len_(len), pos_(0) {}

    uint32_t read(uint8_t *buf, uint32_t len) {
        uint32_t available = len_ - pos_;
        uint32_t to_read = std::min(len, available);
        if (to_read == 0) return 0;
        std::memcpy(buf, data_ + pos_, to_read);
        pos_ += to_read;
        return to_read;
    }

    uint32_t GetPosition() const { return pos_; }

private:
    const uint8_t *data_;
    uint32_t       len_;
    uint32_t       pos_;
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
// the version at which the lock was acquired.
void lockSWithFlush(PageState &ps, u64 &local_version, const char *page_addr, u64 page_size);


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

    void RegisterPrefetch(duckdb::idx_t pos, uint64_t len, bool /*can_merge*/ = true) override {
        handle.RegisterPrefetch(pos, static_cast<duckdb::idx_t>(len));
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
// SimpleFileTransport
//
// Minimal Thrift transport for BuildPageDirectory.  Reads sequentially from a
// plain FileHandle via FileSystem::Read - no VMA, no caching layer.
// Used only during page directory construction (one-time, per file).
// ─────────────────────────────────────────────────────────────────────────────
class SimpleFileTransport
    : public duckdb_apache::thrift::transport::TVirtualTransport<SimpleFileTransport> {
public:
    SimpleFileTransport(duckdb::FileSystem &fs_p, duckdb::FileHandle &handle_p, duckdb::idx_t size_p)
        : fs(fs_p), handle(handle_p), location(0), size(size_p) {}

    uint32_t read(uint8_t *buf, uint32_t len) {
        duckdb::idx_t actual = std::min(static_cast<duckdb::idx_t>(len), size - location);
        if (actual == 0) return 0;
        fs.Read(handle, buf, static_cast<int64_t>(actual), location);
        location += actual;
        return static_cast<uint32_t>(actual);
    }

    void          SetLocation(duckdb::idx_t loc) { location = loc; }
    duckdb::idx_t GetLocation() const            { return location; }
    void          Skip(duckdb::idx_t n)          { location += n; }

private:
    duckdb::FileSystem &fs;
    duckdb::FileHandle &handle;
    duckdb::idx_t       location;
    duckdb::idx_t       size;
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
