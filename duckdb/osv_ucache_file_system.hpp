#pragma once

// OSv uCache-backed FileSystem for DuckDB.
//
// Architecture:
//   OsvUCacheFileSystem::OpenFile()
//     → creates OsvUCacheFileHandle
//         → ucache::VMA  (backed by local_ufile — LBA table + direct NVMe, poll mode)
//
//   OsvUCacheFileSystem::Read(handle, buf, nr_bytes, location)
//     → memcpy(buf, vma->start + location, nr_bytes)
//         → page fault (first access per 2 MiB page)
//         → ucache::handlePageFault → local_ufile::read → NVMe (polled)
//
//   OsvUCacheFileSystem::OpenParquetHandle(path, …)
//     → returns OsvCachingFileHandle (see osv_parquet_file_handle.hpp)
//     → the Parquet reader reads DIRECTLY from VMA memory (zero-copy)
//     → DuckDB calls RegisterPrefetch(pos, len) → enqueue_prefetch()
//         → uCacheManager->prefetch(vma, pl) issues async NVMe IO immediately
//     → page fault arrives later; checkPipeline() polls the in-flight IO to
//       completion rather than starting a new synchronous read.
//
// Prerequisites (must be satisfied before OpenFile is called):
//   • ucache::uCacheManager must be non-null.
//     Call osv_ucache_init() once at start-up to handle this.
//   • OSv's page-fault handler routes faults to uCacheManager->handlePageFault
//     when the faulting address falls inside a registered VMA.

#include <duckdb/common/file_system.hpp>
#include <duckdb/common/file_open_flags.hpp>
#include <duckdb/common/optional_ptr.hpp>
#include <duckdb/common/open_file_info.hpp>

#include <osv/ucache.hh>
#include <osv/mmu.hh>
#include <osv/align.hh>

// Parquet-integration interface and OsvCachingFileHandle definition.
// Must be included AFTER the OSv kernel headers above since
// osv_parquet_file_handle.hpp forward-declares ucache types that are now full.
#include "osv_parquet_file_handle.hpp"

#include <mutex>
#include <unordered_map>
#include <cstring>
#include <algorithm>

namespace osv_duckdb {

// OsvCachingFileHandle and OsvParquetFileSystemBase defined in osv_parquet_file_handle.hpp (already included above).


// ─────────────────────────────────────────────────────────────────────────────
// OsvUCacheFileHandle
//
// A DuckDB FileHandle whose reads go through an OSv uCache VMA.
// The VMA is backed by local_ufile (LBA table, direct NVMe, poll mode).
// ─────────────────────────────────────────────────────────────────────────────
class OsvUCacheFileHandle : public duckdb::FileHandle {
public:
    OsvUCacheFileHandle(duckdb::FileSystem &owner_fs,
                        const duckdb::string &path,
                        duckdb::FileOpenFlags flags,
                        ucache::VMA *vma_ptr,
                        duckdb::unique_ptr<duckdb::FileHandle> inner_handle)
        : duckdb::FileHandle(owner_fs, path, flags),
          vma(vma_ptr),
          inner(std::move(inner_handle)),
          file_size(static_cast<duckdb::idx_t>(vma_ptr->file->size)),
          position(0)
    {}

    void Close() override {
        // VMA lifetime is managed by uCacheManager; inner handle may be reset.
        inner.reset();
    }

    ucache::VMA *vma;                           // owned by uCacheManager->vmas
    duckdb::unique_ptr<duckdb::FileHandle> inner; // for metadata queries only
    duckdb::idx_t file_size;
    duckdb::idx_t position;                     // for sequential reads
};


// ─────────────────────────────────────────────────────────────────────────────
// OsvUCacheFileSystem
//
// A DuckDB FileSystem that transparently caches all file reads through OSv's
// uCache page cache.  All non-read operations are delegated to an inner
// FileSystem (typically LocalFileSystem).
// ─────────────────────────────────────────────────────────────────────────────
class OsvUCacheFileSystem : public duckdb::FileSystem, public OsvParquetFileSystemBase {
public:
    explicit OsvUCacheFileSystem(duckdb::unique_ptr<duckdb::FileSystem> inner)
        : inner_fs_(std::move(inner)) {}

    ~OsvUCacheFileSystem() override = default;

    duckdb::string GetName() const override { return "OsvUCacheFileSystem"; }

    // ── File open ────────────────────────────────────────────────────────────
    duckdb::unique_ptr<duckdb::FileHandle> OpenFile(
        const duckdb::string &path,
        duckdb::FileOpenFlags flags,
        duckdb::optional_ptr<duckdb::FileOpener> opener = nullptr) override;

    // Open a Parquet-optimised handle that reads directly from VMA memory.
    // The handle's prefetch policy is wired to parquet_prefetch_pol so that
    // DuckDB's RegisterPrefetch / PrefetchRegistered calls drive async NVMe I/O
    // through uCache's prefetch hookpoint.
    duckdb::unique_ptr<OsvCachingFileHandle> OpenParquetHandle(
        const duckdb::string &path,
        duckdb::FileOpenFlags flags,
        duckdb::optional_ptr<duckdb::FileOpener> opener = nullptr);

    // Pre-create the uCache VMA for path without opening a DuckDB handle.
    // Idempotent: returns immediately if the VMA already exists.
    // Call before timing a query to isolate VMA creation overhead.
    void PreOpen(const duckdb::string &path);

    // ── Cached read path ─────────────────────────────────────────────────────
    void Read(duckdb::FileHandle &handle, void *buffer, int64_t nr_bytes,
              duckdb::idx_t location) override;
    int64_t Read(duckdb::FileHandle &handle, void *buffer,
                 int64_t nr_bytes) override;

    // ── Write path (bypasses cache) ──────────────────────────────────────────
    void Write(duckdb::FileHandle &handle, void *buffer, int64_t nr_bytes,
               duckdb::idx_t location) override;
    int64_t Write(duckdb::FileHandle &handle, void *buffer,
                  int64_t nr_bytes) override;

    // ── Seek / position ───────────────────────────────────────────────────────
    void Seek(duckdb::FileHandle &handle, duckdb::idx_t location) override;
    void Reset(duckdb::FileHandle &handle) override;
    duckdb::idx_t SeekPosition(duckdb::FileHandle &handle) override;
    bool CanSeek() override { return true; }

    // ── Metadata ─────────────────────────────────────────────────────────────
    int64_t GetFileSize(duckdb::FileHandle &handle) override;
    duckdb::timestamp_t GetLastModifiedTime(duckdb::FileHandle &handle) override;
    duckdb::string GetVersionTag(duckdb::FileHandle &handle) override;
    duckdb::FileType GetFileType(duckdb::FileHandle &handle) override;
    bool OnDiskFile(duckdb::FileHandle &handle) override { return true; }

    void Truncate(duckdb::FileHandle &handle, int64_t new_size) override;
    void FileSync(duckdb::FileHandle &handle) override;
    bool Trim(duckdb::FileHandle &handle, duckdb::idx_t offset_bytes,
              duckdb::idx_t length_bytes) override;

    // ── Delegated file-system operations ────────────────────────────────────
    bool FileExists(const duckdb::string &filename,
                    duckdb::optional_ptr<duckdb::FileOpener> opener = nullptr) override;
    void RemoveFile(const duckdb::string &filename,
                    duckdb::optional_ptr<duckdb::FileOpener> opener = nullptr) override;
    bool TryRemoveFile(const duckdb::string &filename,
                       duckdb::optional_ptr<duckdb::FileOpener> opener = nullptr) override;
    void MoveFile(const duckdb::string &source, const duckdb::string &target,
                  duckdb::optional_ptr<duckdb::FileOpener> opener = nullptr) override;
    bool DirectoryExists(const duckdb::string &directory,
                         duckdb::optional_ptr<duckdb::FileOpener> opener = nullptr) override;
    void CreateDirectory(const duckdb::string &directory,
                         duckdb::optional_ptr<duckdb::FileOpener> opener = nullptr) override;
    void RemoveDirectory(const duckdb::string &directory,
                         duckdb::optional_ptr<duckdb::FileOpener> opener = nullptr) override;
    bool ListFiles(const duckdb::string &directory,
                   const std::function<void(const duckdb::string &, bool)> &callback,
                   duckdb::FileOpener *opener = nullptr) override;
    bool IsPipe(const duckdb::string &filename,
                duckdb::optional_ptr<duckdb::FileOpener> opener = nullptr) override;
    duckdb::vector<duckdb::OpenFileInfo> Glob(const duckdb::string &path,
                                              duckdb::FileOpener *opener = nullptr) override;
    void RegisterSubSystem(duckdb::unique_ptr<duckdb::FileSystem>) override {}
    void RegisterSubSystem(duckdb::FileCompressionType,
                           duckdb::unique_ptr<duckdb::FileSystem>) override {}
    void UnregisterSubSystem(const duckdb::string &) override {}
    duckdb::unique_ptr<duckdb::FileSystem> ExtractSubSystem(const duckdb::string &) override { return nullptr; }
    duckdb::vector<duckdb::string> ListSubSystems() override { return {}; }
    void SetDisabledFileSystems(const duckdb::vector<duckdb::string> &) override {}
    bool SubSystemIsDisabled(const duckdb::string &) override { return false; }
    bool IsDisabledForPath(const duckdb::string &) override { return false; }
    duckdb::string GetHomeDirectory() override;
    duckdb::string ExpandPath(const duckdb::string &path) override;
    duckdb::string PathSeparator(const duckdb::string &path) override;
    bool IsPathAbsolute(const duckdb::string &path) override;

    // Expose the inner filesystem for use by OpenParquetHandle.
    duckdb::FileSystem &InnerFS() override { return *inner_fs_; }

    // Store a page directory for path, built by the Parquet reader after metadata
    // load.  Idempotent: ignored if a directory for this path already exists.
    // Returns a pointer to the (possibly pre-existing) directory.
    PageDirectory *StorePageDirectory(const duckdb::string &path, PageDirectory dir) override;

    // Return the page directory for path, or nullptr if not yet built.
    PageDirectory *GetPageDirectory(const duckdb::string &path);

private:
    static OsvUCacheFileHandle &Cast(duckdb::FileHandle &handle);

    // Get or create the uCache VMA for path.  Returns the VMA.
    // Creates a local_ufile backed VMA via uCacheManager->mmap() on first call.
    ucache::VMA *GetOrCreateVMA(const duckdb::string &path,
                                duckdb::FileOpenFlags flags,
                                duckdb::optional_ptr<duckdb::FileOpener> opener);

    duckdb::unique_ptr<duckdb::FileSystem> inner_fs_;

    std::mutex vma_mu_;
    std::unordered_map<duckdb::string, PageDirectory> page_dirs_;
};


// ─────────────────────────────────────────────────────────────────────────────
// osv_ucache_init
//
// Initialise uCacheManager if it has not already been initialised.
// Call once before creating any OsvUCacheFileSystem instance.
//
// phys_size  – bytes of physical memory to dedicate to the cache.
//              Pass 0 to use 50 % of total system RAM (default).
// batch      – eviction batch size (default 64).
// ─────────────────────────────────────────────────────────────────────────────
inline void osv_ucache_init(u64 phys_size, int batch, int prefetch_batch) {
    if (ucache::uCacheManager == nullptr) {
        ucache::uCacheManager = new ucache::uCache();
    }
    if (ucache::uCacheManager->totalPhysSize == 0) {
        if (phys_size == 0) {
            phys_size = ucache::stat_total_phys_mem() / 2;
        }
        ucache::createCache(phys_size, batch, prefetch_batch);
    }
}

} // namespace osv_duckdb
