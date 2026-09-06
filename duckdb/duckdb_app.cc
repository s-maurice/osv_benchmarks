/*
 * DuckDB TPC-H benchmark runner.
 *
 * Runs on OSv (entry point: app_main) and Linux (entry point: main).
 *
 * Parameters via environment variables:
 *
 *   TPCH_QUERY=<N|all>          run query N (1-22), or all queries sequentially (default: all)
 *   TPCH_REPEAT=<K>             when TPCH_QUERY=N:   repeat query N K times (default: 1)
 *                               when TPCH_QUERY=all: repeat the full Q1-Q22 sequence K times
 *
 *   -- Linux only --
 *   argv[1] = <N>               total thread count including the main thread
 *                               (default: hardware_concurrency)
 *                               DuckDB worker threads = N-1, main thread on CPU N-1.
 *
 *   -- OSv only --
 *   UCACHE_MEM=<size>           bytes for uCache (default: 50% of RAM)
 *                               accepts a K/M/G suffix (e.g. 4G, 2048M)
 *
 *   -- Both platforms --
 *   DUCKDB_MEM=<size>           DuckDB computation buffer pool
 *                               (default: 80% of UCACHE_MEM on OSv,
 *                                or 40% of total RAM on Linux)
 *   DUCKDB_FILE_CACHE=<size|0>  DuckDB external file cache cap.
 *                               OSv default: 0 (disabled - uCache handles I/O).
 *                               Linux default: unset (enabled, no cap).
 *                               A non-zero size enables the cache and limits it
 *                               to that many bytes; the amount is added on top
 *                               of DUCKDB_MEM when computing max_memory.
 *                               e.g. DUCKDB_FILE_CACHE=4G
 *
 * On OSv, built into the kernel via:  just build-duckdb
 * On Linux, built via:                make linux   (in benchmarks/duckdb/)
 */

// ── Platform-specific includes ───────────────────────────────────────────────

#ifdef __OSV__
# include <osv/application.hh>
# include "osv_ucache_file_system.hpp"
#else
# include <sys/sysinfo.h>
# include <cstdint>
    typedef uint64_t u64;
#endif

#include <pthread.h>
#include <thread>
#include <sched.h>

#include <duckdb.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/storage/external_file_cache.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <time.h>

#include "tpch_queries.hpp"

// ── Helpers ──────────────────────────────────────────────────────────────────

// Pin the calling thread to cpu_id.
// DuckDB worker threads are pinned to CPUs 0..(N-2); this pins the main
// application thread to CPU N-1 so it doesn't migrate during NVMe I/O.
static void pin_this_thread(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

#ifdef __OSV__
// Benchmark-wide peak memory stats, updated after each query.
static u64    g_bm_peak_used       = 0;
static size_t g_bm_min_huge_blocks = ~(size_t)0;
#endif

// Runs qnum once; on OSv silently tracks peak memory via a background sampler
// and prints a single result line.  Returns elapsed ms, or -1.0 on error.
static double run_query(duckdb::Connection &con, int qnum) {
    const char *sql = tpch::kQueries[qnum];
    printf("\n=== TPC-H Q%02d ===\n", qnum);

#ifdef __OSV__
    ucache::reset_io_stats();
    u64 peak_used;
    size_t min_hugeblk;
    {
        u64 pfree  = ucache::stat_free_phys_mem();
        u64 ptotal = ucache::stat_total_phys_mem();
        peak_used   = ptotal - pfree;
        min_hugeblk = ucache::stat_free_huge_blocks();
    }
    std::atomic<bool> query_done{false};
    std::thread mem_sampler([&peak_used, &min_hugeblk, &query_done]() {
        while (!query_done.load(std::memory_order_relaxed)) {
            u64 pfree  = ucache::stat_free_phys_mem();
            u64 ptotal = ucache::stat_total_phys_mem();
            u64 used   = ptotal - pfree;
            if (used > peak_used) peak_used = used;
            size_t hb = ucache::stat_free_huge_blocks();
            if (hb < min_hugeblk) min_hugeblk = hb;
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
    });
#endif

    double t0     = now_ms();
    auto result   = con.Query(sql);
    double elapsed = now_ms() - t0;

#ifdef __OSV__
    query_done.store(true, std::memory_order_relaxed);
    mem_sampler.join();
    // Final sample after query completes.
    {
        u64 pfree  = ucache::stat_free_phys_mem();
        u64 ptotal = ucache::stat_total_phys_mem();
        u64 used   = ptotal - pfree;
        if (used > peak_used) peak_used = used;
        size_t hb = ucache::stat_free_huge_blocks();
        if (hb < min_hugeblk) min_hugeblk = hb;
    }
    if (peak_used   > g_bm_peak_used)       g_bm_peak_used       = peak_used;
    if (min_hugeblk < g_bm_min_huge_blocks) g_bm_min_huge_blocks = min_hugeblk;
#endif

    if (result->HasError()) {
        printf("  ERROR: %s\n", result->GetError().c_str());
        return -1.0;
    }

    printf("  rows=%" PRIu64 "  time=%.1f ms", (uint64_t)result->RowCount(), elapsed);
#ifdef __OSV__
    {
        const double GiB = 1024.0 * 1024.0 * 1024.0;
        printf("  peak_used=%.2f GiB  min_huge_blocks=%zu",
               peak_used / GiB, min_hugeblk);
    }
#endif
    printf("\n");
    // TPCH_PRINT=1 dumps result rows
    if (getenv("TPCH_PRINT")) {
        printf("%s", result->ToString().c_str());
    }
#ifdef __OSV__
    ucache::print_llfree_stats();
    ucache::print_stats();
#endif
    return elapsed;
}

// ── Entry point ──────────────────────────────────────────────────────────────

#ifdef __OSV__
extern "C" int app_main(int /*argc*/, char** /*argv*/)
#else
int main(int argc, char** argv)
#endif
{
    printf("DuckDB version: %s\n", duckdb::DuckDB::LibraryVersion());

#ifndef __OSV__
    // argv[1]: total thread count including the main thread.
    // DuckDB worker threads = nthreads - 1; main thread is pinned to CPU nthreads-1.
    int nthreads = (int)std::thread::hardware_concurrency();
    if (argc >= 2) {
        int n = atoi(argv[1]);
        if (n >= 1) nthreads = n;
    }
    printf("Threads: %d (main + %d DuckDB workers)\n", nthreads, nthreads - 1);
#endif

    const char *query_env  = getenv("TPCH_QUERY");
    const char *repeat_env = getenv("TPCH_REPEAT");

    // query_num == 0 means "all" (run queries 1-22 in order).
    bool run_all   = !query_env || strcmp(query_env, "all") == 0;
    int  query_num = run_all ? 0 : atoi(query_env);
    int  repeat    = repeat_env ? atoi(repeat_env) : 1;

    if (repeat < 1) repeat = 1;

    // Parse a size string with optional K/M/G suffix into bytes.
    auto parse_mem = [](const char *s) -> u64 {
        char *end;
        u64 v = strtoull(s, &end, 10);
        if      (*end == 'K' || *end == 'k') v <<= 10;
        else if (*end == 'M' || *end == 'm') v <<= 20;
        else if (*end == 'G' || *end == 'g') v <<= 30;
        return v;
    };

    // ── Memory sizing ─────────────────────────────────────────────────────────

#ifdef __OSV__
    // Initialise uCache (OSv's NVMe page cache).
    const char *ucache_env = getenv("UCACHE_MEM");
    u64 ucache_mem = ucache_env ? parse_mem(ucache_env) : 0;
    u64 total_mem  = ucache::stat_total_phys_mem();
    u64 effective_ucache = ucache_mem ? ucache_mem : total_mem / 2;
    printf("uCache: %s (%.1f GiB)\n",
           ucache_env ? ucache_env : "50% of RAM",
           effective_ucache / (1024.0 * 1024 * 1024));
    const int evict_batch = getenv("UCACHE_EVICT_BATCH") ? atoi(getenv("UCACHE_EVICT_BATCH")) : 2048;
    const int prefetch_batch = getenv("UCACHE_PREFETCH_BATCH") ? atoi(getenv("UCACHE_PREFETCH_BATCH")) : 2048;
    osv_duckdb::osv_ucache_init(ucache_mem, evict_batch, prefetch_batch);
#else
    struct sysinfo si = {};
    sysinfo(&si);
    u64 total_mem        = (u64)si.totalram * si.mem_unit;
    u64 effective_ucache = total_mem / 2;   // used only as baseline for DUCKDB_MEM default
#endif

    const char *duckdb_env = getenv("DUCKDB_MEM");
    u64 duckdb_mem = duckdb_env ? parse_mem(duckdb_env)
                                : (effective_ucache * 4) / 5;
    printf("DuckDB buffer pool: %s (%.1f GiB)\n",
           duckdb_env ? duckdb_env : "80% of RAM baseline",
           duckdb_mem / (1024.0 * 1024 * 1024));

    // DUCKDB_FILE_CACHE:
    //   OSv default - disabled (uCache is the I/O cache, no redundant copy needed).
    //   Linux default - enabled, no cap (DuckDB's CachingFileSystem is the cache).
    //   0 / "off"   → disable on either platform.
    //   <size>      → enable with that byte cap; added to max_memory.
    const char *file_cache_env = getenv("DUCKDB_FILE_CACHE");
#ifdef __OSV__
    const bool file_cache_disabled = !file_cache_env ||
                                     strcmp(file_cache_env, "0")   == 0 ||
                                     strcmp(file_cache_env, "off") == 0;
#else
    const bool file_cache_disabled = file_cache_env &&
                                     (strcmp(file_cache_env, "0")   == 0 ||
                                      strcmp(file_cache_env, "off") == 0);
#endif
    const u64 file_cache_mem = (!file_cache_disabled && file_cache_env)
                                ? parse_mem(file_cache_env) : 0;
    if (file_cache_disabled) {
        printf("DuckDB file cache: disabled\n");
    } else if (file_cache_mem > 0) {
        printf("DuckDB file cache: %.1f GiB\n",
               file_cache_mem / (1024.0 * 1024 * 1024));
    } else {
        printf("DuckDB file cache: enabled (no cap)\n");
    }

    // max_memory must cover both the computation pool and the file cache pool.
    u64 total_duckdb_mem = duckdb_mem + file_cache_mem;
    char duckdb_mem_str[32];
    snprintf(duckdb_mem_str, sizeof(duckdb_mem_str), "%" PRIu64 "B", total_duckdb_mem);

    // ── DuckDB configuration ──────────────────────────────────────────────────

    duckdb::DBConfig config;
    config.SetOptionByName("autoload_known_extensions",    duckdb::Value::BOOLEAN(false));
    config.SetOptionByName("autoinstall_known_extensions", duckdb::Value::BOOLEAN(false));
    config.SetOptionByName("max_memory", duckdb::Value(duckdb_mem_str));
    config.SetOptionByName("pin_threads", duckdb::Value("on"));
#ifndef __OSV__
    config.SetOptionByName("threads", duckdb::Value::BIGINT(nthreads - 1));
#endif
    if (file_cache_disabled) {
        config.SetOptionByName("enable_external_file_cache",
                               duckdb::Value::BOOLEAN(false));
    }
#ifdef __OSV__
    auto osv_fs_owned = duckdb::make_uniq<osv_duckdb::OsvUCacheFileSystem>(
        duckdb::FileSystem::CreateLocal());
    osv_duckdb::OsvUCacheFileSystem *osv_fs = osv_fs_owned.get();
    config.file_system = std::move(osv_fs_owned);
#endif

    duckdb::DuckDB db(nullptr, &config);
    duckdb::Connection con(db);

    // if (file_cache_mem > 0) {
    //     duckdb::ExternalFileCache::Get(*con.context).SetMaxBytes(file_cache_mem);
    // }

    // DuckDB pins its N-1 worker threads to CPUs 0..(N-2). The main thread
    // is not a DuckDB worker so it has no affinity and can migrate between CPUs
    // during page-fault-driven NVMe I/O, causing poll_req to use a stale qidx.
    // Pin the main thread to the last CPU in the thread range.
#ifdef __OSV__
    {
        int ncpus = (int)std::thread::hardware_concurrency();
        if (ncpus > 0) pin_this_thread(ncpus - 1);
    }
#else
    if (nthreads > 0) pin_this_thread(nthreads - 1);
#endif

    // ── Run queries ───────────────────────────────────────────────────────────

    if (!run_all && (query_num < 1 || query_num > 22)) {
        printf("Usage: set TPCH_QUERY=<1-22|all> [TPCH_REPEAT=K]\n");
        printf("  TPCH_QUERY=N    run TPC-H query N (1-22)\n");
        printf("  TPCH_QUERY=all  run all queries 1-22 sequentially (default)\n");
        printf("  TPCH_REPEAT=K   single query: repeat K times; all: repeat full sequence K times\n");
        return 1;
    }

#ifdef __OSV__
    // Pre-open uCache VMAs for all files the selected queries will access.
    // mmap() is idempotent - subsequent DuckDB opens return the existing VMA.
    // This isolates VMA creation time from the timed query execution below.
    {
        double t_preopen = now_ms();
        if (run_all) {
            for (int q = 1; q <= 22; q++)
                for (int i = 0; tpch::kQueryFiles[q][i] != nullptr; i++)
                    osv_fs->PreOpen(tpch::kQueryFiles[q][i]);
        } else {
            for (int i = 0; tpch::kQueryFiles[query_num][i] != nullptr; i++)
                osv_fs->PreOpen(tpch::kQueryFiles[query_num][i]);
        }
        printf("  [pre-open] time=%.1f ms\n", now_ms() - t_preopen);

        // Pre-read all parquet files so the page directories are built and excluded
        // from the timings. LIMIT 0 does this without scanning.
        double t_meta = now_ms();
        {
            auto warm_file = [&](const char *path) {
                duckdb::string sql = "SELECT * FROM '";
                sql += path;
                sql += "' LIMIT 0";
                auto r = con.Query(sql);
                if (r->HasError())
                    printf("  [metadata] WARN %s: %s\n", path, r->GetError().c_str());
            };
            if (run_all) {
                for (int q = 1; q <= 22; q++)
                    for (int i = 0; tpch::kQueryFiles[q][i] != nullptr; i++)
                        warm_file(tpch::kQueryFiles[q][i]);
            } else {
                for (int i = 0; tpch::kQueryFiles[query_num][i] != nullptr; i++)
                    warm_file(tpch::kQueryFiles[query_num][i]);
            }
        }
        printf("  [metadata] time=%.1f ms\n", now_ms() - t_meta);

        u64 phys_free  = ucache::stat_free_phys_mem();
        u64 phys_total = ucache::stat_total_phys_mem();
        u64 phys_used  = phys_total - phys_free;
        printf("  [memory]   used=%.2f GiB  free=%.2f GiB  total=%.2f GiB\n",
               phys_used  / (1024.0*1024*1024),
               phys_free  / (1024.0*1024*1024),
               phys_total / (1024.0*1024*1024));
    }
#endif

    if (run_all) {
        for (int rep = 0; rep < repeat; rep++) {
            double t_sum = 0.0;
            if (repeat > 1) printf("\n=== SEQUENCE RUN %d/%d ===\n", rep + 1, repeat);
            for (int q = 1; q <= 22; q++) {
                double t = run_query(con, q);
                if (t >= 0.0) t_sum += t;
            }
            printf("\n=== TOTAL time=%.1f ms ===\n", t_sum);
        }
    } else {
        double t_sum = 0.0;
        for (int r = 0; r < repeat; r++) {
            if (repeat > 1) printf("\n=== RUN %d/%d ===\n", r + 1, repeat);
            double t = run_query(con, query_num);
            if (t < 0.0) break;
            t_sum += t;
        }
        if (repeat > 1) printf("\n=== TOTAL time=%.1f ms ===\n", t_sum);
    }

#ifdef __OSV__
    {
        const double GiB = 1024.0 * 1024.0 * 1024.0;
        size_t min_hb = (g_bm_min_huge_blocks == ~(size_t)0) ? 0 : g_bm_min_huge_blocks;
        printf("\n=== BENCHMARK PEAK: used=%.2f GiB  min_huge_blocks=%zu ===\n",
               g_bm_peak_used / GiB, min_hb);
    }
#endif
    return 0;
}
