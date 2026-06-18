#include <atomic>
#include <algorithm>
#include <cassert>
#include <csignal>
#include <exception>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <vector>
#include <span>
#include <cmath>

#include <errno.h>
#include <libaio.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <immintrin.h>
#include "rte_string.hh"
#include <bitset>

#include <osv/ucache.hh>
#include <osv/mempool.hh>
#include <osv/sched.hh>
#include <cstring>
__thread uint16_t workerThreadId __attribute__ ((tls_model ("initial-exec"))) = 0;
__thread int32_t tpcchistorycounter __attribute__ ((tls_model ("initial-exec"))) = 0;
#include "tpcc/TPCCWorkload.hpp"

using namespace std;

int custom_memcmp(const void* ptr1, const void* ptr2, size_t num){
   return rte_memcmp(ptr1, ptr2, num);
}

void* custom_memcpy(void* dest, const void* src, size_t count){
   return rte_memcpy(dest, src, count);
}

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef u64 PID; // page id type

static const u64 pageSize = 4096;

#define die(msg) do { perror(msg); exit(EXIT_FAILURE); } while(0)

uint64_t rdtsc(){
   return processor::rdtsc();
}

// allocate memory using huge pages
void* allocHuge(size_t size) {
   void* p = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
   madvise(p, size, MADV_HUGEPAGE);
   return p;
}

// use when lock is not free
void yield(u64 counter) {
   _mm_pause();
}

// Adapted from Tabby.
// We assume 48-bit virtual addresses
// Since we use 4k pages, our page virtual-addresses always have 12 tailing zeros.
// We use 48-12 = 36 bits for storing the virt. addr of our page
// We use the remaining 64-36 = 28 bits for the state + version
// We use 8 bits for the state, and 20 for the version.
// In theory, we could also use atomic u128.
struct VMPageState {
   atomic<u64> vmAddrAndVersionAndState;

   static const u64 Unlocked = 0;
   static const u64 MaxShared = 252;
   static const u64 Locked = 253;
   static const u64 Marked = 254;
   static const u64 Evicted = 255; // currently unused, use kTabbyInvalidAddr instead

   static constexpr u64 kTabbyInvalidAddr = 0;
   static constexpr u64 kTabbyBaseVMAddressMask = 0xffff000000000000;
   static constexpr u64 kTabbyBaseVMAddress = 0xffff800000000000;
   static constexpr u64 kTabbyVersionMask = 0x00000000000fffff;
   static constexpr u64 kTabbyStateMask = 0x000000000ff00000;
   static constexpr u64 kTabbyAddressSpaceStart = 0xffff800000000000;

   VMPageState() {}

   // Trim a 48-bit vm address down to 36 bits
   static inline u64 compressNormalVMAddress(u64 addr) { return ((~kTabbyBaseVMAddressMask) & addr) >> 12; }
   // Recover a 48-bit address from a 36-bit address
   // OSv allocates virtual memory in low address space (0x0002...), not high (0xffff...) as Tabby assumes
   static inline u64 decompressVMAddress(u64 addr) { return addr << 12; /* | kTabbyBaseVMAddressMask */ }

   static inline u64 sameVersion(u64 addr, u64 version, u64 newState) { return (compressNormalVMAddress(addr) << 28) | (version) | (newState << 20); }
   static inline u64 nextVersion(u64 addr, u64 version, u64 newState) { return (compressNormalVMAddress(addr) << 28) | ((version + 1) & kTabbyVersionMask) | (newState << 20); }

   bool tryLockX(u64 oldStateAndVersion) {
      ucache::assert_crash(getState(oldStateAndVersion) == Unlocked || getState(oldStateAndVersion) == Marked);
      auto v = oldStateAndVersion;
      auto next = nextVersion(getVMAddress(v), getVersion(v), Locked);
      return vmAddrAndVersionAndState.compare_exchange_strong(v, next);
   }

   void unlockX() {
      ucache::assert_crash(getState() == Locked);
      auto v = vmAddrAndVersionAndState.load();
      auto next = nextVersion(getVMAddress(v), getVersion(v), Unlocked);
      vmAddrAndVersionAndState.store(next, std::memory_order_release);
   }

   // signals that this physical page is no longer in the page table
   // we do not use this currently, prefer unlockXInvalidated
   // that is because this would require us to generate more branches in the switch statements to handle the evicted case
   void unlockXEvicted() {
      ucache::assert_crash(getState() == Locked);
      auto v = vmAddrAndVersionAndState.load();
      auto next = nextVersion(getVMAddress(v), getVersion(v), Evicted);
      vmAddrAndVersionAndState.store(next, std::memory_order_release);
   }

   // used to unlock a page when it is evicted. Converts the state to unlocked.
   // but marks the page as invalidated via the addr being invalidated.
   void unlockXInvalidated() {
      ucache::assert_crash(getState() == Locked);
      auto v = vmAddrAndVersionAndState.load();
      auto next = nextVersion(kTabbyInvalidAddr, getVersion(v), Unlocked);
      vmAddrAndVersionAndState.store(next, std::memory_order_release);
   }

   void downgradeLock() {
      ucache::assert_crash(getState() == Locked);
      auto v = vmAddrAndVersionAndState.load();
      auto next = nextVersion(getVMAddress(v), getVersion(v), 1);
      vmAddrAndVersionAndState.store(next, std::memory_order_release);
   }

   bool tryLockS(u64 oldStateAndVersion) {
      auto v = oldStateAndVersion;
      u64 s = getState(v);

      if (s<MaxShared) {
         auto next = sameVersion(getVMAddress(v), getVersion(v), s+1);
         return vmAddrAndVersionAndState.compare_exchange_strong(v, next);
      }

      if (s==Marked) {
         auto next = nextVersion(getVMAddress(v), getVersion(v), 1);
         return vmAddrAndVersionAndState.compare_exchange_strong(v, next);
      }

      return false;
   }

   void unlockS() {
      while (true) {
         u64 v = vmAddrAndVersionAndState.load();
         u64 state = getState(v);
         ucache::assert_crash(state>0 && state<=MaxShared);

         auto next = sameVersion(getVMAddress(v), getVersion(v), state-1);
         if (vmAddrAndVersionAndState.compare_exchange_strong(v, next))
            return;
      }
   }

   bool tryMark(u64 oldStateAndVersion) {
      auto v = oldStateAndVersion;
      ucache::assert_crash(getState(v)==Unlocked);
      
      auto next = sameVersion(getVMAddress(v), getVersion(v), Marked);   
      return vmAddrAndVersionAndState.compare_exchange_strong(v, next);
   }

   static u64 getState(u64 v) { return (v & kTabbyStateMask) >> 20; };
   static u64 getVersion(u64 v) { return v & kTabbyVersionMask; };
   static u64 getVMAddress(u64 v) { return decompressVMAddress(v >> 28); };

   u64 getState() { return getState(vmAddrAndVersionAndState.load()); }
   u64 getVMAddress() { return getVMAddress(vmAddrAndVersionAndState.load()); }
   u64 getVersion() { return getVersion(vmAddrAndVersionAndState.load()); }

   u64 load() { return vmAddrAndVersionAndState.load(); }

   void operator=(VMPageState&) = delete;
};

struct alignas(4096) Page {
   VMPageState state;
   bool dirty;
};

static const u64 metadataPageId = 0;

struct MetaDataPage {
   // dirty must be at offset sizeof(VMPageState) to match Page::dirty
   alignas(VMPageState) char _state[sizeof(VMPageState)];
   bool dirty;
   PID roots[(pageSize - sizeof(VMPageState) - sizeof(bool)) / sizeof(PID)];

   PID getRoot(unsigned slot) { return roots[slot]; }
};

struct BufferManager {
   static const u64 mb = 1024ull * 1024;
   static const u64 gb = 1024ull * 1024 * 1024;
   u64 virtSize;
   u64 virtCount;
   ucache::VMA* ucache_vma;

   atomic<u64> allocCount;

   Page* virtMem;

   VMPageState& getPageState(PID pid) {
      return virtMem[pid].state;
   }

   BufferManager();
   ~BufferManager() {}

   Page* fixX(PID pid);
   void unfixX(PID pid);
   Page* fixS(PID pid);
   void unfixS(PID pid);

   bool isValidPtr(void* page) { return (page >= virtMem) && (page < (virtMem + virtSize + 16)); }
   bool isValidPID(PID pid) { return pid >= 0 && pid < virtCount; }
   PID toPID(void* page) { ucache::assert_crash(isValidPtr(page)); return reinterpret_cast<Page*>(page) - virtMem; }
   Page* toPtr(PID pid) { ucache::assert_crash(isValidPID(pid)); return virtMem + pid; }

   void flushLocalTLB(u64 address);
   Page* allocPage();
};

BufferManager bm;

struct OLCRestartException {};

template<class T>
struct GuardO {
   PID pid;
   T* ptr;
   u64 version;
   static const u64 moved = ~0ull;

   // constructor
   explicit GuardO(u64 pid) : pid(pid), ptr(reinterpret_cast<T*>(bm.toPtr(pid))) {
      init();
   }

   template<class T2>
   GuardO(u64 pid, GuardO<T2>& parent)  {
      parent.checkVersionAndRestart();
      this->pid = pid;
      ptr = reinterpret_cast<T*>(bm.toPtr(pid));
      init();
   }

   GuardO(GuardO&& other) {
      pid = other.pid;
      ptr = other.ptr;
      version = other.version;
   }

   void init() {
      ucache::assert_crash(pid != moved);
      auto expectedVMAddress = reinterpret_cast<u64>(ptr);

      for (u64 repeatCounter=0; ; repeatCounter++) {
         VMPageState& ps = bm.getPageState(pid);
         auto v = ps.load();

         // we loaded a different page
         if (VMPageState::getVMAddress(v) != expectedVMAddress) {
            bm.flushLocalTLB(expectedVMAddress);
            continue;
         }

         ucache::assert_crash(VMPageState::getState(v) != VMPageState::Evicted);         

         switch (VMPageState::getState(v)) {
            case VMPageState::Marked: {
               u64 newV = VMPageState::sameVersion(VMPageState::getVMAddress(v), VMPageState::getVersion(v), VMPageState::Unlocked);
               if (ps.vmAddrAndVersionAndState.compare_exchange_weak(v, newV)) {
                  version = newV;
                  return;
               }
               break;
            }
            case VMPageState::Locked:
               break;
            case VMPageState::Evicted:
               __builtin_unreachable();
               break;
            default:
               version = v;
               return;
         }
         yield(repeatCounter);
      }
   }

   // move assignment operator
   GuardO& operator=(GuardO&& other) {
      if (pid != moved)
         checkVersionAndRestart();
      pid = other.pid;
      ptr = other.ptr;
      version = other.version;
      other.pid = moved;
      other.ptr = nullptr;
      return *this;
   }

   // assignment operator
   GuardO& operator=(const GuardO&) = delete;

   // copy constructor
   GuardO(const GuardO&) = delete;

   void checkVersionAndRestart() {
      if (pid != moved) {
         auto expectedVMAddress = reinterpret_cast<u64>(ptr);
         VMPageState& ps = bm.getPageState(pid);
         u64 stateAndVersion = ps.load();

         if (version == stateAndVersion) // fast path, nothing changed
            return;

         // same version, same page
         if (VMPageState::getVMAddress(stateAndVersion) == expectedVMAddress &&
             VMPageState::getVersion(stateAndVersion) == VMPageState::getVersion(version)) {
            u64 state = VMPageState::getState(stateAndVersion);

            if (state <= VMPageState::MaxShared)
               return; // ignore shared locks

            if (state == VMPageState::Marked) {
               auto newV = VMPageState::sameVersion(expectedVMAddress, VMPageState::getVersion(stateAndVersion), VMPageState::Unlocked);
               if (ps.vmAddrAndVersionAndState.compare_exchange_weak(stateAndVersion, newV))
                  return; // mark cleared
            }
         }

         if (std::uncaught_exceptions() == 0)
            throw OLCRestartException();
      }
   }

   // destructor
   ~GuardO() noexcept(false) {
      checkVersionAndRestart();
   }

   T* operator->() {
      ucache::assert_crash(pid != moved);
      return ptr;
   }

   void release() {
      checkVersionAndRestart();
      pid = moved;
      ptr = nullptr;
   }
};

template<class T>
struct GuardX {
   PID pid;
   T* ptr;
   static const u64 moved = ~0ull;

   // constructor
   GuardX(): pid(moved), ptr(nullptr) {}

   // constructor
   explicit GuardX(u64 pid) : pid(pid) {
      ptr = reinterpret_cast<T*>(bm.fixX(pid));
      ptr->dirty = true;
   }

   // upgrade
   explicit GuardX(GuardO<T>&& other) : pid(moved) {
      ucache::assert_crash(other.pid != moved);
      auto expectedVMAddress = reinterpret_cast<u64>(other.ptr);

      for (u64 repeatCounter=0; ; repeatCounter++) {
         VMPageState& ps = bm.getPageState(other.pid);
         auto stateAndVersion = ps.load();

         // we loaded a different page
         if (VMPageState::getVMAddress(stateAndVersion) != expectedVMAddress) {
            bm.flushLocalTLB(expectedVMAddress);
            continue;
         }

         // check that version is still the same
         if (VMPageState::getVersion(stateAndVersion) != VMPageState::getVersion(other.version))
            throw OLCRestartException();
         
         ucache::assert_crash(VMPageState::getState(stateAndVersion) != VMPageState::Evicted);         

         auto state = VMPageState::getState(stateAndVersion);
         if ((state == VMPageState::Unlocked) || (state == VMPageState::Marked)) {
            if (ps.tryLockX(stateAndVersion)) {
               pid = other.pid;
               ptr = other.ptr;
               ptr->dirty = true;
               other.pid = moved;
               other.ptr = nullptr;
               return;
            }
         }
         yield(repeatCounter);
      }
   }

   // assignment operator
   GuardX& operator=(const GuardX&) = delete;

   // move assignment operator
   GuardX& operator=(GuardX&& other) {
      if (pid != moved) {
         bm.unfixX(pid);
      }
      pid = other.pid;
      ptr = other.ptr;
      other.pid = moved;
      other.ptr = nullptr;
      return *this;
   }

   // copy constructor
   GuardX(const GuardX&) = delete;

   // destructor
   ~GuardX() {
      if (pid != moved)
         bm.unfixX(pid);
   }

   T* operator->() {
      ucache::assert_crash(pid != moved);
      return ptr;
   }

   void release() {
      if (pid != moved) {
         bm.unfixX(pid);
         pid = moved;
      }
   }
};

template<class T>
struct AllocGuard : public GuardX<T> {
   template <typename ...Params>
   AllocGuard(Params&&... params) {
      // allocPage already fixes the page exclusively
      GuardX<T>::ptr = reinterpret_cast<T*>(bm.allocPage());
      new (GuardX<T>::ptr) T(std::forward<Params>(params)...);
      GuardX<T>::pid = bm.toPID(GuardX<T>::ptr);
   }
};

template<class T>
struct GuardS {
   PID pid;
   T* ptr;
   static const u64 moved = ~0ull;

   // constructor
   explicit GuardS(u64 pid) : pid(pid) {
      ptr = reinterpret_cast<T*>(bm.fixS(pid));
   }

   GuardS(GuardO<T>&& other) : pid(moved) {
      ucache::assert_crash(other.pid != moved);
      auto expectedVMAddress = reinterpret_cast<u64>(other.ptr);

      for (u64 repeatCounter=0; ; repeatCounter++) {
         VMPageState& ps = bm.getPageState(other.pid);
         auto stateAndVersion = ps.load();

         // we loaded a different page
         if (VMPageState::getVMAddress(stateAndVersion) != expectedVMAddress) {
            bm.flushLocalTLB(expectedVMAddress);
            continue;
         }

         // if version doesn't match, abort
         if (VMPageState::getVersion(stateAndVersion) != VMPageState::getVersion(other.version))
            throw OLCRestartException();

         ucache::assert_crash(VMPageState::getState(stateAndVersion) != VMPageState::Evicted);

         if (ps.tryLockS(other.version)) { // XXX: optimize?
            pid = other.pid;
            ptr = other.ptr;
            other.pid = moved;
            other.ptr = nullptr;
            return;
         } else {
            throw OLCRestartException();
         }

         yield(repeatCounter);
      }
   }

   GuardS(GuardS&& other) {
      if (pid != moved)
         bm.unfixS(pid);
      pid = other.pid;
      ptr = other.ptr;
      other.pid = moved;
      other.ptr = nullptr;
   }

   // assignment operator
   GuardS& operator=(const GuardS&) = delete;

   // move assignment operator
   GuardS& operator=(GuardS&& other) {
      if (pid != moved)
         bm.unfixS(pid);
      pid = other.pid;
      ptr = other.ptr;
      other.pid = moved;
      other.ptr = nullptr;
      return *this;
   }

   // copy constructor
   GuardS(const GuardS&) = delete;

   // destructor
   ~GuardS() {
      if (pid != moved)
         bm.unfixS(pid);
   }

   T* operator->() {
      ucache::assert_crash(pid != moved);
      return ptr;
   }

   void release() {
      if (pid != moved) {
         bm.unfixS(pid);
         pid = moved;
      }
   }
};

u64 envOr(const char* env, u64 value) {
   if (getenv(env))
      return atof(getenv(env));
   return value;
}


// we use this to store which PIDs we saw were dirty.
// this is because we only pass one list of eviction candidates to vmcache.
// We need to remember which ones were dirty. We cannot re-read the page dirty state, as that may have changed.
// We only push to this vector if we have the shared lock on these dirty candidates
thread_local std::vector<PID> currentEvictionDirtyLockedCandidates;

// this callback is problematic. The dirty state may have changed.
// We need to ensure that this thread (evicting thread) is holding a shared lock.
bool vmcache_isDirty(ucache::Buffer* buf) {
   auto pid = bm.toPID(buf->baseVirt);
   return std::find(currentEvictionDirtyLockedCandidates.begin(), currentEvictionDirtyLockedCandidates.end(), pid) != currentEvictionDirtyLockedCandidates.end();
}

// do nothing. we clear dirty when we load from disk.
void vmcache_clearDirty(ucache::Buffer* buf) {}

// called after the page has been loaded from disk, before it is made present
// in the VMA. buf->baseVirt points into the linear physical map.
void vmcache_post_io_pre_mapped(ucache::Buffer* buf) {
   // be careful, we cannot use bm::toPID here, because the addr is not in the vma.
   auto page = reinterpret_cast<Page*>(buf->baseVirt);
   VMPageState& ps = page->state;

   auto v = ps.load();
   auto addr = VMPageState::getVMAddress(v);
   auto state = VMPageState::getState(v);
   auto vers = VMPageState::getVersion(v);

   // check that the page was written with at least one read lock held
   ucache::assert_crash(state >= 1 && state <= VMPageState::MaxShared);
   ucache::assert_crash(page->dirty);

   // is there some way to assert that the addr is correct?

   // reset VMPageState
   // possibly doesn't require atomic write?
   // because we are the only thread with access to this in the temporary mapping
   // also: consider resetting version here.
   auto new_ps = VMPageState::nextVersion(addr, vers, VMPageState::Unlocked);
   ps.vmAddrAndVersionAndState.store(new_ps, std::memory_order_release);
   
   // reset dirty (not atomic, similar reasoning?)
   page->dirty = false;

   // we have virtual addrs aliasing into a hardware page
   // uCache performs a full barriered PTE CAS after this anyways
}

// called after buffers are written, but before tlb is flushed
// called again after tlb is flushed
bool vmcache_canBeEvicted(ucache::Buffer* buf) {
   PID pid = bm.toPID(buf->baseVirt);
   auto expectedVMAddress = reinterpret_cast<u64>(bm.toPtr(pid));
   ucache::assert_crash(buf->baseVirt == bm.toPtr(pid));

   // this loop doesn't yield. It can only fail if we access the wrong page.
   // this page should always be in the page table when this function is called.
   for (u64 repeatCounter=0; ; repeatCounter++) {
      VMPageState& ps = bm.getPageState(pid);
      // page access. will pull PTE into local TLB.
      // see comment below.
      auto v = ps.load();

      // we loaded a different page
      if (VMPageState::getVMAddress(v) != reinterpret_cast<u64>(expectedVMAddress)) {
         bm.flushLocalTLB(expectedVMAddress);
         continue;
      }

      // this is reached the second time the this function is called from uCache.
      // The tlb entry has already been invalidated.
      // But the PTE has not been removed yet.
      // Unfortunately, we re-walk the page table here to validate that it is locked.
      // This results in invalid TLB entries on the local thread.
      // This is not a problem for this tabby implementation, as the next access will recognise the invalid addr. and invalidate.
     if(VMPageState::getState(v) == VMPageState::Locked) {
        // we took the X lock last time this function was called.

         // must be done before the PTE is cleared; after that we can no longer access the page to unlock it.
         // invalidating the VM address means stale TLB entries will fail the address-mismatch check.
         ps.unlockXInvalidated();
         return true;
      }

      // this is reached the first time the this function is called from uCache.

      // since this is called by ucache for both clean and dirty candidates, we need to differentiate them.
      // this is inefficient.
      bool isDirtyCandidate = std::find(currentEvictionDirtyLockedCandidates.begin(), currentEvictionDirtyLockedCandidates.end(), pid) != currentEvictionDirtyLockedCandidates.end();
      if (!isDirtyCandidate) {
         // clean
         
         // try to lock clean page candidates
         if (VMPageState::getState(v) == VMPageState::Marked) { // clean candidate
            if(ps.tryLockX(v)) {
               return true;
            } else {
               return false;
            }
         }
         // another thread un-marked this page
         return false;
      } else {
         // dirty
         
         // the page is no longer dirty, as we just wrote it.
         // multiple people may have a shared lock here.
         // but reader threads should not be looking at the dirty flag.
         // if two threads are evicting at the same time, one should win the resident-set removal.
         bm.virtMem[pid].dirty = false;
         
         // try to upgrade lock to X for dirty page candidates
         auto upgradedV = VMPageState::sameVersion(VMPageState::getVMAddress(v), VMPageState::getVersion(v), VMPageState::Locked);
         if((VMPageState::getState(v) == 1) && ps.vmAddrAndVersionAndState.compare_exchange_weak(v, upgradedV)){
            return true;
         } else {
            // someone else has a shared lock on this page too, we cannot evict it.
            ps.unlockS();
            return false;
         }
      }

      // should be unreachable
      ucache::assert_crash(false);
   }
}

// called by chooseEvictionCandidates
void vmcache_evict_policy(ucache::VMA* vma, u64 nbToEvict, ucache::EvictList el) {
   // fresh eviction batch.
   currentEvictionDirtyLockedCandidates.clear();
   currentEvictionDirtyLockedCandidates.reserve(nbToEvict);
      
   while (el.size() < nbToEvict) {
      u64 stillToFind = nbToEvict - el.size();
      u64 id = vma->residentSet->getNextBatch(stillToFind);
      u64 upperBound = id + stillToFind;

      for(u64 i = 0; i < stillToFind; i++) {
         u64 index = (id+i) & vma->residentSet->mask;
         ucache::Buffer* buf = vma->residentSet->getEntry(index);
         if(buf == NULL){
            continue;
         }
         ucache::assert_crash(vma->isValidPtr(buf->baseVirt));
         ucache::BufferSnapshot* bs;
         PID pid = bm.toPID(buf->baseVirt);

         // since the buffer was in the resident set, we know it's in memory
         VMPageState& ps = bm.getPageState(pid);
         auto v = ps.load();

         auto expectedVMAddress = reinterpret_cast<u64>(bm.toPtr(pid));
         ucache::assert_crash(buf->baseVirt == bm.toPtr(pid));

         // we loaded a different page
         if (VMPageState::getVMAddress(v) != expectedVMAddress) [[unlikely]] {
            bm.flushLocalTLB(expectedVMAddress);
            // theoretically we could continue with this page after flushing
            // cerr << "saw wrong page addr during eviction\n";
            continue;
         }

         ucache::assert_crash(VMPageState::getState(v) != VMPageState::Evicted);         

         // find candidates, lock dirty ones in shared mode
         switch (VMPageState::getState(v)) {
            case VMPageState::Marked:
               bs = new ucache::BufferSnapshot(bm.ucache_vma->nbPages);
               buf->updateSnapshot(bs);
               if(bm.virtMem[pid].dirty) {
                  if(ps.tryLockS(v)) {
                     if(vma->addEvictionCandidate(buf, bs, el)) {
                        // locked. remember this.
                        currentEvictionDirtyLockedCandidates.emplace_back(pid);
                     } else {
                        // failed. cleanup.
                        bm.getPageState(pid).unlockS();
                        delete bs;
                     }
                  } else {
                     // could not lock shared. cleanup.
                     delete bs;
                  }
               } else {
                  if(!vma->addEvictionCandidate(buf, bs, el)) {
                     delete bs;
                  }
               }
               break;
            case VMPageState::Unlocked:
               ps.tryMark(v);
               break;
            default:
               break; // skip
         };
      }
   }
}

// // called in Buffer::EvictingToUncached, AFTER local tlb entry for the buffer is cleared.
// // AND after the PTE is cleared.
// void vmcache_release_evicted(ucache::Buffer* buf){
//    // this will re-fault in the page to unlock it.
//    bm.getPageState(bm.toPID(buf->baseVirt)).unlockXEvicted();
// }

// called in Buffer::EvictingToCached. This occurs when we fail to evict.
// ie. called after the first call to canBeEvicted failed.
void vmcache_release_evicted(ucache::Buffer* buf) {
   // clean pages would not be locked.
   // dirty pages have a shared lock.

   PID pid = bm.toPID(buf->baseVirt);
   auto expectedVMAddress = bm.toPtr(pid);
   ucache::assert_crash(buf->baseVirt == expectedVMAddress);

   // free the shared lock for dirty pages.
   bool isDirtyCandidate = std::find(currentEvictionDirtyLockedCandidates.begin(), currentEvictionDirtyLockedCandidates.end(), pid) != currentEvictionDirtyLockedCandidates.end();
   if (isDirtyCandidate) {
      VMPageState& ps = bm.getPageState(pid);
      auto v = ps.load(); 

      // we may need to use a retry loop here.
      // for now, we just validate that the address matches
      ucache::assert_crash(VMPageState::getVMAddress(v) == reinterpret_cast<u64>(expectedVMAddress));

      ps.unlockS();
   }
}

BufferManager::BufferManager() {
   this->virtSize = envOr("VIRTGB", 16)* 1024ul * 1024 * 1024;
   u64 physSize = envOr("PHYSGB", 4)*1024ul*1024*1024;
   this->virtCount = virtSize / pageSize;
   ucache::assert_crash(virtSize>=physSize);
   u64 virtAllocSize = virtSize + (1<<16); // we allocate 64KB extra to prevent segfaults during optimistic reads

   ucache::createCache(physSize, envOr("BATCH", 64));

   ucache::initFile("/nvme/cache", virtAllocSize);

   ucache::VMAOptions vma_options = { .skipTLBShootdown = true, .isolatedPhysicalPagePool = true, .isolatedPhysicalPagePoolSize = physSize };

   ucache_vma = ucache::uCacheManager->mmap("/nvme/cache", virtAllocSize, pageSize, NULL, &vma_options);
   virtMem = (Page*)ucache_vma->start;
   // kTabbyInvalidAddr must decompress to an address outside the VMA
   u64 invalidDecompressed = VMPageState::getVMAddress(VMPageState::sameVersion(VMPageState::kTabbyInvalidAddr, 0, VMPageState::Unlocked));
   ucache::assert_crash(invalidDecompressed < reinterpret_cast<u64>(virtMem) ||
                        invalidDecompressed >= reinterpret_cast<u64>(virtMem + virtCount));
   ucache_vma->callback_implems.isDirty_implem = vmcache_isDirty;
   ucache_vma->callback_implems.clearDirty_implem = vmcache_clearDirty;
   ucache_vma->callback_implems.evict_pol = vmcache_evict_policy;
   ucache_vma->callback_implems.canBeEvicted_implem = vmcache_canBeEvicted;
   ucache_vma->callback_implems.post_io_pre_mapped_callback_implem = vmcache_post_io_pre_mapped;

   // do not call this, because our entry is already cleared. This would re-fault in the page.
   // ucache_vma->callback_implems.post_EvictingToUncached_callback_implem = vmcache_release_evicted;
   
   allocCount = 1; // pid 0 reserved for meta data

   // set up metadata page
   {
      auto pid = metadataPageId; 
      ucache::Buffer* buf = ucache_vma->getBuffer(toPtr(pid));
      ucache::uCacheManager->handleFault(ucache_vma, buf, true);

      memset(toPtr(pid), 0, pageSize);
      toPtr(pid)->dirty = false; // redundant

      // set up the version counter
      auto& ps = getPageState(pid);
      auto next = VMPageState::sameVersion(reinterpret_cast<u64>(toPtr(pid)), 0, VMPageState::Unlocked);
      ps.vmAddrAndVersionAndState.store(next, std::memory_order_release);
   }

   cerr << "integrated_vmcache_tabby " << " virtgb:" << virtSize/gb << " physgb:" << physSize/gb << endl;
}

// allocate a new page and fix it
Page* BufferManager::allocPage() {
   u64 pid = allocCount++;
   if (pid >= virtCount) {
      cerr << "VIRTGB is too low" << endl;
      exit(EXIT_FAILURE);
   }

   // fault the page and clear it.

   // previously handleFault
   // handleFault(pid, true);
   ucache::Buffer* buf = ucache_vma->getBuffer(toPtr(pid));
   ucache::uCacheManager->handleFault(ucache_vma, buf, true);

   memset(toPtr(pid), 0, pageSize);
   toPtr(pid)->dirty = false; // redundant

   // set up the version counter, lock it
   auto& ps = getPageState(pid);
   auto next = VMPageState::sameVersion(reinterpret_cast<u64>(toPtr(pid)), 0, VMPageState::Locked);
   ps.vmAddrAndVersionAndState.store(next, std::memory_order_release);

   return toPtr(pid);
}

Page* BufferManager::fixX(PID pid) {
   auto expectedVMAddress = reinterpret_cast<u64>(toPtr(pid));

   for (u64 repeatCounter=0; ; repeatCounter++) {
      VMPageState& ps = getPageState(pid);
      auto stateAndVersion = ps.load();

      // we loaded a different page
      if (VMPageState::getVMAddress(stateAndVersion) != expectedVMAddress) {
         bm.flushLocalTLB(expectedVMAddress);
         continue;
      }

      ucache::assert_crash(VMPageState::getState(stateAndVersion) != VMPageState::Evicted);
      switch (VMPageState::getState(stateAndVersion)) {
         case VMPageState::Marked: case VMPageState::Unlocked: {
            if (ps.tryLockX(stateAndVersion)){
               return toPtr(pid);
            }
            break;
         }
         case VMPageState::Evicted:
            __builtin_unreachable();
            break;
      }
      yield(repeatCounter);
   }
}

Page* BufferManager::fixS(PID pid) {
   auto expectedVMAddress = reinterpret_cast<u64>(toPtr(pid));

   for (u64 repeatCounter=0; ; repeatCounter++) {
      VMPageState& ps = getPageState(pid);
      auto stateAndVersion = ps.load();

      // we loaded a different page
      if (VMPageState::getVMAddress(stateAndVersion) != expectedVMAddress) {
         bm.flushLocalTLB(expectedVMAddress);
         continue;
      }

      ucache::assert_crash(VMPageState::getState(stateAndVersion) != VMPageState::Evicted);
      switch (VMPageState::getState(stateAndVersion)) {
         case VMPageState::Locked: {
            break;
         }
         default: {
            if (ps.tryLockS(stateAndVersion))
               return toPtr(pid);
         }
      }
      yield(repeatCounter);
   }
}

void BufferManager::unfixS(PID pid) {
   getPageState(pid).unlockS();
}

void BufferManager::unfixX(PID pid) {
   getPageState(pid).unlockX();
}

void BufferManager::flushLocalTLB(u64 addr) {
   ucache::assert_crash(isValidPtr(reinterpret_cast<void*>(addr)));
   auto buffer = ucache_vma->getBuffer(reinterpret_cast<void*>(addr));
   ucache::uCache::flushBufferLocalTLBEntries(buffer);
}

//---------------------------------------------------------------------------

struct BTreeNode;

struct BTreeNodeHeader {
   static const unsigned underFullSize = (pageSize/2) + (pageSize/4);  // merge nodes more empty
   static const u64 noNeighbour = ~0ull;

   struct FenceKeySlot {
      u16 offset;
      u16 len;
   };

   // Raw storage instead of VMPageState: placement new on char[] is a no-op (trivial type,
   // no constructor called), so the lock state set by allocPage() is preserved across
   // BTreeNode construction. Accessed exclusively via bm.getPageState(pid).
   alignas(VMPageState) char _state[sizeof(VMPageState)];
   bool dirty;

   union {
      PID upperInnerNode; // inner
      PID nextLeafNode = noNeighbour; // leaf
   };

   bool hasRightNeighbour() { return nextLeafNode != noNeighbour; }

   FenceKeySlot lowerFence = {0, 0};  // exclusive
   FenceKeySlot upperFence = {0, 0};  // inclusive

   bool hasLowerFence() { return !!lowerFence.len; };

   u16 count = 0;
   bool isLeaf;
   u16 spaceUsed = 0;
   u16 dataOffset = static_cast<u16>(pageSize);
   u16 prefixLen = 0;

   static const unsigned hintCount = 16;
   u32 hint[hintCount];
   u32 padding;

   BTreeNodeHeader(bool isLeaf) : isLeaf(isLeaf) {}
   ~BTreeNodeHeader() {}
};

static unsigned min(unsigned a, unsigned b)
{
   return a < b ? a : b;
}

template <class T>
static T loadUnaligned(void* p)
{
   T x;
   custom_memcpy(&x, p, sizeof(T));
   return x;
}

// Get order-preserving head of key (assuming little endian)
static u32 head(u8* key, unsigned keyLen)
{
   switch (keyLen) {
      case 0:
         return 0;
      case 1:
         return static_cast<u32>(key[0]) << 24;
      case 2:
         return static_cast<u32>(__builtin_bswap16(loadUnaligned<u16>(key))) << 16;
      case 3:
         return (static_cast<u32>(__builtin_bswap16(loadUnaligned<u16>(key))) << 16) | (static_cast<u32>(key[2]) << 8);
      default:
         return __builtin_bswap32(loadUnaligned<u32>(key));
   }
}

struct BTreeNode : public BTreeNodeHeader {
   struct Slot {
      u16 offset;
      u16 keyLen;
      u16 payloadLen;
      union {
         u32 head;
         u8 headBytes[4];
      };
   } __attribute__((packed));
   union {
      Slot slot[(pageSize - sizeof(BTreeNodeHeader)) / sizeof(Slot)];  // grows from front
      u8 heap[pageSize - sizeof(BTreeNodeHeader)];                // grows from back
   };

   static constexpr unsigned maxKVSize = ((pageSize - sizeof(BTreeNodeHeader) - (2 * sizeof(Slot)))) / 4;

   BTreeNode(bool isLeaf) : BTreeNodeHeader(isLeaf) { dirty = true; }

   u8* ptr() { return reinterpret_cast<u8*>(this); }
   bool isInner() { return !isLeaf; }
   span<u8> getLowerFence() { return { ptr() + lowerFence.offset, lowerFence.len}; }
   span<u8> getUpperFence() { return { ptr() + upperFence.offset, upperFence.len}; }
   u8* getPrefix() { return ptr() + lowerFence.offset; } // any key on page is ok

   unsigned freeSpace() { return dataOffset - (reinterpret_cast<u8*>(slot + count) - ptr()); }
   unsigned freeSpaceAfterCompaction() { return pageSize - (reinterpret_cast<u8*>(slot + count) - ptr()) - spaceUsed; }

   bool hasSpaceFor(unsigned keyLen, unsigned payloadLen)
   {
      return spaceNeeded(keyLen, payloadLen) <= freeSpaceAfterCompaction();
   }

   u8* getKey(unsigned slotId) { return ptr() + slot[slotId].offset; }
   span<u8> getPayload(unsigned slotId) { return {ptr() + slot[slotId].offset + slot[slotId].keyLen, slot[slotId].payloadLen}; }

   PID getChild(unsigned slotId) { return loadUnaligned<PID>(getPayload(slotId).data()); }

   // How much space would inserting a new key of len "keyLen" require?
   unsigned spaceNeeded(unsigned keyLen, unsigned payloadLen) {
      return sizeof(Slot) + (keyLen - prefixLen) + payloadLen;
   }

   void makeHint()
   {
      unsigned dist = count / (hintCount + 1);
      for (unsigned i = 0; i < hintCount; i++)
         hint[i] = slot[dist * (i + 1)].head;
   }

   void updateHint(unsigned slotId)
   {
      unsigned dist = count / (hintCount + 1);
      unsigned begin = 0;
      if ((count > hintCount * 2 + 1) && (((count - 1) / (hintCount + 1)) == dist) && ((slotId / dist) > 1))
         begin = (slotId / dist) - 1;
      for (unsigned i = begin; i < hintCount; i++)
         hint[i] = slot[dist * (i + 1)].head;
   }

   void searchHint(u32 keyHead, u16& lowerOut, u16& upperOut)
   {
      if (count > hintCount * 2) {
         u16 dist = upperOut / (hintCount + 1);
         u16 pos, pos2;
         for (pos = 0; pos < hintCount; pos++)
            if (hint[pos] >= keyHead)
               break;
         for (pos2 = pos; pos2 < hintCount; pos2++)
            if (hint[pos2] != keyHead)
               break;
         lowerOut = pos * dist;
         if (pos2 < hintCount)
            upperOut = (pos2 + 1) * dist;
      }
   }

   // lower bound search, foundExactOut indicates if there is an exact match, returns slotId
   u16 lowerBound(span<u8> skey, bool& foundExactOut)
   {
      foundExactOut = false;

      // check prefix
      int cmp = custom_memcmp(skey.data(), getPrefix(), min(skey.size(), prefixLen));
      if (cmp < 0) // key is less than prefix
         return 0;
      if (cmp > 0) // key is greater than prefix
         return count;
      if (skey.size() < prefixLen) // key is equal but shorter than prefix
         return 0;
      u8* key = skey.data() + prefixLen;
      unsigned keyLen = skey.size() - prefixLen;

      // check hint
      u16 lower = 0;
      u16 upper = count;
      u32 keyHead = head(key, keyLen);
      searchHint(keyHead, lower, upper);

      // binary search on remaining range
      while (lower < upper) {
         u16 mid = ((upper - lower) / 2) + lower;
         if (keyHead < slot[mid].head) {
            upper = mid;
         } else if (keyHead > slot[mid].head) {
            lower = mid + 1;
         } else { // head is equal, check full key
            int cmp = custom_memcmp(key, getKey(mid), min(keyLen, slot[mid].keyLen));
            if (cmp < 0) {
               upper = mid;
            } else if (cmp > 0) {
               lower = mid + 1;
            } else {
               if (keyLen < slot[mid].keyLen) { // key is shorter
                  upper = mid;
               } else if (keyLen > slot[mid].keyLen) { // key is longer
                  lower = mid + 1;
               } else {
                  foundExactOut = true;
                  return mid;
               }
            }
         }
      }
      return lower;
   }

   // lowerBound wrapper ignoring exact match argument (for convenience)
   u16 lowerBound(span<u8> key)
   {
      bool ignore;
      return lowerBound(key, ignore);
   }

   // insert key/value pair
   void insertInPage(span<u8> key, span<u8> payload)
   {
      unsigned needed = spaceNeeded(key.size(), payload.size());
      if (needed > freeSpace()) {
         ucache::assert_crash(needed <= freeSpaceAfterCompaction());
         compactify();
      }
      unsigned slotId = lowerBound(key);
      memmove(slot + slotId + 1, slot + slotId, sizeof(Slot) * (count - slotId));
      storeKeyValue(slotId, key, payload);
      count++;
      updateHint(slotId);
   }

   bool removeSlot(unsigned slotId)
   {
      spaceUsed -= slot[slotId].keyLen;
      spaceUsed -= slot[slotId].payloadLen;
      memmove(slot + slotId, slot + slotId + 1, sizeof(Slot) * (count - slotId - 1));
      count--;
      makeHint();
      return true;
   }

   bool removeInPage(span<u8> key)
   {
      bool found;
      unsigned slotId = lowerBound(key, found);
      if (!found)
         return false;
      return removeSlot(slotId);
   }

   void copyNode(BTreeNodeHeader* dst, BTreeNodeHeader* src) {
      u64 ofs = offsetof(BTreeNodeHeader, upperInnerNode);
      custom_memcpy(reinterpret_cast<u8*>(dst)+ofs, reinterpret_cast<u8*>(src)+ofs, sizeof(BTreeNode)-ofs);
   }

   void compactify()
   {
      unsigned should = freeSpaceAfterCompaction();
      static_cast<void>(should);
      BTreeNode tmp(isLeaf);
      tmp.setFences(getLowerFence(), getUpperFence());
      copyKeyValueRange(&tmp, 0, 0, count);
      tmp.upperInnerNode = upperInnerNode;
      copyNode(this, &tmp);
      makeHint();
      ucache::assert_crash(freeSpace() == should);
   }

   // merge right node into this node
   bool mergeNodes(unsigned slotId, BTreeNode* parent, BTreeNode* right)
   {
      if (!isLeaf)
         // TODO: implement inner merge
         return true;

      ucache::assert_crash(right->isLeaf);
      ucache::assert_crash(parent->isInner());
      BTreeNode tmp(isLeaf);
      tmp.setFences(getLowerFence(), right->getUpperFence());
      unsigned leftGrow = (prefixLen - tmp.prefixLen) * count;
      unsigned rightGrow = (right->prefixLen - tmp.prefixLen) * right->count;
      unsigned spaceUpperBound =
         spaceUsed + right->spaceUsed + (reinterpret_cast<u8*>(slot + count + right->count) - ptr()) + leftGrow + rightGrow;
      if (spaceUpperBound > pageSize)
         return false;
      copyKeyValueRange(&tmp, 0, 0, count);
      right->copyKeyValueRange(&tmp, count, 0, right->count);
      PID pid = bm.toPID(this);
      custom_memcpy(parent->getPayload(slotId+1).data(), &pid, sizeof(PID));
      parent->removeSlot(slotId);
      tmp.makeHint();
      tmp.nextLeafNode = right->nextLeafNode;

      copyNode(this, &tmp);
      return true;
   }

   // store key/value pair at slotId
   void storeKeyValue(u16 slotId, span<u8> skey, span<u8> payload)
   {
      // slot
      u8* key = skey.data() + prefixLen;
      unsigned keyLen = skey.size() - prefixLen;
      slot[slotId].head = head(key, keyLen);
      slot[slotId].keyLen = keyLen;
      slot[slotId].payloadLen = payload.size();
      // key
      unsigned space = keyLen + payload.size();
      dataOffset -= space;
      spaceUsed += space;
      slot[slotId].offset = dataOffset;
      ucache::assert_crash(getKey(slotId) >= reinterpret_cast<u8*>(&slot[slotId]));
      custom_memcpy(getKey(slotId), key, keyLen);
      custom_memcpy(getPayload(slotId).data(), payload.data(), payload.size());
   }

   void copyKeyValueRange(BTreeNode* dst, u16 dstSlot, u16 srcSlot, unsigned srcCount)
   {
      if (prefixLen <= dst->prefixLen) {  // prefix grows
         unsigned diff = dst->prefixLen - prefixLen;
         for (unsigned i = 0; i < srcCount; i++) {
            unsigned newKeyLen = slot[srcSlot + i].keyLen - diff;
            unsigned space = newKeyLen + slot[srcSlot + i].payloadLen;
            dst->dataOffset -= space;
            dst->spaceUsed += space;
            dst->slot[dstSlot + i].offset = dst->dataOffset;
            u8* key = getKey(srcSlot + i) + diff;
            custom_memcpy(dst->getKey(dstSlot + i), key, space);
            dst->slot[dstSlot + i].head = head(key, newKeyLen);
            dst->slot[dstSlot + i].keyLen = newKeyLen;
            dst->slot[dstSlot + i].payloadLen = slot[srcSlot + i].payloadLen;
         }
      } else {
         for (unsigned i = 0; i < srcCount; i++)
            copyKeyValue(srcSlot + i, dst, dstSlot + i);
      }
      dst->count += srcCount;
      assert((dst->ptr() + dst->dataOffset) >= reinterpret_cast<u8*>(dst->slot + dst->count));
   }

   void copyKeyValue(u16 srcSlot, BTreeNode* dst, u16 dstSlot)
   {
      unsigned fullLen = slot[srcSlot].keyLen + prefixLen;
      u8 key[fullLen];
      custom_memcpy(key, getPrefix(), prefixLen);
      custom_memcpy(key+prefixLen, getKey(srcSlot), slot[srcSlot].keyLen);
      dst->storeKeyValue(dstSlot, {key, fullLen}, getPayload(srcSlot));
   }

   void insertFence(FenceKeySlot& fk, span<u8> key)
   {
      ucache::assert_crash(freeSpace() >= key.size());
      dataOffset -= key.size();
      spaceUsed += key.size();
      fk.offset = dataOffset;
      fk.len = key.size();
      custom_memcpy(ptr() + dataOffset, key.data(), key.size());
   }

   void setFences(span<u8> lower, span<u8> upper)
   {
      insertFence(lowerFence, lower);
      insertFence(upperFence, upper);
      for (prefixLen = 0; (prefixLen < min(lower.size(), upper.size())) && (lower[prefixLen] == upper[prefixLen]); prefixLen++)
         ;
   }

   void splitNode(BTreeNode* parent, unsigned sepSlot, span<u8> sep)
   {
      ucache::assert_crash(sepSlot > 0);
      ucache::assert_crash(sepSlot < (pageSize / sizeof(PID)));

      BTreeNode tmp(isLeaf);
      BTreeNode* nodeLeft = &tmp;

      AllocGuard<BTreeNode> newNode(isLeaf);
      BTreeNode* nodeRight = newNode.ptr;

      nodeLeft->setFences(getLowerFence(), sep);
      nodeRight->setFences(sep, getUpperFence());

      PID leftPID = bm.toPID(this);
      u16 oldParentSlot = parent->lowerBound(sep);
      if (oldParentSlot == parent->count) {
         ucache::assert_crash(parent->upperInnerNode == leftPID);
         parent->upperInnerNode = newNode.pid;
      } else {
         ucache::assert_crash(parent->getChild(oldParentSlot) == leftPID);
         custom_memcpy(parent->getPayload(oldParentSlot).data(), &newNode.pid, sizeof(PID));
      }
      parent->insertInPage(sep, {reinterpret_cast<u8*>(&leftPID), sizeof(PID)});

      if (isLeaf) {
         copyKeyValueRange(nodeLeft, 0, 0, sepSlot + 1);
         copyKeyValueRange(nodeRight, 0, nodeLeft->count, count - nodeLeft->count);
         nodeLeft->nextLeafNode = newNode.pid;
         nodeRight->nextLeafNode = this->nextLeafNode;
      } else {
         // in inner node split, separator moves to parent (count == 1 + nodeLeft->count + nodeRight->count)
         copyKeyValueRange(nodeLeft, 0, 0, sepSlot);
         copyKeyValueRange(nodeRight, 0, nodeLeft->count + 1, count - nodeLeft->count - 1);
         nodeLeft->upperInnerNode = getChild(nodeLeft->count);
         nodeRight->upperInnerNode = upperInnerNode;
      }
      nodeLeft->makeHint();
      nodeRight->makeHint();
      copyNode(this, nodeLeft);
   }

   struct SeparatorInfo {
      unsigned len;      // len of new separator
      unsigned slot;     // slot at which we split
      bool isTruncated;  // if true, we truncate the separator taking len bytes from slot+1
   };

   unsigned commonPrefix(unsigned slotA, unsigned slotB)
   {
      ucache::assert_crash(slotA < count);
      unsigned limit = min(slot[slotA].keyLen, slot[slotB].keyLen);
      u8 *a = getKey(slotA), *b = getKey(slotB);
      unsigned i;
      for (i = 0; i < limit; i++)
         if (a[i] != b[i])
            break;
      return i;
   }

   SeparatorInfo findSeparator(bool splitOrdered)
   {
      ucache::assert_crash(count > 1);
      if (isInner()) {
         // inner nodes are split in the middle
         unsigned slotId = count / 2;
         return SeparatorInfo{static_cast<unsigned>(prefixLen + slot[slotId].keyLen), slotId, false};
      }

      // find good separator slot
      unsigned bestPrefixLen, bestSlot;

      if (splitOrdered) {
         bestSlot = count - 2;
      } else if (count > 16) {
         unsigned lower = (count / 2) - (count / 16);
         unsigned upper = (count / 2);

         bestPrefixLen = commonPrefix(lower, 0);
         bestSlot = lower;

         if (bestPrefixLen != commonPrefix(upper - 1, 0))
            for (bestSlot = lower + 1; (bestSlot < upper) && (commonPrefix(bestSlot, 0) == bestPrefixLen); bestSlot++)
               ;
      } else {
         bestSlot = (count-1) / 2;
      }


      // try to truncate separator
      unsigned common = commonPrefix(bestSlot, bestSlot + 1);
      if ((bestSlot + 1 < count) && (slot[bestSlot].keyLen > common) && (slot[bestSlot + 1].keyLen > (common + 1)))
         return SeparatorInfo{prefixLen + common + 1, bestSlot, true};

      return SeparatorInfo{static_cast<unsigned>(prefixLen + slot[bestSlot].keyLen), bestSlot, false};
   }

   void getSep(u8* sepKeyOut, SeparatorInfo info)
   {
      custom_memcpy(sepKeyOut, getPrefix(), prefixLen);
      custom_memcpy(sepKeyOut + prefixLen, getKey(info.slot + info.isTruncated), info.len - prefixLen);
   }

   PID lookupInner(span<u8> key)
   {
      unsigned pos = lowerBound(key);
      if (pos == count)
         return upperInnerNode;
      return getChild(pos);
   }
};

static_assert(sizeof(BTreeNode) == pageSize, "btree node size problem");

struct BTree {
   private:

   void trySplit(GuardX<BTreeNode>&& node, GuardX<BTreeNode>&& parent, span<u8> key, unsigned payloadLen);
   void ensureSpace(BTreeNode* toSplit, span<u8> key, unsigned payloadLen);

   public:
   unsigned slotId;
   atomic<bool> splitOrdered;

   BTree();
   ~BTree();

   GuardO<BTreeNode> findLeafO(span<u8> key) {
      GuardO<MetaDataPage> meta(metadataPageId);
      GuardO<BTreeNode> node(meta->getRoot(slotId), meta);
      meta.release();

      while (node->isInner())
         node = GuardO<BTreeNode>(node->lookupInner(key), node);
      return node;
   }

   // point lookup, returns payload len on success, or -1 on failure
   int lookup(span<u8> key, u8* payloadOut, unsigned payloadOutSize) {
      for (u64 repeatCounter=0; ; repeatCounter++) {
         try {
            GuardO<BTreeNode> node = findLeafO(key);
            bool found;
            unsigned pos = node->lowerBound(key, found);
            if (!found)
               return -1;

            // key found, copy payload
            custom_memcpy(payloadOut, node->getPayload(pos).data(), min(node->slot[pos].payloadLen, payloadOutSize));
            return node->slot[pos].payloadLen;
         } catch(const OLCRestartException&) { yield(repeatCounter); }
      }
   }

   template<class Fn>
   bool lookup(span<u8> key, Fn fn) {
      for (u64 repeatCounter=0; ; repeatCounter++) {
         try {
            GuardO<BTreeNode> node = findLeafO(key);
            bool found;
            unsigned pos = node->lowerBound(key, found);
            if (!found)
               return false;

            // key found
            fn(node->getPayload(pos));
            return true;
         } catch(const OLCRestartException&) { yield(repeatCounter); }
      }
   }

   void insert(span<u8> key, span<u8> payload);
   bool remove(span<u8> key);

   template<class Fn>
   bool updateInPlace(span<u8> key, Fn fn) {
      for (u64 repeatCounter=0; ; repeatCounter++) {
         try {
            GuardO<BTreeNode> node = findLeafO(key);
            bool found;
            unsigned pos = node->lowerBound(key, found);
            if (!found)
               return false;

            {
               GuardX<BTreeNode> nodeLocked(move(node));
               fn(nodeLocked->getPayload(pos));
               return true;
            }
         } catch(const OLCRestartException&) { yield(repeatCounter); }
      }
   }

   GuardS<BTreeNode> findLeafS(span<u8> key) {
      for (u64 repeatCounter=0; ; repeatCounter++) {
         try {
            GuardO<MetaDataPage> meta(metadataPageId);
            GuardO<BTreeNode> node(meta->getRoot(slotId), meta);
            meta.release();

            while (node->isInner())
               node = GuardO<BTreeNode>(node->lookupInner(key), node);

            return GuardS<BTreeNode>(move(node));
         } catch(const OLCRestartException&) { yield(repeatCounter); }
      }
   }

   template<class Fn>
   void scanAsc(span<u8> key, Fn fn) {
      GuardS<BTreeNode> node = findLeafS(key);
      bool found;
      unsigned pos = node->lowerBound(key, found);
      for (u64 repeatCounter=0; ; repeatCounter++) { // XXX
         if (pos<node->count) {
            if (!fn(*node.ptr, pos))
               return;
            pos++;
         } else {
            if (!node->hasRightNeighbour())
               return;
            pos = 0;
            node = GuardS<BTreeNode>(node->nextLeafNode);
         }
      }
   }

   template<class Fn>
   void scanDesc(span<u8> key, Fn fn) {
      GuardS<BTreeNode> node = findLeafS(key);
      bool exactMatch;
      int pos = node->lowerBound(key, exactMatch);
      if (pos == node->count) {
         pos--;
         exactMatch = true; // XXX:
      }
      for (u64 repeatCounter=0; ; repeatCounter++) { // XXX
         while (pos>=0) {
            if (!fn(*node.ptr, pos, exactMatch))
               return;
            pos--;
         }
         if (!node->hasLowerFence())
            return;
         node = findLeafS(node->getLowerFence());
         pos = node->count-1;
      }
   }
};

static unsigned btreeslotcounter = 0;

BTree::BTree() : splitOrdered(false) {
   GuardX<MetaDataPage> page(metadataPageId);
   AllocGuard<BTreeNode> rootNode(true);
   slotId = btreeslotcounter++;
   page->roots[slotId] = rootNode.pid;
}

BTree::~BTree() {}

void BTree::trySplit(GuardX<BTreeNode>&& node, GuardX<BTreeNode>&& parent, span<u8> key, unsigned payloadLen)
{

   // create new root if necessary
   if (parent.pid == metadataPageId) {
      MetaDataPage* metaData = reinterpret_cast<MetaDataPage*>(parent.ptr);
      AllocGuard<BTreeNode> newRoot(false);
      newRoot->upperInnerNode = node.pid;
      metaData->roots[slotId] = newRoot.pid;
      parent = move(newRoot);
   }

   // split
   BTreeNode::SeparatorInfo sepInfo = node->findSeparator(splitOrdered.load());
   u8 sepKey[sepInfo.len];
   node->getSep(sepKey, sepInfo);

   if (parent->hasSpaceFor(sepInfo.len, sizeof(PID))) {  // is there enough space in the parent for the separator?
      node->splitNode(parent.ptr, sepInfo.slot, {sepKey, sepInfo.len});
      return;
   }

   // must split parent to make space for separator, restart from root to do this
   node.release();
   parent.release();
   ensureSpace(parent.ptr, {sepKey, sepInfo.len}, sizeof(PID));
}

void BTree::ensureSpace(BTreeNode* toSplit, span<u8> key, unsigned payloadLen)
{
   for (u64 repeatCounter=0; ; repeatCounter++) {
      try {
         GuardO<BTreeNode> parent(metadataPageId);
         GuardO<BTreeNode> node(reinterpret_cast<MetaDataPage*>(parent.ptr)->getRoot(slotId), parent);

         while (node->isInner() && (node.ptr != toSplit)) {
            parent = move(node);
            node = GuardO<BTreeNode>(parent->lookupInner(key), parent);
         }
         if (node.ptr == toSplit) {
            if (node->hasSpaceFor(key.size(), payloadLen))
               return; // someone else did split concurrently
            GuardX<BTreeNode> parentLocked(move(parent));
            GuardX<BTreeNode> nodeLocked(move(node));
            trySplit(move(nodeLocked), move(parentLocked), key, payloadLen);
         }
         return;
      } catch(const OLCRestartException&) { yield(repeatCounter); }
   }
}

void BTree::insert(span<u8> key, span<u8> payload)
{
   ucache::assert_crash((key.size()+payload.size()) <= BTreeNode::maxKVSize);

   for (u64 repeatCounter=0; ; repeatCounter++) {
      try {
         GuardO<BTreeNode> parent(metadataPageId);
         GuardO<BTreeNode> node(reinterpret_cast<MetaDataPage*>(parent.ptr)->getRoot(slotId), parent);

         while (node->isInner()) {
            parent = move(node);
            node = GuardO<BTreeNode>(parent->lookupInner(key), parent);
         }
         ucache::assert_crash(bm.isValidPID(parent.pid));
         ucache::assert_crash(bm.isValidPID(node.pid));
         if (node->hasSpaceFor(key.size(), payload.size())) {
            // only lock leaf
            GuardX<BTreeNode> nodeLocked(move(node));
            parent.release();
            nodeLocked->insertInPage(key, payload);
            return; // success
         }

         // lock parent and leaf
         GuardX<BTreeNode> parentLocked(move(parent));
         GuardX<BTreeNode> nodeLocked(move(node));
         trySplit(move(nodeLocked), move(parentLocked), key, payload.size());
         // insert hasn't happened, restart from root
      } catch(const OLCRestartException&) { yield(repeatCounter); }
   }
}

bool BTree::remove(span<u8> key)
{
   for (u64 repeatCounter=0; ; repeatCounter++) {
      try {
         GuardO<BTreeNode> parent(metadataPageId);
         GuardO<BTreeNode> node(reinterpret_cast<MetaDataPage*>(parent.ptr)->getRoot(slotId), parent);

         u16 pos;
         while (node->isInner()) {
            pos = node->lowerBound(key);
            PID nextPage = (pos == node->count) ? node->upperInnerNode : node->getChild(pos);
            parent = move(node);
            node = GuardO<BTreeNode>(nextPage, parent);
         }

         bool found;
         unsigned slotId = node->lowerBound(key, found);
         if (!found)
            return false;

         unsigned sizeEntry = node->slot[slotId].keyLen + node->slot[slotId].payloadLen;
         if ((node->freeSpaceAfterCompaction()+sizeEntry >= BTreeNodeHeader::underFullSize) && (parent.pid != metadataPageId) && (parent->count >= 2) && ((pos + 1) < parent->count)) {
            // underfull
            GuardX<BTreeNode> parentLocked(move(parent));
            GuardX<BTreeNode> nodeLocked(move(node));
            GuardX<BTreeNode> rightLocked(parentLocked->getChild(pos + 1));
            nodeLocked->removeSlot(slotId);
            if (rightLocked->freeSpaceAfterCompaction() >= (pageSize-BTreeNodeHeader::underFullSize)) {
               if (nodeLocked->mergeNodes(pos, parentLocked.ptr, rightLocked.ptr)) {
                  // XXX: should reuse page Id
               }
            }
         } else {
            GuardX<BTreeNode> nodeLocked(move(node));
            parent.release();
            nodeLocked->removeSlot(slotId);
         }
         return true;
      } catch(const OLCRestartException&) { yield(repeatCounter); }
   }
}
typedef u64 KeyType;

template <class Record>
struct vmcacheAdapter
{
   BTree tree;

   public:
   void scan(const typename Record::Key& key, const std::function<bool(const typename Record::Key&, const Record&)>& found_record_cb, std::function<void()> reset_if_scan_failed_cb) {
      u8 k[Record::maxFoldLength()];
      u16 l = Record::foldKey(k, key);
      u8 kk[Record::maxFoldLength()];
      tree.scanAsc({k, l}, [&](BTreeNode& node, unsigned slot) {
         custom_memcpy(kk, node.getPrefix(), node.prefixLen);
         custom_memcpy(kk+node.prefixLen, node.getKey(slot), node.slot[slot].keyLen);
         typename Record::Key typedKey;
         Record::unfoldKey(kk, typedKey);
         return found_record_cb(typedKey, *reinterpret_cast<const Record*>(node.getPayload(slot).data()));
      });
   }
   // -------------------------------------------------------------------------------------
   void scanDesc(const typename Record::Key& key, const std::function<bool(const typename Record::Key&, const Record&)>& found_record_cb, std::function<void()> reset_if_scan_failed_cb) {
      u8 k[Record::maxFoldLength()];
      u16 l = Record::foldKey(k, key);
      u8 kk[Record::maxFoldLength()];
      bool first = true;
      tree.scanDesc({k, l}, [&](BTreeNode& node, unsigned slot, bool exactMatch) {
         if (first) { // XXX: hack
            first = false;
            if (!exactMatch)
               return true;
         }
         custom_memcpy(kk, node.getPrefix(), node.prefixLen);
         custom_memcpy(kk+node.prefixLen, node.getKey(slot), node.slot[slot].keyLen);
         typename Record::Key typedKey;
         Record::unfoldKey(kk, typedKey);
         return found_record_cb(typedKey, *reinterpret_cast<const Record*>(node.getPayload(slot).data()));
      });
   }
   // -------------------------------------------------------------------------------------
   void insert(const typename Record::Key& key, const Record& record) {
      u8 k[Record::maxFoldLength()];
      u16 l = Record::foldKey(k, key);
      tree.insert({k, l}, {(u8*)(&record), sizeof(Record)});
   }
   // -------------------------------------------------------------------------------------
   template<class Fn>
   void lookup1(const typename Record::Key& key, Fn fn) {
      u8 k[Record::maxFoldLength()];
      u16 l = Record::foldKey(k, key);
      bool succ = tree.lookup({k, l}, [&](span<u8> payload) {
         fn(*reinterpret_cast<const Record*>(payload.data()));
      });
      ucache::assert_crash(succ);
   }
   // -------------------------------------------------------------------------------------
   template<class Fn>
   void update1(const typename Record::Key& key, Fn fn) {
      u8 k[Record::maxFoldLength()];
      u16 l = Record::foldKey(k, key);
      tree.updateInPlace({k, l}, [&](span<u8> payload) {
         fn(*reinterpret_cast<Record*>(payload.data()));
      });
   }
   // -------------------------------------------------------------------------------------
   // Returns false if the record was not found
   bool erase(const typename Record::Key& key) {
      u8 k[Record::maxFoldLength()];
      u16 l = Record::foldKey(k, key);
      return tree.remove({k, l});
   }
   // -------------------------------------------------------------------------------------
   template <class Field>
   Field lookupField(const typename Record::Key& key, Field Record::*f) {
      Field value;
      lookup1(key, [&](const Record& r) { value = r.*f; });
      return value;
   }

   u64 count() {
      u64 cnt = 0;
      tree.scanAsc({(u8*)nullptr, 0}, [&](BTreeNode& node, unsigned slot) { cnt++; return true; });
      return cnt;
   }

   u64 countw(Integer w_id) {
      u8 k[sizeof(Integer)];
      fold(k, w_id);
      u64 cnt = 0;
      u8 kk[Record::maxFoldLength()];
      tree.scanAsc({k, sizeof(Integer)}, [&](BTreeNode& node, unsigned slot) {
         custom_memcpy(kk, node.getPrefix(), node.prefixLen);
         custom_memcpy(kk+node.prefixLen, node.getKey(slot), node.slot[slot].keyLen);
         if (custom_memcmp(k, kk, sizeof(Integer))!=0)
            return false;
         cnt++;
         return true;
      });
      return cnt;
   }
};

int pin_thread_to_core(int core_id) {
   cpu_set_t cpuset;
   CPU_ZERO(&cpuset);
   CPU_SET(core_id, &cpuset);

   pthread_t current_thread = pthread_self();
   return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

template<class Fn>
void parallel_for(uint64_t begin, uint64_t end, uint64_t nthreads, Fn fn) {
   std::vector<std::thread> threads;
   uint64_t n = end-begin;
   if (n<nthreads)
      nthreads = n;
   uint64_t perThread = n/nthreads;
   for (unsigned i=0; i<nthreads; i++) {
      threads.emplace_back([&,i]() {
         pin_thread_to_core(i);
         uint64_t b = (perThread*i) + begin;
         uint64_t e = (i==(nthreads-1)) ? end : (b+perThread);
         fn(i, b, e);
      });
   }
   for (auto& t : threads)
      t.join();
}

int main(int argc, char** argv) {
   unsigned nthreads = sched::cpus.size();
   u64 n = envOr("DATASIZE", 10);
   u64 runForSec = envOr("RUNFOR", 30);
   bool isRndread = envOr("RNDREAD", 0);

   u64 statDiff = 1e8;
   atomic<u64> txProgress(0);
   atomic<bool> keepRunning(true);
   auto systemName = "ucache_tabby";

   auto statFn = [&]() {
      cout << "ts,tx,rmb,wmb,system,threads,datasize,workload,batch" << endl;
      u64 cnt = 0;
      for (uint64_t i=0; i<runForSec; i++) {
         sleep(1);
         float rmb = (ucache::uCacheManager->readSize.exchange(0))/(1024.0*1024);
         float wmb = (ucache::uCacheManager->writeSize.exchange(0))/(1024.0*1024);
         u64 prog = txProgress.exchange(0);
         u64 pf = ucache::uCacheManager->pageFaults.exchange(0);
         cout << cnt++ << "," << prog << "," << rmb << "," << wmb << "," << systemName << "," << nthreads << "," << n << "," << (isRndread?"rndread":"tpcc") << "," << ucache::uCacheManager->evict_batch << endl;
      }
      keepRunning = false;
   };

   if (isRndread) {
      BTree bt;
      bt.splitOrdered = false;

      {
         // insert
         parallel_for(0, n, nthreads, [&](uint64_t worker, uint64_t begin, uint64_t end) {
            workerThreadId = worker;
            array<u8, 120> payload;
            for (u64 i=begin; i<end; i++) {
               union { u64 v1; u8 k1[sizeof(u64)]; };
               v1 = __builtin_bswap64(i);
               custom_memcpy(payload.data(), k1, sizeof(u64));
               bt.insert({k1, sizeof(KeyType)}, payload);
            }
         });
      }
      cerr << "space: " << (bm.allocCount.load()*pageSize)/(float)bm.gb << " GB " << endl;

      ucache::uCacheManager->readSize = 0;
      ucache::uCacheManager->writeSize = 0;
      thread statThread(statFn);

      parallel_for(0, nthreads, nthreads, [&](uint64_t worker, uint64_t begin, uint64_t end) {
            workerThreadId = worker;
         u64 cnt = 0;
         u64 start = rdtsc();
         while (keepRunning.load()) {
            union { u64 v1; u8 k1[sizeof(u64)]; };
            v1 = __builtin_bswap64(RandomGenerator::getRand<u64>(0, n));

            array<u8, 120> payload; 
            bool succ = bt.lookup({k1, sizeof(u64)}, [&](span<u8> p) {
		           custom_memcpy(payload.data(), p.data(), p.size());
            });
            ucache::assert_crash(succ);
            ucache::assert_crash(custom_memcmp(k1, payload.data(), sizeof(u64))==0);

            cnt++;
            u64 stop = rdtsc();
            if ((stop-start) > statDiff) {
               txProgress += cnt;
               start = stop;
               cnt = 0;
            }
         }
         txProgress += cnt;
      });

      statThread.join();
      bm.ucache_vma->file->close();
      return 0;
   }

   // TPC-C
   Integer warehouseCount = n;

   vmcacheAdapter<warehouse_t> warehouse;
   vmcacheAdapter<district_t> district;
   vmcacheAdapter<customer_t> customer;
   vmcacheAdapter<customer_wdl_t> customerwdl;
   vmcacheAdapter<history_t> history;
   vmcacheAdapter<neworder_t> neworder;
   vmcacheAdapter<order_t> order;
   vmcacheAdapter<order_wdc_t> order_wdc;
   vmcacheAdapter<orderline_t> orderline;
   vmcacheAdapter<item_t> item;
   vmcacheAdapter<stock_t> stock;

   TPCCWorkload<vmcacheAdapter> tpcc(warehouse, district, customer, customerwdl, history, neworder, order, order_wdc, orderline, item, stock, true, warehouseCount, true);
   {
      tpcc.loadItem();
      tpcc.loadWarehouse();
      parallel_for(1, warehouseCount+1, nthreads, [&](uint64_t worker, uint64_t begin, uint64_t end) {
         workerThreadId = worker;
         for (Integer w_id=begin; w_id<end; w_id++) {
            tpcc.loadStock(w_id);
            tpcc.loadDistrinct(w_id);
            for (Integer d_id = 1; d_id <= 10; d_id++) {
                tpcc.loadCustomer(w_id, d_id);
                tpcc.loadOrders(w_id, d_id);
            }
         }
      });
   }
   cerr << "space: " << (bm.allocCount.load()*pageSize)/(float)bm.gb << " GB " << endl;
   ucache::uCacheManager->readSize = 0;
   ucache::uCacheManager->writeSize = 0;
   thread statThread(statFn);

   parallel_for(0, nthreads, nthreads, [&](uint64_t worker, uint64_t begin, uint64_t end) {
      workerThreadId = worker;
      u64 cnt = 0;
      u64 start = rdtsc();
      while (keepRunning.load()) {
         int w_id = tpcc.urand(1, warehouseCount); // wh crossing
         tpcc.tx(w_id);
         cnt++;
         u64 stop = rdtsc();
         if ((stop-start) > statDiff) {
            txProgress += cnt;
            start = stop;
            cnt = 0;
         }
      }
      txProgress += cnt;
   });

   statThread.join();
   cerr << "space: " << (bm.allocCount.load()*pageSize)/(float)bm.gb << " GB " << endl;
   bm.ucache_vma->file->close();
   return 0;
}
