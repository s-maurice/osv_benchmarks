src = ../..
BENCH_CXXFLAGS = -std=gnu++20 -O3 -mavx -mavx2 -DCONF_debug_memory=0

INCLUDES = -I$(src)/include -I$(src) -I$(src)/arch/common -I$(src)/arch/x64 \
	-I$(src)/build/release.x64/gen/include

print-extra-cxxflags:
	@echo $(BENCH_CXXFLAGS)

lib: libintegrated_vmcache_tabby.a

libintegrated_vmcache_tabby.a: integrated_vmcache_tabby.cpp tpcc/* rte_string.hh Makefile
	g++ $(BENCH_CXXFLAGS) $(INCLUDES) -D__OSV__ -flto -ffat-lto-objects -c integrated_vmcache_tabby.cpp -o integrated_vmcache_tabby.o
	ar rcs libintegrated_vmcache_tabby.a integrated_vmcache_tabby.o

module:

clean:
	rm -f integrated_vmcache_tabby.o libintegrated_vmcache_tabby.a
