src = ../..
BENCH_CXXFLAGS = -std=gnu++20 -O3 -march=znver4 -DCONF_debug_memory=0

INCLUDES = -I$(src)/include -I$(src) -I$(src)/arch/common -I$(src)/arch/x64 \
	-I$(src)/build/release.x64/gen/include

print-extra-cxxflags:
	@echo $(BENCH_CXXFLAGS)

lib: libintegrated_vmcache.a

libintegrated_vmcache.a: integrated_vmcache.cpp tpcc/* rte_string.hh Makefile
	g++ $(BENCH_CXXFLAGS) $(INCLUDES) -D__OSV__ -flto -ffat-lto-objects -c integrated_vmcache.cpp -o integrated_vmcache.o
	ar rcs libintegrated_vmcache.a integrated_vmcache.o

module:

clean:
	rm -f integrated_vmcache.o libintegrated_vmcache.a
