src= ../..
CXXFLAGS = -g -DNDEBUG -DCONF_debug_memory=0 -std=c++20 -mavx -mavx2 -O3
CXXFLAGS += -fnon-call-exceptions -fasynchronous-unwind-tables -shared -fPIC
INCLUDES = -I$(src)/include -I$(src) -I$(src)/arch/common -I$(src)/arch/x64 -I$(src)/build/release/gen/include
INCLUDES += -I$(src)/modules/lwext4/upstream/lwext4/include -I$(src)/modules/lwext4/upstream/lwext4/build_lib_only/include

integrated_vmcache_tabby: integrated_vmcache_tabby.cpp tpcc/* rte_string.hh
	g++ $(CXXFLAGS) -DOSV $(INCLUDES) integrated_vmcache_tabby.cpp -o integrated_vmcache_tabby_eval

module: integrated_vmcache_tabby

clean:
	rm -f integrated_vmcache_tabby
