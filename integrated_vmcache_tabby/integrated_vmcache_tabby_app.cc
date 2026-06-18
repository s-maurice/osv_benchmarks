#include <osv/application.hh>

int benchmark_main(int argc, char** argv);

extern "C" int app_main(int argc, char** argv) {
    return benchmark_main(argc, argv);
}
