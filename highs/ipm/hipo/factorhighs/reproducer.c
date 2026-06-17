/* Minimal reproducer for a segmentation fault in HiPO at process exit.
 *
 * HiGHS branch: hipo-c, built with -DHIPO=ON -DFAST_BUILD=ON.
 *
 * Symptom: a single successful call to FactorHighs_initialise() makes the
 * process crash during static-destructor teardown, AFTER main() returns.
 * No other HiPO function needs to be called.
 *
 * Confirmed:
 *   - Linking/loading libhighs WITHOUT calling FactorHighs_initialise() exits
 *     cleanly (code 0).
 *   - Calling only FactorHighs_initialise(0) and returning -> SIGSEGV (139).
 *   - Adding FactorHighs_terminate() before returning does NOT help.
 *   - A SIGSEGV handler installed in main() is never invoked, so the fault
 *     happens in a context where it cannot run (worker-thread / linker teardown).
 *
 * The HiPO computation itself (analyse/factorise/solve) returns correct
 * results; only the process exit is affected.
 *
 * Build (adjust paths to your HiGHS install):
 *   gcc hipo_teardown_segfault.c -o hipo_teardown_segfault \
 *       -I/path/to/HiGHS/install/include/highs \
 *       -L/path/to/HiGHS/install/lib -lhighs
 *
 * Run (libhighs_extras.so and the BLAS must be on the dlopen search path):
 *   export LD_LIBRARY_PATH=/path/to/HiGHS/install/lib:/path/to/HiGHS/build/lib:/path/to/openblas/lib
 *   ./hipo_teardown_segfault ; echo "exit code: $?"
 *
 * Expected (buggy) output:
 *   init status: 0
 *   returning from main now
 *   Segmentation fault (core dumped)
 *   exit code: 139
 */

#include "ipm/hipo/factorhighs/FactorHighs_c_api.h"
#include <stdio.h>

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0); /* unbuffered, so output is visible before the crash */

  HighsInt status = FactorHighs_initialise(0); /* 0 = default number of threads */
  printf("init status: %lld\n", (long long) status);
  if (status != 0) {
    printf("HiPO not available (is libhighs_extras.so on the search path?)\n");
    return 1;
  }

  printf("returning from main now\n");
  return 0; /* <-- SIGSEGV happens after this, during static-destructor teardown */
}
