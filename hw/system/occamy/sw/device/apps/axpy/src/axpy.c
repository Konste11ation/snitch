#include "snrt.h"
#include "data.h"

// Define your kernel
void axpy(uint32_t l, double a, double *x, double *y, double *z) {
    for (uint32_t i = 0; i < l ; i++) {
        z[i] = a * x[i] + y[i];
    }
    snrt_fpu_fence();
}

int main() {

    // Perform some operations (e.g. clear interrupt) after wakeup
    post_wakeup_cl();

    // DM core does not participate in the computation
    if(snrt_is_compute_core()) {
        uint32_t start_cycle = mcycle();
        axpy(L, a, x, y, z);
        uint32_t end_cycle = mcycle();
    }

    // Synchronize all cores and send an interrupt to CVA6
    return_to_cva6(SYNC_ALL);
}