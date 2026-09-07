/* Host driver for the C-kernel parity test (not part of the MCU image).
 * Reads a stream of float32 n_fft windows from stdin, prints logits.
 * Compile: zig cc -O2 -I freertos freertos/o1sound_kernel.c test_driver.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include "o1sound_kernel.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

int main(void) {
#ifdef _WIN32
    /* Binary mode: without this the CRT treats 0x1A in the float stream
     * as a text-mode EOF. */
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    o1s_state st;
    o1s_init(&st);
    float frame[O1S_N_FFT];
    float logits[O1S_CLASSES];
    unsigned long n = 0;
    fprintf(stderr, "driver start nfft=%d classes=%d\n", O1S_N_FFT, O1S_CLASSES);
    fflush(stderr);
    size_t got;
    while ((got = fread(frame, sizeof(float), O1S_N_FFT, stdin)) == O1S_N_FFT) {
        o1s_step(frame, logits, &st);
        printf("%lu", n);
        for (int c = 0; c < O1S_CLASSES; ++c) printf(" %.9g", logits[c]);
        printf("\n");
        ++n;
    }
    fprintf(stderr, "loop end: frames=%lu last_fread=%zu ferror=%d feof=%d\n",
            n, got, ferror(stdin), feof(stdin));
    return 0;
}
