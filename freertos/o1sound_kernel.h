/* O1-Sound FreeRTOS kernel — public interface.
 * See o1sound_kernel.c for the reference implementation notes. */
#ifndef O1SOUND_KERNEL_H
#define O1SOUND_KERNEL_H

#include "generated/weights.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float re, im;
} o1s_complex;

/* Carried state: 2 x O1S_HIDDEN floats, constant for the life of a stream. */
typedef struct {
    float h[2 * O1S_HIDDEN];
} o1s_state;

/* Zero the recurrent state (start of a stream). */
void o1s_init(o1s_state *st);

/* Advance one frame: a full n_fft (480-sample) window in, O1S_CLASSES
 * logits out. The window must be the most recent 480 samples (steady-state
 * equivalence with torch.stft(center=True); edge frames near stream start
 * differ — see tests/test_c_parity.py for the exact agreement region). */
void o1s_step(const float *frame480, float *logits, o1s_state *st);

#ifdef __cplusplus
}
#endif

#endif /* O1SOUND_KERNEL_H */
