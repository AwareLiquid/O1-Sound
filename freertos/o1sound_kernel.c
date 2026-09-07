/* O1-Sound reference kernel for FreeRTOS targets.
 *
 * Streaming wake-word inference in plain C: log-mel frontend -> layer norm ->
 * 2 liquid-core cells -> linear head. Math matches the PyTorch model's
 * `O1Sound.step()` / `LogMel.forward()`; host-side parity is gated by
 * tests/test_c_parity.py (zig cc compiles this file, feeds identical frames,
 * compares logits within 1e-3).
 *
 * The frontend here is a REFERENCE implementation: a direct O(N^2) DFT of
 * n_fft=480. It is correct and parity-gated but not the production MCU path
 * (a mixed-radix FFT, or zero-padding to 512 with a re-validated filterbank,
 * is the optimization to make after the port works). The kernel core
 * (cells + head) is identical either way.
 */
#include "o1sound_kernel.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ------------------------------------------------------------------ */
/* Frontend: per-frame log-mel, torch.stft-compatible.                */
/* window[] has n_fft entries; mel_fb[] is (n_mels x n_freqs) row-major. */
/* ------------------------------------------------------------------ */

static void dft480(const float *win, o1s_complex *out) {
    /* Direct DFT, n_fft = 480. Onesided output: bins 0..240. */
    const int n = O1S_N_FFT;
    for (int k = 0; k < O1S_N_FREQS; ++k) {
        float re = 0.0f, im = 0.0f;
        for (int t = 0; t < n; ++t) {
            float ang = -2.0f * M_PI * (float)(k * t) / (float)n;
            float c = cosf(ang), s = sinf(ang);
            re += win[t] * c;
            im += win[t] * s;
        }
        out[k].re = re;
        out[k].im = im;
    }
}

static void log_mel_frame(const float *frame480, float *mel) {
    float wbuf[O1S_N_FFT];
    for (int i = 0; i < O1S_N_FFT; ++i) wbuf[i] = frame480[i] * o1s_window[i];
    o1s_complex spec[O1S_N_FREQS];
    dft480(wbuf, spec);
    for (int m = 0; m < O1S_N_MELS; ++m) {
        float p = 0.0f;
        for (int k = 0; k < O1S_N_FREQS; ++k) {
            float pw = spec[k].re * spec[k].re + spec[k].im * spec[k].im;
            p += o1s_mel_fb[m * O1S_N_FREQS + k] * pw;
        }
        mel[m] = logf(p + O1S_EPS_LOG);
    }
}

/* ------------------------------------------------------------------ */
/* Kernel core: norm -> cells -> head. State is 2 x O1S_HIDDEN floats. */
/* ------------------------------------------------------------------ */

static void layer_norm(const float *x, float *y) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < O1S_N_MELS; ++i) mean += x[i];
    mean /= (float)O1S_N_MELS;
    for (int i = 0; i < O1S_N_MELS; ++i) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)O1S_N_MELS;
    float inv = 1.0f / sqrtf(var + O1S_EPS_LN);
    for (int i = 0; i < O1S_N_MELS; ++i) {
        y[i] = (x[i] - mean) * inv * o1s_norm_w[i] + o1s_norm_b[i];
    }
}

/* Layer 0: 40 -> 640. Layer 1: 640 -> 640.
 * NB: the recurrent sum must read the OLD h for every row — compute all
 * pre-activations first, then apply the leak update (torch does the same:
 * pre = inp(x) + rec(h); h = h + alpha*(-h + tanh(pre))). */
static void cell0_step(const float *x40, float *h) {
    float pre[O1S_HIDDEN];
    for (int i = 0; i < O1S_HIDDEN; ++i) {
        float acc = o1s_l0_inp_b[i];
        for (int j = 0; j < O1S_N_MELS; ++j) acc += o1s_l0_inp_w[i * O1S_N_MELS + j] * x40[j];
        for (int k = 0; k < O1S_HIDDEN; ++k) acc += o1s_l0_rec_w[i * O1S_HIDDEN + k] * h[k];
        pre[i] = acc;
    }
    for (int i = 0; i < O1S_HIDDEN; ++i)
        h[i] = h[i] + o1s_l0_alpha[i] * (-h[i] + tanhf(pre[i]));
}

static void cell1_step(const float *x640, float *h) {
    float pre[O1S_HIDDEN];
    for (int i = 0; i < O1S_HIDDEN; ++i) {
        float acc = o1s_l1_inp_b[i];
        for (int j = 0; j < O1S_HIDDEN; ++j) acc += o1s_l1_inp_w[i * O1S_HIDDEN + j] * x640[j];
        for (int k = 0; k < O1S_HIDDEN; ++k) acc += o1s_l1_rec_w[i * O1S_HIDDEN + k] * h[k];
        pre[i] = acc;
    }
    for (int i = 0; i < O1S_HIDDEN; ++i)
        h[i] = h[i] + o1s_l1_alpha[i] * (-h[i] + tanhf(pre[i]));
}

void o1s_init(o1s_state *st) {
    memset(st->h, 0, sizeof(st->h));
}

/* Streaming step: one n_fft window (480 samples) in -> logits out. */
void o1s_step(const float *frame480, float *logits, o1s_state *st) {
    float mel[O1S_N_MELS];
    log_mel_frame(frame480, mel);

    float x[O1S_N_MELS];
    layer_norm(mel, x);

    float h1[O1S_HIDDEN], h2[O1S_HIDDEN];
    memcpy(h1, st->h, sizeof(h1));
    memcpy(h2, st->h + O1S_HIDDEN, sizeof(h2));
    cell0_step(x, h1);
    cell1_step(h1, h2);
    memcpy(st->h, h1, sizeof(h1));
    memcpy(st->h + O1S_HIDDEN, h2, sizeof(h2));

    for (int c = 0; c < O1S_CLASSES; ++c) {
        float acc = o1s_head_b[c];
        for (int i = 0; i < O1S_HIDDEN; ++i) acc += o1s_head_w[c * O1S_HIDDEN + i] * h2[i];
        logits[c] = acc;
    }
}
