/* O1-Sound FreeRTOS demo task — integration sketch.
 *
 * Shows the streaming contract on a real target: an I2S/PDM DMA callback
 * fills a 16 kHz sample ring buffer; this task wakes every 160 samples
 * (10 ms), slides the 480-sample window, runs the kernel, and fires the
 * wake callback when the wake logit clears the threshold for K consecutive
 * frames (debounce).
 *
 * This file is an integration sketch, NOT a compiled target: it assumes a
 * vendor audio driver + CMSIS-style HAL that only exist on real hardware.
 * The kernel core (o1sound_kernel.c) is the ported, parity-gated part;
 * everything below is the wiring.
 */
#include "o1sound_kernel.h"
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"

/* ---- hardware assumptions (fill in from your BSP) ------------------- */
#define O1S_SAMPLE_RATE    16000
#define O1S_HOP            160          /* 10 ms */
#define O1S_RING_SAMPLES   (O1S_N_FFT + O1S_HOP)
#define O1S_THRESHOLD      0.9f         /* wake logit, class index 1 */
#define O1S_DEBOUNCE_K     3            /* consecutive frames to fire */

static volatile int16_t s_ring[O1S_RING_SAMPLES];
static volatile uint32_t s_head;       /* oldest sample index */
static SemaphoreHandle_t s_frames;     /* counting: full hops available */
static o1s_state s_state;

/* Called from the I2S DMA half/full callback (ISR context). */
void o1s_on_audio_samples(const int16_t *buf, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        s_ring[s_head] = buf[i];
        s_head = (s_head + 1) % O1S_RING_SAMPLES;
        if ((s_head % O1S_HOP) == 0) {
            BaseType_t hp = pdFALSE;
            xSemaphoreGiveFromISR(s_frames, &hp);
            portYIELD_FROM_ISR(hp);
        }
    }
}

/* Called on a confirmed wake event (task context, once per debounce). */
extern void o1s_on_wake(void);

static float ring_window[O1S_N_FFT];

static void o1s_task(void *arg) {
    (void)arg;
    o1s_init(&s_state);
    float logits[O1S_CLASSES];
    uint32_t debounce = 0;

    for (;;) {
        if (xSemaphoreTake(s_frames, portMAX_DELAY) != pdTRUE) continue;

        /* Assemble the most recent 480 samples. */
        for (int i = 0; i < O1S_N_FFT; ++i) {
            uint32_t idx = (s_head + O1S_RING_SAMPLES - O1S_N_FFT + i)
                           % O1S_RING_SAMPLES;
            ring_window[i] = (float)s_ring[idx] / 32768.0f;
        }

        o1s_step(ring_window, logits, &s_state);

        /* Class 1 = wake (binary model), or OR of language heads. */
        float wake_score = logits[1];
        if (wake_score > O1S_THRESHOLD) {
            if (++debounce >= O1S_DEBOUNCE_K) {
                o1s_on_wake();
                debounce = 0;
            }
        } else {
            debounce = 0;
        }
    }
}

/* BSP calls this once at boot. */
void o1s_freertos_init(uint32_t stack_words) {
    s_frames = xSemaphoreCreateCounting(O1S_RING_SAMPLES / O1S_HOP, 0);
    xTaskCreate(o1s_task, "o1sound", stack_words, NULL,
                configMAX_PRIORITIES - 2, NULL);
}
