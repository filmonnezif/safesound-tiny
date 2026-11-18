// SafeSound core: log-mel front end + DS-CNN forward pass, plain C, float.
// The same file runs on the ESP32-S3 and on a PC (tools/check_core.c), so the device can be
// checked against the PyTorch model. Front end = safesound/frontend.py: 16 kHz, frames of
// 512 samples every 320 (non-centred), periodic Hann, power spectrum, 40 Slaney-style mel
// filters 50-8000 Hz, log(E + 1) on the int16 scale. A 2 s window = 100 frames.
#pragma once
#pragma GCC optimize("O3")
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"

#ifndef SS_ALLOC
#define SS_ALLOC(n) malloc(n)
#endif

#define SS_SR 16000
#define SS_NFFT 512
#define SS_HOP 320
#define SS_MELS 40
#define SS_FRAMES 100
#define SS_BINS (SS_NFFT / 2 + 1)

static float ss_win[SS_NFFT];
static float ss_fb[SS_MELS][SS_BINS];
static float ss_cos[SS_NFFT / 2], ss_sin[SS_NFFT / 2];
static float *ss_bufA, *ss_bufB;

static double ss_hz2mel(double f) { return 2595.0 * log10(1.0 + f / 700.0); }
static double ss_mel2hz(double m) { return 700.0 * (pow(10.0, m / 2595.0) - 1.0); }

static int ss_init(void) {
  for (int i = 0; i < SS_NFFT; i++) ss_win[i] = (float)(0.5 - 0.5 * cos(2 * M_PI * i / SS_NFFT));
  for (int i = 0; i < SS_NFFT / 2; i++) {
    ss_cos[i] = (float)cos(2 * M_PI * i / SS_NFFT);
    ss_sin[i] = (float)-sin(2 * M_PI * i / SS_NFFT);
  }
  double edges[SS_MELS + 2], lo = ss_hz2mel(50.0), hi = ss_hz2mel(8000.0);
  for (int i = 0; i < SS_MELS + 2; i++) edges[i] = ss_mel2hz(lo + (hi - lo) * i / (SS_MELS + 1));
  for (int m = 0; m < SS_MELS; m++)
    for (int k = 0; k < SS_BINS; k++) {
      double f = (double)k * (SS_SR / 2) / (SS_BINS - 1);
      double up = (f - edges[m]) / (edges[m + 1] - edges[m]);
      double down = (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]);
      double v = fmax(0.0, fmin(up, down)) * 2.0 / (edges[m + 2] - edges[m]);
      ss_fb[m][k] = (float)v;
    }
  size_t n = (size_t)SS_WIDTH * 20 * 50 * sizeof(float);
  ss_bufA = (float *)SS_ALLOC(n);
  ss_bufB = (float *)SS_ALLOC(n);
  return ss_bufA && ss_bufB;
}

// in-place complex FFT of length SS_NFFT
static void ss_fft(float *re, float *im) {
  const int n = SS_NFFT;
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (int len = 2; len <= n; len <<= 1) {
    int step = n / len;
    for (int i = 0; i < n; i += len)
      for (int k = 0; k < len / 2; k++) {
        float wr = ss_cos[k * step], wi = ss_sin[k * step];
        int p = i + k, q = p + len / 2;
        float tr = re[q] * wr - im[q] * wi, ti = re[q] * wi + im[q] * wr;
        re[q] = re[p] - tr; im[q] = im[p] - ti; re[p] += tr; im[p] += ti;
      }
  }
}

// one frame: x = 512 samples in [-1, 1] -> 40 log-mel values
static void ss_frame(const float *x, float *mel) {
  static float re[SS_NFFT], im[SS_NFFT], pw[SS_BINS];
  for (int i = 0; i < SS_NFFT; i++) { re[i] = x[i] * 32768.0f * ss_win[i]; im[i] = 0; }
  ss_fft(re, im);
  for (int k = 0; k < SS_BINS; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
  for (int m = 0; m < SS_MELS; m++) {
    float e = 0;
    for (int k = 0; k < SS_BINS; k++) e += ss_fb[m][k] * pw[k];
    mel[m] = logf(e + 1.0f);
  }
}

// The activations live in (slow) external PSRAM on the ESP32-S3; each layer copies small
// pieces into fast internal RAM first, so the inner loops never touch PSRAM.
#define SS_TILE 40
static float ss_chan[20 * 50], ss_chan_out[20 * 50];
static float ss_tin[SS_WIDTH * SS_TILE], ss_tacc[SS_TILE];

// depthwise 3x3, pad 1, stride s, + bias + ReLU. in: C x H x W -> out: C x Ho x Wo
static void ss_dw(const float *in, float *out, int C, int H, int W, int s, const float *w, const float *b,
                  int *Ho, int *Wo) {
  int ho = (H - 1) / s + 1, wo = (W - 1) / s + 1;
  for (int c = 0; c < C; c++) {
    const float *k = w + c * 9, *ip = ss_chan;
    memcpy(ss_chan, in + (size_t)c * H * W, sizeof(float) * H * W);
    float *op = ss_chan_out;
    for (int y = 0; y < ho; y++)
      for (int x = 0; x < wo; x++) {
        float acc = b[c];
        for (int ky = 0; ky < 3; ky++) {
          int iy = y * s + ky - 1;
          if (iy < 0 || iy >= H) continue;
          for (int kx = 0; kx < 3; kx++) {
            int ix = x * s + kx - 1;
            if (ix < 0 || ix >= W) continue;
            acc += k[ky * 3 + kx] * ip[iy * W + ix];
          }
        }
        op[y * wo + x] = acc > 0 ? acc : 0;
      }
    memcpy(out + (size_t)c * ho * wo, ss_chan_out, sizeof(float) * ho * wo);
  }
  *Ho = ho; *Wo = wo;
}

// pointwise 1x1 C -> C, + bias + ReLU
static void ss_pw(const float *in, float *out, int C, int HW, const float *w, const float *b) {
  for (int p0 = 0; p0 < HW; p0 += SS_TILE) {                  // a tile of pixels, all channels
    int n = HW - p0 < SS_TILE ? HW - p0 : SS_TILE;
    for (int c = 0; c < C; c++) memcpy(ss_tin + c * SS_TILE, in + (size_t)c * HW + p0, sizeof(float) * n);
    for (int o = 0; o < C; o++) {
      for (int i = 0; i < n; i++) ss_tacc[i] = b[o];
      const float *wr = w + o * C;
      for (int c = 0; c < C; c++) {
        float wv = wr[c];
        const float *ip = ss_tin + c * SS_TILE;
        for (int i = 0; i < n; i++) ss_tacc[i] += wv * ip[i];
      }
      float *op = out + (size_t)o * HW + p0;
      for (int i = 0; i < n; i++) op[i] = ss_tacc[i] > 0 ? ss_tacc[i] : 0;
    }
  }
}

// feat: 40 x 100 log-mel (mel-major: feat[m * 100 + t]) -> SS_CLASSES probabilities
static void ss_net(const float *feat, float *prob) {
  const int C = SS_WIDTH;
  int H = 20, W = 50;
  float *A = ss_bufA, *B = ss_bufB;
  for (int o = 0; o < C; o++)                                   // conv 3x3 /2, 1 -> C
    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
        float acc = SS_C0_B[o];
        for (int ky = 0; ky < 3; ky++) {
          int iy = 2 * y + ky - 1;
          if (iy < 0 || iy >= SS_MELS) continue;
          for (int kx = 0; kx < 3; kx++) {
            int ix = 2 * x + kx - 1;
            if (ix < 0 || ix >= SS_FRAMES) continue;
            float v = (feat[iy * SS_FRAMES + ix] - SS_MEAN[iy]) * SS_INVSTD[iy];
            acc += SS_C0_W[o * 9 + ky * 3 + kx] * v;
          }
        }
        A[(size_t)o * H * W + y * W + x] = acc > 0 ? acc : 0;
      }
  for (int b = 0; b < SS_BLOCKS; b++) {
    int ho, wo;
    ss_dw(A, B, C, H, W, b == 1 ? 2 : 1, SS_DW_W[b], SS_DW_B[b], &ho, &wo);
    H = ho; W = wo;
    ss_pw(B, A, C, H * W, SS_PW_W[b], SS_PW_B[b]);
  }
  float pooled[SS_WIDTH];
  for (int c = 0; c < C; c++) {                                 // mean over freq, max + mean over time
    float mx = -1e30f, sum = 0;
    for (int x = 0; x < W; x++) {
      float m = 0;
      for (int y = 0; y < H; y++) m += A[(size_t)c * H * W + y * W + x];
      m /= H;
      if (m > mx) mx = m;
      sum += m;
    }
    pooled[c] = mx + sum / W;
  }
  for (int k = 0; k < SS_CLASSES; k++) {
    float z = SS_FC_B[k];
    for (int c = 0; c < C; c++) z += SS_FC_W[k * C + c] * pooled[c];
    prob[k] = 1.0f / (1.0f + expf(-z));
  }
}
