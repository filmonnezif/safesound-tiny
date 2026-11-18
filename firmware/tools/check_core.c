// PC check of the firmware core: scores of a 16 kHz mono 16-bit WAV, same windows as
// export_model.py --ref (2 s of silence first, 2 s windows every 0.5 s).
//   cc -O2 -I firmware/detector firmware/tools/check_core.c -lm -o /tmp/check_core
#include <stdio.h>
#include <stdint.h>
#include "ss_core.h"

int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb");
  if (!f) { perror("open"); return 1; }
  char id[4]; uint32_t sz; int16_t *pcm = NULL; long n = 0;
  fseek(f, 12, SEEK_SET);
  while (fread(id, 1, 4, f) == 4 && fread(&sz, 4, 1, f) == 1) {
    if (!memcmp(id, "data", 4)) { n = sz / 2; pcm = malloc(sz); fread(pcm, 2, n, f); break; }
    fseek(f, sz + (sz & 1), SEEK_CUR);
  }
  if (!pcm) { fprintf(stderr, "no data chunk\n"); return 1; }
  long pad = 2 * SS_SR, total = pad + n;
  float *x = calloc(total, sizeof(float));
  for (long i = 0; i < n; i++) x[pad + i] = pcm[i] / 32768.0f;
  long nf = 1 + (total - SS_NFFT) / SS_HOP;
  float *F = malloc(sizeof(float) * SS_MELS * nf), mel[SS_MELS];
  ss_init();
  for (long t = 0; t < nf; t++) {
    ss_frame(x + t * SS_HOP, mel);
    for (int m = 0; m < SS_MELS; m++) F[m * nf + t] = mel[m];
  }
  static float win[SS_MELS * SS_FRAMES];
  float p[SS_CLASSES];
  for (long k = 0; k < n / (SS_SR / 2); k++) {
    long e = (pad + (k + 1) * (SS_SR / 2) - SS_NFFT) / SS_HOP;
    for (int m = 0; m < SS_MELS; m++)
      for (int t = 0; t < SS_FRAMES; t++) win[m * SS_FRAMES + t] = F[m * nf + e - 99 + t];
    ss_net(win, p);
    printf("%6.1f", (k + 1) * 0.5);
    for (int c = 0; c < SS_CLASSES; c++) printf(" %.4f", p[c]);
    printf("\n");
  }
  return 0;
}
