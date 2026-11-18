// SafeSound - step 1: check that the ESP32 and the INMP441 microphone work.
//
// Prints, 10 times a second, the sound level (dBFS) and the loudest frequency,
// with a bar graph. Clap, talk or whistle: the numbers must move.
// Send 'r' over serial (or run tools/record.py) to record 5 s and save it as a WAV
// on the PC; the WAV can be opened in the MATLAB demo: safesound_demo('rec.wav').
//
// Board: ESP32-S3 (N16R8) - also works on a classic ESP32 DevKit (other pins below).
// Serial: 921600 baud.

#include <Arduino.h>
#include <ESP_I2S.h>
#include <math.h>

// ---------------- wiring (change here if your board labels differ) ----------------
#if CONFIG_IDF_TARGET_ESP32S3
const int PIN_SCK = 41;   // INMP441 SCK  (bit clock)
const int PIN_WS  = 42;   // INMP441 WS   (word select / LR clock)
const int PIN_SD  = 2;    // INMP441 SD   (data out of the mic)
const int PIN_LED = 48;   // on-board RGB LED on most ESP32-S3 CAM boards (-1 = none)
#else                     // classic ESP32 DevKit
const int PIN_SCK = 14;
const int PIN_WS  = 15;
const int PIN_SD  = 32;
const int PIN_LED = 2;    // blue LED on most DevKits
#endif
// INMP441: VDD -> 3V3, GND -> GND, L/R -> GND (left channel)

const int SR = 16000;              // same sample rate as the detector
const int N = 512;                 // 32 ms analysis block
const int REC_SECONDS = 5;

I2SClass i2s;
int32_t raw[N];
float re[N], im[N];

static void fft(float *x, float *y, int n) {          // in-place radix-2 FFT
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { float t = x[i]; x[i] = x[j]; x[j] = t; t = y[i]; y[i] = y[j]; y[j] = t; }
  }
  for (int len = 2; len <= n; len <<= 1) {
    float a = -2 * PI / len, wr = cosf(a), wi = sinf(a);
    for (int i = 0; i < n; i += len) {
      float cr = 1, ci = 0;
      for (int k = 0; k < len / 2; k++) {
        int p = i + k, q = p + len / 2;
        float tr = x[q] * cr - y[q] * ci, ti = x[q] * ci + y[q] * cr;
        x[q] = x[p] - tr; y[q] = y[p] - ti; x[p] += tr; y[p] += ti;
        float t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}

// 2nd-order Butterworth high-pass at 100 Hz (RBJ biquad): removes the sub-audio rumble
// (handling, air, loose pins) that the INMP441 passes; the detector ignores < 100 Hz anyway
struct Biquad {
  float b0, b1, b2, a1, a2, z1 = 0, z2 = 0;
  Biquad(float fc, float fs) {
    float w = 2 * PI * fc / fs, c = cosf(w), al = sinf(w) / (2 * 0.7071f), a0 = 1 + al;
    b0 = (1 + c) / 2 / a0; b1 = -(1 + c) / a0; b2 = b0; a1 = -2 * c / a0; a2 = (1 - al) / a0;
  }
  float step(float x) {                    // transposed direct form II
    float y = b0 * x + z1;
    z1 = b1 * x - a1 * y + z2;
    z2 = b2 * x - a2 * y;
    return y;
  }
};
Biquad hp(100.0f, SR);

// read n mono samples, scaled to [-1, 1) and high-passed; returns false on a read error
static bool readBlock(float *out, int n) {
  size_t want = n * sizeof(int32_t), got = 0;
  while (got < want) {
    size_t r = i2s.readBytes((char *)raw + got, want - got);
    if (r == 0) return false;
    got += r;
  }
  for (int i = 0; i < n; i++) out[i] = hp.step((raw[i] >> 8) / 8388608.0f);   // 24-bit in 32-bit slot
  return true;
}

static void led(int level) {               // 0..255 green -> red with loudness
  if (PIN_LED < 0) return;
#if CONFIG_IDF_TARGET_ESP32S3
  rgbLedWrite(PIN_LED, level, 255 - level, 0);
#else
  digitalWrite(PIN_LED, level > 128);
#endif
}

static void record() {                     // 5 s of 16-bit PCM to the PC
  const int total = SR * REC_SECONDS;
  Serial.printf("\nREC %d %d\n", SR, total);
  static float blk[N];
  static int16_t pcm[N];
  for (int done = 0; done < total; done += N) {
    if (!readBlock(blk, N)) break;
    for (int i = 0; i < N; i++) {
      float v = blk[i] * 16.0f;            // +24 dB so speech is not tiny in 16 bits
      pcm[i] = (int16_t)constrain(v * 32767.0f, -32768.0f, 32767.0f);
    }
    int n = min(N, total - done);
    Serial.write((uint8_t *)pcm, n * 2);
  }
  Serial.println("\nEND");
}

void setup() {
  Serial.begin(921600);
  delay(1500);
  Serial.println("\nSafeSound mic test");
  Serial.printf("pins: SCK=%d WS=%d SD=%d, %d Hz\n", PIN_SCK, PIN_WS, PIN_SD, SR);
#if CONFIG_IDF_TARGET_ESP32S3
  Serial.printf("PSRAM: %u bytes, flash: %u bytes\n", ESP.getPsramSize(), ESP.getFlashChipSize());
#endif
  if (PIN_LED >= 0) pinMode(PIN_LED, OUTPUT);

  i2s.setPins(PIN_SCK, PIN_WS, -1, PIN_SD);
  if (!i2s.begin(I2S_MODE_STD, SR, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO, I2S_STD_SLOT_LEFT)) {
    Serial.println("ERROR: I2S did not start - check the pin numbers");
    while (true) delay(1000);
  }
  float tmp[N];
  for (int i = 0; i < 10; i++) readBlock(tmp, N);       // let the mic settle (~0.3 s)
  Serial.println("level(dBFS)  peak(Hz)  ");
}

void loop() {
  if (Serial.available() && Serial.read() == 'r') { record(); return; }

  static float x[N];
  static int zeros = 0;
  float acc = 0, meanv = 0;
  for (int b = 0; b < 3; b++) {                         // ~100 ms
    if (!readBlock(x, N)) { Serial.println("ERROR: I2S read failed"); delay(500); return; }
    meanv = 0;
    for (int i = 0; i < N; i++) meanv += x[i];
    meanv /= N;
    for (int i = 0; i < N; i++) {
      float v = x[i] - meanv;                            // remove DC
      acc += v * v;
      re[i] = v * (0.5f - 0.5f * cosf(2 * PI * i / (N - 1)));   // Hann window
      im[i] = 0;
    }
  }
  float rms = sqrtf(acc / (3 * N));
  float db = 20 * log10f(rms + 1e-9f);

  fft(re, im, N);
  int kmax = 1; float pmax = 0;
  for (int k = 4; k < N / 2; k++) {                     // skip < 125 Hz
    float p = re[k] * re[k] + im[k] * im[k];
    if (p > pmax) { pmax = p; kmax = k; }
  }
  int peakHz = kmax * SR / N;

  // all-zero or constant data = mic not answering (wiring, L/R or power)
  zeros = (rms < 1e-7f) ? zeros + 1 : 0;
  if (zeros == 10) Serial.println("WARNING: the mic sends only zeros - check SD/SCK/WS, VDD=3V3, L/R=GND");

  int bars = constrain((int)((db + 90) / 2), 0, 40);   // -90 dBFS .. -10 dBFS
  char bar[41];
  for (int i = 0; i < 40; i++) bar[i] = i < bars ? '#' : '.';
  bar[40] = 0;
  Serial.printf("%7.1f  %6d  %s\n", db, peakHz, bar);
  led(constrain((int)((db + 70) * 5), 0, 255));
}
