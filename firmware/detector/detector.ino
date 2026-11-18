// SafeSound detector on the ESP32-S3 (N16R8) + INMP441.
//
// Audio task (core 1): reads the mic, high-passes at 30 Hz, makes a log-mel frame every 20 ms.
// Network task (core 0): every 0.5 s runs the DS-CNN on the last 2 s (100 frames) and prints
// the 6 class probabilities; a class alerts when its probability passes its threshold
// (1 false alarm per hour on the tuning set). Serial 921600 baud.
// Serial commands: '+' / '-' change the input gain by 6 dB.
//
// Wiring (same as mic_test): VDD 3V3, GND, L/R GND, SCK 41, WS 42, SD 2.

#include <Arduino.h>
#include <ESP_I2S.h>
#include "esp_heap_caps.h"
#define SS_ALLOC(n) heap_caps_malloc((n), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
#include "ss_core.h"

const int PIN_SCK = 41, PIN_WS = 42, PIN_SD = 2, PIN_LED = 48;

I2SClass i2s;
static float gainDb = 0.0f;

// ring of log-mel frames; the network copies the last 100 when a window is due
static float ring[SS_FRAMES][SS_MELS];
static volatile int ringPos = 0;            // next frame slot
static volatile uint32_t frameCount = 0;
static float window_[SS_MELS * SS_FRAMES];
static TaskHandle_t netTask;
static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
static volatile bool netBusy = false;
static volatile uint32_t skipped = 0;

struct HP {                                 // 2nd-order Butterworth high-pass, 30 Hz
  float b0, b1, b2, a1, a2, z1 = 0, z2 = 0;
  HP(float fc, float fs) {
    float w = 2 * PI * fc / fs, c = cosf(w), al = sinf(w) / (2 * 0.7071f), a0 = 1 + al;
    b0 = (1 + c) / 2 / a0; b1 = -(1 + c) / a0; b2 = b0; a1 = -2 * c / a0; a2 = (1 - al) / a0;
  }
  float step(float x) { float y = b0 * x + z1; z1 = b1 * x - a1 * y + z2; z2 = b2 * x - a2 * y; return y; }
} hp(30.0f, SS_SR);

static void audioTask(void *) {
  static int32_t raw[SS_HOP];
  static float buf[SS_NFFT];                // last 512 samples
  float mel[SS_MELS];
  int filled = 0;
  for (;;) {
    size_t want = sizeof(raw), got = 0;
    while (got < want) got += i2s.readBytes((char *)raw + got, want - got);
    float g = powf(10.0f, gainDb / 20.0f);
    memmove(buf, buf + SS_HOP, (SS_NFFT - SS_HOP) * sizeof(float));
    for (int i = 0; i < SS_HOP; i++)
      buf[SS_NFFT - SS_HOP + i] = hp.step((raw[i] >> 8) / 8388608.0f) * g;
    if (filled < SS_NFFT) { filled += SS_HOP; if (filled < SS_NFFT) continue; }
    ss_frame(buf, mel);
    portENTER_CRITICAL(&mux);
    memcpy(ring[ringPos], mel, sizeof(mel));
    ringPos = (ringPos + 1) % SS_FRAMES;
    frameCount++;
    portEXIT_CRITICAL(&mux);
    if (frameCount >= SS_FRAMES && frameCount % 25 == 0) {      // every 0.5 s
      if (netBusy) skipped++;
      else xTaskNotifyGive(netTask);
    }
  }
}

static void networkTask(void *) {
  float prob[SS_CLASSES];
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    netBusy = true;
    portENTER_CRITICAL(&mux);
    int start = ringPos;                     // oldest frame
    for (int t = 0; t < SS_FRAMES; t++)
      for (int m = 0; m < SS_MELS; m++) window_[m * SS_FRAMES + t] = ring[(start + t) % SS_FRAMES][m];
    uint32_t fc = frameCount;
    portEXIT_CRITICAL(&mux);

    // loudness of the window vs the training data (mean log-mel, for calibrating the gain)
    float lvl = 0, ref = 0;
    for (int m = 0; m < SS_MELS; m++) {
      for (int t = 0; t < SS_FRAMES; t++) lvl += window_[m * SS_FRAMES + t];
      ref += SS_MEAN[m];
    }
    lvl /= SS_MELS * SS_FRAMES; ref /= SS_MELS;

    uint32_t t0 = micros();
    ss_net(window_, prob);
    uint32_t ms = (micros() - t0) / 1000;

    int best = 0;
    for (int k = 1; k < SS_CLASSES; k++) if (prob[k] / SS_THRESH[k] > prob[best] / SS_THRESH[best]) best = k;
    String alerts;
    for (int k = 0; k < SS_CLASSES; k++)
      if (prob[k] >= SS_THRESH[k]) alerts += String(alerts.length() ? "," : "") + SS_NAMES[k];

    Serial.printf("t=%7.1f net=%4ums lvl=%5.2f(train %4.2f) |", fc * 0.02f, ms, lvl, ref);
    for (int k = 0; k < SS_CLASSES; k++) Serial.printf(" %s %.2f", SS_NAMES[k], prob[k]);
    if (alerts.length()) Serial.printf("  >>> ALERT %s", alerts.c_str());
    if (skipped) Serial.printf("  (skipped %u)", skipped);
    Serial.println();
    rgbLedWrite(PIN_LED, alerts.length() ? 255 : 0, 0, alerts.length() ? 0 : 20);
    netBusy = false;
  }
}

void setup() {
  Serial.begin(921600);
  delay(1500);
  Serial.println("\nSafeSound detector");
  Serial.printf("model: width %d, blocks %d; PSRAM %u bytes, flash %u bytes\n", SS_WIDTH, SS_BLOCKS,
                ESP.getPsramSize(), ESP.getFlashChipSize());
  Serial.print("thresholds:");
  for (int k = 0; k < SS_CLASSES; k++) Serial.printf(" %s %.3f", SS_NAMES[k], SS_THRESH[k]);
  Serial.println();
  if (!ss_init()) { Serial.println("ERROR: no PSRAM for the network buffers"); while (true) delay(1000); }
  pinMode(PIN_LED, OUTPUT);
  i2s.setPins(PIN_SCK, PIN_WS, -1, PIN_SD);
  if (!i2s.begin(I2S_MODE_STD, SS_SR, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO, I2S_STD_SLOT_LEFT)) {
    Serial.println("ERROR: I2S did not start"); while (true) delay(1000);
  }
  xTaskCreatePinnedToCore(networkTask, "net", 16384, nullptr, 1, &netTask, 0);
  xTaskCreatePinnedToCore(audioTask, "audio", 16384, nullptr, 5, nullptr, 1);
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '+') gainDb += 6;
    if (c == '-') gainDb -= 6;
    if (c == '+' || c == '-') Serial.printf("gain %+.0f dB\n", gainDb);
  }
  delay(20);
}
