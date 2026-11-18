// SafeSound detector on the ESP32-S3 (N16R8) + INMP441, with alert outputs.
//
// Audio task (core 1): reads the mic, high-passes at 30 Hz, makes a log-mel frame every 20 ms,
//   and tracks the sound level against an adaptive noise floor (wake indicator).
// Network task (core 0): every 0.5 s runs the DS-CNN on the last 2 s (100 frames); a class alerts
//   when its probability passes its threshold (1 false alarm per hour on the tuning set).
// Alert task: on an alert, the servo "vibrates" with the class's rhythm and the RGB LED shows the
//   class colour for a few seconds. The on-board LED shows the wake state: dim blue = idle,
//   white = sound above the room's background (stays on 2 s after the sound).
//
// Serial 921600 baud. Commands: '+' / '-' input gain +-6 dB; '0'..'5' test the alert of one class
//   (0 alarm, 1 siren, 2 doorbell, 3 knock, 4 baby cry, 5 car horn) without any sound.
//
// Wiring
//   INMP441: VDD 3V3, GND, L/R GND, SCK GPIO41, WS GPIO42, SD GPIO2
//   Servo (SG90): brown GND, red 5V, orange signal GPIO14
//   RGB LED (common cathode): R -> 220 ohm -> GPIO21, G -> 220 ohm -> GPIO47,
//                             B -> 220 ohm -> GPIO1, long leg (cathode) -> GND

#include <Arduino.h>
#include <ESP_I2S.h>
#include "esp_heap_caps.h"
#define SS_ALLOC(n) heap_caps_malloc((n), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
#include "ss_core.h"

// ---------------- pins ----------------
const int PIN_SCK = 41, PIN_WS = 42, PIN_SD = 2;      // microphone
const int PIN_SERVO = 14;                              // servo signal
const int PIN_R = 21, PIN_G = 47, PIN_B = 1;           // external RGB LED
const int PIN_ONBOARD = 48;                            // on-board RGB LED (WS2812) = wake indicator
const bool RGB_COMMON_ANODE = false;                   // set true for a common-anode LED (long leg to 3V3)

// ---------------- per-class alert look ----------------
// colours: alarm red, siren blue, doorbell green, knock yellow, baby cry pink, car horn cyan
const uint8_t COLOR[SS_CLASSES][3] = {{255, 0, 0}, {0, 0, 255}, {0, 255, 0},
                                      {255, 160, 0}, {255, 0, 120}, {0, 255, 255}};
// vibration rhythms: {on ms, off ms, repeats}
//   alarm 3 x long-ish, siren 1 long, doorbell 2 short, knock 3 quick, baby 2 long, horn 4 fast
const uint16_t RHYTHM[SS_CLASSES][3] = {{400, 200, 3}, {1500, 0, 1}, {150, 150, 2},
                                        {100, 100, 3}, {700, 300, 2}, {100, 80, 4}};
const uint32_t COOLDOWN_MS = 6000;                     // same class alerts at most every 6 s
const uint32_t COLOR_HOLD_MS = 4000;                   // RGB LED stays on this long after an alert

I2SClass i2s;
static float gainDb = 0.0f;

// ring of log-mel frames; the network copies the last 100 when a window is due
static float ring[SS_FRAMES][SS_MELS];
static volatile int ringPos = 0;
static volatile uint32_t frameCount = 0;
static float window_[SS_MELS * SS_FRAMES];
static TaskHandle_t netTask;
static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
static volatile bool netBusy = false;
static volatile uint32_t skipped = 0;
static QueueHandle_t alertQueue;                       // class indices to act on
static volatile uint32_t awakeUntil = 0;               // millis() until which the wake LED is on

struct HP {                                             // 2nd-order Butterworth high-pass, 30 Hz
  float b0, b1, b2, a1, a2, z1 = 0, z2 = 0;
  HP(float fc, float fs) {
    float w = 2 * PI * fc / fs, c = cosf(w), al = sinf(w) / (2 * 0.7071f), a0 = 1 + al;
    b0 = (1 + c) / 2 / a0; b1 = -(1 + c) / a0; b2 = b0; a1 = -2 * c / a0; a2 = (1 - al) / a0;
  }
  float step(float x) { float y = b0 * x + z1; z1 = b1 * x - a1 * y + z2; z2 = b2 * x - a2 * y; return y; }
} hp(30.0f, SS_SR);

// ---------------- outputs ----------------
static void rgb(uint8_t r, uint8_t g, uint8_t b) {
  if (RGB_COMMON_ANODE) { r = 255 - r; g = 255 - g; b = 255 - b; }
  ledcWrite(PIN_R, r); ledcWrite(PIN_G, g); ledcWrite(PIN_B, b);
}

static void servoAngle(int deg) {                       // 50 Hz, 0.5-2.4 ms pulse, 14-bit duty
  uint32_t us = 500 + (uint32_t)deg * 1900 / 180;
  ledcWrite(PIN_SERVO, us * 16384 / 20000);
}

static void buzz(uint16_t ms) {                         // "vibrate": swing the servo back and forth
  uint32_t end = millis() + ms;
  bool side = false;
  while ((int32_t)(end - millis()) > 0) {
    servoAngle(side ? 70 : 110);
    side = !side;
    vTaskDelay(pdMS_TO_TICKS(70));                      // SG90 needs ~60-80 ms for a 40 deg swing
  }
  servoAngle(90);
}

static void alertTask(void *) {
  int k;
  uint32_t colorUntil = 0;
  for (;;) {
    if (xQueueReceive(alertQueue, &k, pdMS_TO_TICKS(100)) == pdTRUE) {
      rgb(COLOR[k][0], COLOR[k][1], COLOR[k][2]);
      for (int r = 0; r < RHYTHM[k][2]; r++) {
        buzz(RHYTHM[k][0]);
        if (RHYTHM[k][1]) vTaskDelay(pdMS_TO_TICKS(RHYTHM[k][1]));
      }
      colorUntil = millis() + COLOR_HOLD_MS;
      xQueueReset(alertQueue);                          // drop alerts that queued while buzzing
    }
    if (colorUntil && (int32_t)(millis() - colorUntil) > 0) { rgb(0, 0, 0); colorUntil = 0; }
    bool awake = (int32_t)(awakeUntil - millis()) > 0;  // wake indicator on the on-board LED
    rgbLedWrite(PIN_ONBOARD, awake ? 60 : 0, awake ? 60 : 0, awake ? 60 : 4);
  }
}

static void trigger(int k) {
  static uint32_t last[SS_CLASSES] = {0};
  uint32_t now = millis();
  if (last[k] && now - last[k] < COOLDOWN_MS) return;
  last[k] = now;
  xQueueSend(alertQueue, &k, 0);
}

// ---------------- audio ----------------
static void audioTask(void *) {
  static int32_t raw[SS_HOP];
  static float buf[SS_NFFT];                            // last 512 samples
  float mel[SS_MELS];
  int filled = 0;
  float floorDb = -60.0f;                               // tracked background level
  const float rise = 6.0f * SS_HOP / SS_SR / 2.0f;      // floor rises 6 dB per 2 s
  for (;;) {
    size_t want = sizeof(raw), got = 0;
    while (got < want) got += i2s.readBytes((char *)raw + got, want - got);
    float g = powf(10.0f, gainDb / 20.0f), e = 0;
    memmove(buf, buf + SS_HOP, (SS_NFFT - SS_HOP) * sizeof(float));
    for (int i = 0; i < SS_HOP; i++) {
      float v = hp.step((raw[i] >> 8) / 8388608.0f) * g;
      buf[SS_NFFT - SS_HOP + i] = v;
      e += v * v;
    }
    // wake indicator: level 10 dB above the background floor, held 2 s
    float lvl = 10.0f * log10f(e / SS_HOP + 1e-12f);
    floorDb = lvl < floorDb ? lvl : floorDb + rise;
    if (lvl - floorDb > 10.0f) awakeUntil = millis() + 2000;

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

// ---------------- network ----------------
static void networkTask(void *) {
  float prob[SS_CLASSES];
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    netBusy = true;
    portENTER_CRITICAL(&mux);
    int start = ringPos;
    for (int t = 0; t < SS_FRAMES; t++)
      for (int m = 0; m < SS_MELS; m++) window_[m * SS_FRAMES + t] = ring[(start + t) % SS_FRAMES][m];
    uint32_t fc = frameCount;
    portEXIT_CRITICAL(&mux);

    uint32_t t0 = micros();
    ss_net(window_, prob);
    uint32_t ms = (micros() - t0) / 1000;

    // the most confident class above its threshold alerts (one alert per window)
    int best = -1;
    for (int k = 0; k < SS_CLASSES; k++)
      if (prob[k] >= SS_THRESH[k] && (best < 0 || prob[k] / SS_THRESH[k] > prob[best] / SS_THRESH[best])) best = k;
    if (best >= 0) trigger(best);

    bool awake = (int32_t)(awakeUntil - millis()) > 0;
    Serial.printf("t=%7.1f net=%4ums %s |", fc * 0.02f, ms, awake ? "AWAKE" : "idle ");
    for (int k = 0; k < SS_CLASSES; k++) Serial.printf(" %s %.2f", SS_NAMES[k], prob[k]);
    if (best >= 0) Serial.printf("  >>> ALERT %s", SS_NAMES[best]);
    if (skipped) Serial.printf("  (skipped %u)", skipped);
    Serial.println();
    netBusy = false;
  }
}

void setup() {
  Serial.begin(921600);
  delay(1500);
  Serial.println("\nSafeSound detector + servo + RGB");
  Serial.printf("model: width %d, blocks %d; PSRAM %u bytes\n", SS_WIDTH, SS_BLOCKS, ESP.getPsramSize());

  ledcAttach(PIN_SERVO, 50, 14);
  servoAngle(90);
  ledcAttach(PIN_R, 5000, 8); ledcAttach(PIN_G, 5000, 8); ledcAttach(PIN_B, 5000, 8);
  rgb(0, 0, 0);
  pinMode(PIN_ONBOARD, OUTPUT);
  rgbLedWrite(PIN_ONBOARD, 0, 0, 4);

  if (!ss_init()) { Serial.println("ERROR: no PSRAM for the network buffers"); while (true) delay(1000); }
  i2s.setPins(PIN_SCK, PIN_WS, -1, PIN_SD);
  if (!i2s.begin(I2S_MODE_STD, SS_SR, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO, I2S_STD_SLOT_LEFT)) {
    Serial.println("ERROR: I2S did not start"); while (true) delay(1000);
  }
  alertQueue = xQueueCreate(4, sizeof(int));
  xTaskCreatePinnedToCore(alertTask, "alert", 4096, nullptr, 2, nullptr, 0);
  xTaskCreatePinnedToCore(networkTask, "net", 16384, nullptr, 1, &netTask, 0);
  xTaskCreatePinnedToCore(audioTask, "audio", 16384, nullptr, 5, nullptr, 1);

  // short start-up check of the outputs: each colour + a tiny buzz
  for (int k = 0; k < SS_CLASSES; k++) { rgb(COLOR[k][0], COLOR[k][1], COLOR[k][2]); delay(150); }
  rgb(0, 0, 0);
  buzz(200);
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '+') gainDb += 6;
    if (c == '-') gainDb -= 6;
    if (c == '+' || c == '-') Serial.printf("gain %+.0f dB\n", gainDb);
    if (c >= '0' && c <= '5') {
      int k = c - '0';
      Serial.printf("test alert: %s\n", SS_NAMES[k]);
      xQueueSend(alertQueue, &k, 0);
    }
  }
  delay(20);
}
