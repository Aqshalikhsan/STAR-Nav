/* ppm_trainer.ino ------------------------------------------------------------
 * Laptop  --USB serial-->  Arduino  --PPM on D9-->  radio TRAINER port  --RF-->  drone
 *
 * The Arduino is the "translator box" that turns channel numbers coming from the
 * laptop (from keyboard_control.py, or later the STAR-Nav policy) into a PPM
 * (CPPM) signal your radio's trainer port understands, so the radio transmits
 * those sticks over its normal ELRS/RF link. See ../../README.md for wiring.
 *
 * WIRING (only 2 wires to the radio):
 *   Arduino D9  --------> 3.5mm jack TIP     (PPM signal in)
 *   Arduino GND --------> 3.5mm jack SLEEVE  (ground / common)
 *   Arduino USB --------> laptop             (power + serial packets)
 *
 * SAFETY: if no valid packet arrives for FAILSAFE_MS, the outputs snap to the
 * FAILSAFE values (throttle MIN, ARM channel LOW = disarmed). Always keep the
 * radio's trainer switch as a manual override, and test PROPS OFF first.
 *
 * SERIAL PACKET (little-endian), 19 bytes, streamed ~50 Hz by the laptop:
 *   0xA5 0x5A | ch0_lo ch0_hi | ... | ch7_lo ch7_hi | XOR(payload)
 *   each ch = microseconds (1000..2000, 1500 = centre)
 * -------------------------------------------------------------------------- */

#define CHANNEL_NUMBER  8       // number of channels in the PPM frame
#define SIG_PIN         9       // PPM output pin -> jack TIP
#define FRAME_LENGTH    22500   // total PPM frame length (us)
#define PULSE_LENGTH    300     // separator pulse length (us)
#define ON_STATE        1       // 1 = normal PPM (idle low). Set 0 if your radio needs INVERTED PPM.
#define BAUD            115200
#define FAILSAFE_MS     500     // no packet for this long -> failsafe

// Failsafe / boot values. Order here is the CHANNEL ORDER the laptop must match
// (default AETR + arm): 0=roll 1=pitch 2=throttle 3=yaw 4=arm 5..7=aux.
uint16_t failsafe[CHANNEL_NUMBER] = {1500, 1500, 1000, 1500, 1000, 1500, 1500, 1500};

volatile uint16_t ppm[CHANNEL_NUMBER];
unsigned long lastPacketMs = 0;

// --- serial framing state machine ---
enum { WAIT_H1, WAIT_H2, READ_PAYLOAD, READ_CKSUM };
uint8_t st = WAIT_H1;
uint8_t payload[CHANNEL_NUMBER * 2];
uint8_t pidx = 0;

void setPPM(const uint16_t *src) {
  noInterrupts();                       // 16-bit writes aren't atomic on AVR
  for (uint8_t i = 0; i < CHANNEL_NUMBER; i++) ppm[i] = src[i];
  interrupts();
}

void setup() {
  setPPM(failsafe);
  pinMode(SIG_PIN, OUTPUT);
  digitalWrite(SIG_PIN, !ON_STATE);
  Serial.begin(BAUD);

  // Timer1, CTC, prescaler /8 -> 0.5 us per tick (so us*2 = ticks).
  noInterrupts();
  TCCR1A = 0; TCCR1B = 0; TCNT1 = 0;
  OCR1A = 200;                 // first compare
  TCCR1B |= (1 << WGM12);      // CTC mode
  TCCR1B |= (1 << CS11);       // prescaler /8
  TIMSK1 |= (1 << OCIE1A);     // enable compare interrupt
  interrupts();
  lastPacketMs = millis();
}

// PPM generator (classic David Hasko pattern). Runs entirely in the ISR so the
// timing is jitter-free regardless of what loop() is doing.
ISR(TIMER1_COMPA_vect) {
  static boolean state = true;
  TCNT1 = 0;
  if (state) {                          // start separator pulse
    digitalWrite(SIG_PIN, ON_STATE);
    OCR1A = PULSE_LENGTH * 2;
    state = false;
  } else {                              // end pulse, schedule next
    static uint8_t  cur = 0;
    static uint16_t rest = 0;
    digitalWrite(SIG_PIN, !ON_STATE);
    state = true;
    if (cur >= CHANNEL_NUMBER) {        // frame sync gap = leftover time
      cur = 0;
      rest += PULSE_LENGTH;
      OCR1A = (FRAME_LENGTH - rest) * 2;
      rest = 0;
    } else {
      OCR1A = (ppm[cur] - PULSE_LENGTH) * 2;
      rest += ppm[cur];
      cur++;
    }
  }
}

void loop() {
  while (Serial.available()) {
    uint8_t b = Serial.read();
    switch (st) {
      case WAIT_H1:  st = (b == 0xA5) ? WAIT_H2 : WAIT_H1; break;
      case WAIT_H2:  if (b == 0x5A) { st = READ_PAYLOAD; pidx = 0; } else st = WAIT_H1; break;
      case READ_PAYLOAD:
        payload[pidx++] = b;
        if (pidx >= CHANNEL_NUMBER * 2) st = READ_CKSUM;
        break;
      case READ_CKSUM: {
        uint8_t ck = 0;
        for (uint8_t i = 0; i < CHANNEL_NUMBER * 2; i++) ck ^= payload[i];
        if (ck == b) {                  // valid packet
          uint16_t tmp[CHANNEL_NUMBER];
          for (uint8_t i = 0; i < CHANNEL_NUMBER; i++) {
            uint16_t v = payload[2 * i] | ((uint16_t)payload[2 * i + 1] << 8);
            if (v < 900)  v = 900;      // clamp to a sane PPM range
            if (v > 2100) v = 2100;
            tmp[i] = v;
          }
          setPPM(tmp);
          lastPacketMs = millis();
        }
        st = WAIT_H1;
        break;
      }
    }
  }
  if (millis() - lastPacketMs > FAILSAFE_MS) setPPM(failsafe);  // link lost -> safe
}
