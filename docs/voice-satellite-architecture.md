# Voice Satellite Architecture

Status: early design notes, September 2026

## Direction

The system will have two satellite classes connected to the core computer that
runs speech-to-text, intent routing, the agent, and text-to-speech:

1. **Smart satellite** — wake-word detection, microphone capture, speaker
   playback, touchscreen display, and a local user interface that can collect
   input and send it to the core computer.
2. **Audio satellite** — wake-word detection, microphone capture, and speaker
   playback. It has no touchscreen or local UI.

The satellites should remain thin clients. Conversation state, routing,
skills, model inference, and durable memory belong on the core computer.

## Shared satellite capabilities

- Wi-Fi network connection
- Wake-word detection, preferably locally
- Start/stop recording and end-of-speech detection
- Upload recorded audio to the core computer
- Receive and play synthesized audio
- Microphone mute control with a visible status indicator
- Status states such as idle, listening, processing, speaking, and error
- Versioned device protocol with capability discovery

The protocol should allow the core to discover whether a device supports
`display`, `touch`, `local_ui`, and other optional capabilities rather than
assuming every satellite has the same hardware.

## Smart satellite

The smart satellite is the primary exploratory device. Its touchscreen should
be able to:

- Show listening, processing, and speaking status
- Display the current response or a compact conversation view
- Collect typed input when speech is inconvenient
- Present buttons, confirmations, choices, and skill-specific controls
- Send UI events to the core as ordinary user input or structured events

The display UI is an interaction surface, not a second agent. The core remains
the source of truth for conversation and action state.

The planned D-Link Boxee enclosure has a useful mechanical advantage: its
front grille and other outer panels are detachable. The smart satellite can
therefore use a replacement screen carrier/front panel rather than cutting the
main enclosure. The carrier should retain separate acoustic openings or
positions for the microphone array and speaker, with the display mounted flush
from behind where possible.

### Korvo-1 microphone-placement constraint

The ESP32-S3-Korvo-1 is a strong smart-satellite candidate because it combines
the ESP32-S3, far-field audio capture, codec, amplifier, speaker output, and
wake-word support. Its microphone array is on a separate FPC-connected
sub-board, so the microphone PCB can be mounted separately from the main board
inside the enclosure. However, the three microphones are fixed on that PCB at
a fixed spacing; they cannot be distributed freely around the enclosure
without modifying or replacing the array board.

If independent microphone placement is a firm requirement, use a conventional
ESP32-S3 controller with separate matched digital microphone breakouts instead.

Likely hardware characteristics:

- ESP32-S3-class controller
- At least 8 MB PSRAM for audio buffers and UI assets
- Digital microphone or microphone array
- Full-duplex I²S audio path or an audio codec
- Small speaker with an amplifier
- Touchscreen connected over SPI, RGB, or a board-specific display bus

For the Boxee front panel, a round 4-inch 480x480 capacitive display is also
under consideration. The Waveshare ESP32-S3-Touch-LCD-4 is a useful reference
for this format: it uses a round RGB display and GT911 capacitive touch, with
first-party LVGL examples. A standalone equivalent connected to the
DevKitC would need the ESP32-S3 RGB LCD peripheral and many GPIOs; it would
not be a simple SPI display connection.

### Optional dual-controller arrangement

The ESP32-S3-Korvo-1 and a Waveshare round-display ESP32-S3 board can coexist
as two cooperating controllers. The Korvo can own the microphone array, wake
word, recording, and speaker playback, while the display board owns LVGL,
touch input, and screen rendering. A short 3.3 V UART link (TX, RX, and GND)
is the preferred first interconnect; messages can carry capability discovery,
listening/processing/speaking state, response text, and touch events. Both
boards can share one 5 V supply through separate power branches.

This arrangement preserves the Korvo audio hardware but adds firmware and
protocol complexity. It is a fallback if a suitable standalone round display
cannot be found for the DevKitC.

The same pattern can use a simpler ESP32-S3 audio controller instead of the
Korvo. In that version, the display board is a dedicated UI coprocessor and
the audio board is selected independently for microphone quality, amplifier
support, enclosure placement, and cost. This may be the preferred smart-
satellite architecture if the display board's own ESP32 has the required UI
performance but its onboard audio hardware is unsuitable.

### Enclosure controls and local services

The smart satellite should have a master power switch on the rear or underside
of the enclosure. The audio/controller board's `RESET` and `BOOT` controls can
be extended to two normally-open panel buttons: each button connects its
control pin to GND. `BOOT` is only held while resetting when firmware download
mode is required. The display ESP32 has its own independently programmed
`RESET` and `BOOT` controls, so it should either have a second concealed service
pair or remain accessible through its USB auto-download support.

Espressif's ESP32-S3-BOX-3 is a useful reference platform because it combines
an ESP32-S3, dual microphones, speaker, touchscreen, USB-C programming, and
existing voice-assistant support. See the [ESP32-S3-BOX-3 documentation](https://www.espressif.com/en/node/6980).

## Audio satellite

Audio satellites should use the same audio and status protocol but omit all
display-specific behavior. They may expose only physical buttons or LEDs.

The audio path should support streaming in the future, even if the first
implementation sends one complete recording. Streaming will reduce perceived
latency for both recognition and response playback.

## Candidate audio hardware

- **ReSpeaker Lite + XIAO ESP32S3**: strong candidate where far-field audio is
  important. The ReSpeaker Lite provides a dual-microphone array and XMOS
  processing with echo cancellation, noise suppression, and interference
  cancellation, with USB and I²S modes. See the [ReSpeaker Lite project](https://github.com/respeaker/ReSpeaker_Lite).
- **M5Stack Atom VoiceS3R**: strong candidate where a compact integrated form
  factor matters. It provides an ESP32-S3, 8 MB flash, 8 MB PSRAM, MEMS
  microphone, ES8311 codec, amplifier, and speaker support. See the [Atom VoiceS3R documentation](https://shop.m5stack.com/products/atom-echos3r-smart-speaker-dev-kit).

The board choice is intentionally open. Audio quality, echo cancellation,
touchscreen support, enclosure size, availability, and firmware support should
be tested before standardizing on one design.

## Core connection

The existing core exposes voice endpoints that accept audio and return the
transcription/result/audio pipeline. The first satellite integration can use a
simple request/response exchange:

1. Satellite detects the wake word locally.
2. Satellite records until silence or a stop action.
3. Satellite sends WAV/PCM audio to the core.
4. Core transcribes, routes, runs the agent, and synthesizes a response.
5. Satellite plays the returned audio and updates its status.

The later protocol should add streamed audio input/output, cancellation or
barge-in, authentication, device identity, protocol version, and capability
negotiation.

## Open decisions

- Select the first smart-satellite board and touchscreen.
- Decide whether wake-word detection runs on the ESP32-S3, an audio DSP, or
  both.
- Choose the UI framework (for example LVGL or a board-supported framework).
- Define the wire format and transport for audio and UI events.
- Decide whether satellites connect directly to the core over the LAN or via a
  message broker.
- Design authentication and encryption before placing microphones around the
  home.
- Measure end-to-end latency and speaker-to-microphone echo in a real room.
