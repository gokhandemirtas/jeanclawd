# Voice Satellite Prototype BOM

Status: preliminary component list, September 2026

This is a component-level prototype BOM. It deliberately does not use an
all-in-one voice-assistant board. Buy one complete set for each satellite;
omit the touchscreen items for audio-only satellites.

## Required for every satellite

| Qty | Component | Recommended specification | Purpose |
| ---: | --- | --- | --- |
| 1 | ESP32-S3 development board | [Espressif ESP32-S3-DevKitC-1-N8R8 from Rapid Electronics](https://www.rapidonline.com/espressif-esp32-s3-devkitc-1-n8r8-pcb-board-wi-fi-bt-usb-type-c-devkit-09-1106); genuine board with 8 MB flash and 8 MB PSRAM | Wi-Fi/Bluetooth controller and firmware target |
| 2 | Digital MEMS microphone breakout | Genuine ICS-43434 or equivalent 24-bit I²S microphone breakout; one configured LEFT and one RIGHT | Two-channel microphone array |
| 1 | I²S Class-D amplifier breakout | MAX98357A, not MAX98357B, with exposed BCLK, LRCLK, DIN, VIN, and GND | Converts digital audio to speaker power; the A variant accepts standard I²S |
| 1 | Speaker | 4 Ω, 3 W, approximately 36–45 mm | Audio output |
| 1 | 5 V USB-C power supply | Regulated 5 V, at least 2 A | Power for the ESP32 board and amplifier |
| 1 | USB-C data cable | Known data-capable cable | Flashing and serial logs |
| 1 | Panel-mount USB-C extension | Short USB 2.0-rated female-to-male USB-C panel extension; match the DevKitC connector and keep the internal cable short | Brings programming/power access to the enclosure exterior |
| 1 | Prototype board | Solderless breadboard initially, then perfboard | Wiring and bring-up |
| 1 | Wire set | 26–30 AWG stranded wire, preferably silicone insulated | Audio, power, and GPIO wiring |
| 1 | Decoupling capacitor kit | 0.1 µF and 100 µF electrolytic capacitors | Local power filtering near the ESP32 and amplifier |
| 1 | Physical microphone-mute switch | Latching SPST or pushbutton plus a visible red LED | Hardware privacy control |
| 1 | Status indicator | 5 V WS2812B or SK6812 addressable RGB LED strip, ideally 10–30 pixels for the enclosure | Idle/listening/processing/speaking/error animations |
| 1 | Logic-level shifter | 74AHCT125 or 74HCT245 breakout/module | Converts ESP32-S3 3.3 V data to reliable 5 V LED data |
| 1 | LED protection capacitor | 500–1000 µF electrolytic, rated at least 6.3 V | Buffers LED current changes at the strip input |
| 1 | LED data resistor | 330–500 Ω | Series resistor close to the strip's DIN input |
| 2–3 | Tactile pushbuttons | Momentary, 3.3 V GPIO-compatible | Action, mute, and optional volume/input controls |
| 2 | Remote boot/reset buttons | Normally-open momentary panel buttons wired to the DevKitC `BOOT` and `RESET` controls; use a second pair if the display ESP32 is also inaccessible | Allows firmware flashing and recovery after enclosure assembly |

The ICS-43434 connects directly to a microcontroller over I²S and includes
the microphone, ADC, filtering, and digital interface in one package. Its
datasheet documents the LEFT/RIGHT channel selection used to put two devices
on one I²S data line. [ICS-43434 datasheet](https://product.tdk.com/en/search/set_distributor?back_url=%2Fsystem%2Ffiles%2Fdam%2Fdoc%2Fproduct%2Fsw_piezo%2Fmic%2Fmems-mic%2Fdata_sheet%2Fds-000069-ics-43434-v1.2.pdf&ref_disty=mouser)

The MAX98357A is a suitable separate amplifier because it accepts standard
I²S, requires no external MCLK, runs from 2.5–5.5 V, and can drive up to 3.2 W
into 4 Ω at 5 V. [MAX98357A product information](https://www.analog.com/en/products/max98357a.html)

## Additional components for the smart satellite

| Qty | Component | Recommended specification | Purpose |
| ---: | --- | --- | --- |
| 1 | TFT LCD module | 2.8 in, 320×240, ST7789 or ILI9341, 3.3 V logic, SPI breakout | Main touchscreen display |
| 1 | Touch controller/module | XPT2046 resistive touch module for easiest first prototype, or CST816S capacitive touch breakout | Touch input |
| 1 | Display wiring set | Short 28–30 AWG wires or a compatible FFC breakout | SPI and touch connections |
| 1 | Controller-to-display link | 3-conductor cable or three jumper wires for `TX`, `RX`, and `GND`; use locking JST connectors for the permanent assembly | UART messages between the audio and UI ESP32 boards |
| 1 | Enclosure or front panel | Sized around the selected display and 40 mm speaker | Mechanical assembly |
| 1 | Optional rotary encoder | EC11 encoder with push switch | Useful physical navigation/volume control |

For the first UI prototype, an ILI9341 plus XPT2046 is intentionally preferred
over a high-resolution display: it has mature ESP-IDF/LVGL support and keeps
the wiring and memory requirements modest. A capacitive display can replace it
once the UI layout is settled.

### Korvo-1 alternative

The ESP32-S3-Korvo-1 can replace the controller, microphone, codec, amplifier,
and speaker-interface rows for the smart satellite. Its microphone sub-board is
physically separate from the main board via FPC, but its three microphones are
fixed on that sub-board at a fixed spacing. It supports separation of the
microphone PCB from the controller, not arbitrary placement of each microphone.
Keep the separate-controller plus digital-microphone design if the microphones
must be positioned independently in the enclosure.

### Budget smart-satellite build

For a target electronics cost below £50, use a general ESP32-S3 board and
separate audio/display parts rather than the Korvo-1:

- ESP32-S3 board with 8 MB PSRAM: approximately £8–15
- 2.8-inch SPI TFT with XPT2046 touch: approximately £10–15
- One I²S MEMS microphone initially: approximately £3–6
- MAX98357A I²S amplifier: approximately £3–6
- 4-ohm, 3-watt speaker: approximately £3–5
- Wiring, prototype board, mute switch, and status LED: approximately £8–12

This totals approximately £35–59 depending on supplier and whether one or two
microphones are fitted. The first prototype should use one microphone to stay
comfortably below £50; a second matched microphone can be added later for the
audio front end, beamforming, and echo-cancellation experiments. This estimate
excludes the reused enclosure, tools, and mains power supply.

## Optional improvements

- A third or fourth matched digital microphone for better directionality and
  far-field experiments.
- An I²S audio codec if a line-level input, headphone output, or more advanced
  full-duplex audio routing is needed.
- A dedicated audio DSP or XMOS front-end when echo cancellation remains poor.
- Ferrite bead or LC filtering between the amplifier supply and the digital
  electronics if speaker noise appears in recordings.
- USB logic analyzer for checking I²S clocks, channel selection, and display
  signals.
- Oscilloscope or multimeter for power-rail and amplifier debugging.
- JST-PH or JST-SH connectors for the move from breadboard to perfboard/custom
  PCB.

## Compatibility and wiring constraints

- Keep the prototype at 3.3 V logic. Do not connect 5 V GPIO signals to the
  ESP32-S3.
- Feed the MAX98357A from 5 V when using a 4 Ω speaker; observe the breakout's
  documented power and ground connections.
- Share BCLK and LRCLK between the two microphones, set one microphone LEFT
  and the other RIGHT, and connect their shared I²S data output as documented
  by the microphone breakout.
- Keep the microphone wiring physically away from the amplifier and speaker
  wiring.
- The physical mute switch should disable microphone power or data in hardware
  where possible, not merely send a software flag.
- Do not buy the ESP32-S3 N16R8V variant for the first prototype without
  checking voltage and peripheral requirements; the N8R8 board is simpler for
  a 3.3 V breakout-based design.

## First purchase recommendation

For one smart satellite, purchase one ESP32-S3-DevKitC-1-N8R8, two genuine
ICS-43434 breakout boards, one MAX98357A breakout, one 4 Ω 3 W speaker, one
ILI9341/XPT2046 2.8-inch display assembly, plus the power, wiring, buttons,
LED, and prototyping supplies above. Build and test the audio path before
adding the display.
