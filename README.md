# Webcam Heart-Rate Monitor (rPPG)

Read your pulse from a plain webcam — no contact sensors, just the camera.

## Quick start

```bash
git clone https://github.com/rt9pdm2vsw-glitch/HeartRate.git
cd HeartRate
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Face the camera in steady light, hold still for ~5–10 seconds, and your
heart rate appears. Press `m` to switch between the POS algorithm and the
green-channel baseline, `e` for the pulse-magnification view, `q` to quit.

> On macOS, allow camera access when prompted (the app waits for you).
> **This is a tech demo, not a medical device** — see
> [Limitations](#limitations--please-read).

This is **remote photoplethysmography (rPPG)**: every heartbeat pushes a
pulse of blood through the capillaries just under your skin, which very
slightly changes the skin's colour. A webcam can pick up that tiny,
periodic colour shift. Average the skin colour over a well-perfused region
of your face, filter it to the plausible heart-rate band, and the dominant
frequency *is* your heart rate.

---

## The science

**Why the camera can see your pulse.** Oxygenated haemoglobin in blood
absorbs light. When your heart beats, blood volume in the skin's
capillaries rises and falls at the cardiac frequency, modulating how much
light the skin absorbs versus reflects. The camera records this as a small
oscillation in the average pixel colour of your skin — well below what the
eye notices, but well within the sensor's reach once you average over
thousands of pixels and several seconds.

**Why green light carries the strongest signal.** Haemoglobin's light
absorption peaks in the green part of the visible spectrum (~540 nm).
Green light also penetrates skin to roughly the depth of the capillary
bed. So the green channel shows the largest pulse-synchronous variation —
larger than red (which penetrates deeper and is less modulated) or blue
(which is scattered near the surface and noisier). That's why the simplest
possible rPPG method is just "track the mean green channel", and why it's
the baseline here.

**Why POS beats raw green.** The green channel is strongest but also
fragile: any motion or lighting change moves all three colour channels
together (a *specular* / intensity change), and that swamps the tiny
pulse. **POS (Plane-Orthogonal-to-Skin, Wang et al. 2017)** exploits the
fact that the *pulse* moves the RGB values along a specific, known
direction in colour space (the blood-volume direction), while motion and
lighting move them along the intensity direction. POS projects the
normalized RGB signal onto a plane **orthogonal to the skin-tone /
intensity axis**, then combines the two in-plane projections so that the
common-mode intensity noise cancels and the pulse survives. The result is
far more robust to motion than the raw green channel. This app implements
POS and lets you toggle to the green baseline to compare.

**Related idea — making the flush visible.** You can't normally *see* the
colour change, but you can amplify it. **Eulerian Video Magnification (Wu
et al. 2012)** temporally bandpasses each pixel's colour and amplifies the
result, so the subtle per-beat flush becomes visible on screen. The
optional "magnify" view in this app is a simplified, real-time
approximation of that idea (press `e`). It's a visualization only — the
heart-rate number never depends on it.

### References
- Wang, den Brinker, Stuijk & de Haan (2017), *Algorithmic Principles of
  Remote PPG*, IEEE Trans. Biomedical Engineering 64(7). — the POS method.
- Wu, Rubinstein, Shih, Guttag, Durand & Freeman (2012), *Eulerian Video
  Magnification for Revealing Subtle Changes in the World*, ACM TOG. — the
  "make the flush visible" idea.

---

## How it works (pipeline)

1. **Capture.** Grab webcam frames and timestamp each one with
   `time.perf_counter()`. Real webcams do **not** deliver a steady 30 fps;
   assuming a fixed rate biases the frequency estimate, so we record the
   true time of every frame.
2. **Face + ROI.** MediaPipe tracks the face mesh; we sample the
   **forehead and both cheeks** (well-perfused, usually-exposed skin) and
   average the pixels inside to get one `(R, G, B)` per frame. Both the
   modern MediaPipe Tasks API (≥ 1.0) and the legacy FaceMesh API (0.10.x)
   are supported — with ≥ 1.0 the ~3.7 MB landmark model auto-downloads on
   first run and is cached. If MediaPipe can't be installed at all, we fall
   back to OpenCV's Haar face detector with a fixed forehead box (the
   cascade file is vendored, since OpenCV 5 wheels no longer bundle it).
3. **Rolling buffer.** Keep the last ~10 seconds of `(t, R, G, B)`.
4. **Resample.** Interpolate the irregularly-timed samples onto a uniform
   30 Hz grid so the jittery frame timing doesn't corrupt the FFT.
5. **Pulse extraction.** POS by default (motion-robust), or the
   green-channel mean as a selectable baseline.
6. **Detrend + bandpass.** Linear detrend, then a Butterworth bandpass to
   **0.7–4.0 Hz (42–240 bpm)**.
7. **Heart-rate estimate.** Hann-window the buffer, zero-padded FFT for
   resolution, find the peak frequency in the band, refine it with
   parabolic interpolation for sub-bin precision, and `BPM = peak × 60`.
8. **Confidence / SNR.** Measure the power near the peak (and its first
   harmonic) versus the rest of the in-band power. When SNR is low, the BPM
   is greyed out and shows "acquiring…" so it never displays a confident
   wrong number.

---

## Install

Python 3.9+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `mediapipe` fails to install on your platform, the app still runs — it
automatically uses the OpenCV Haar fallback. You can also force it with
`--no-mediapipe`.

**First run with MediaPipe ≥ 1.0:** the face-landmark model
(`face_landmarker.task`, ~3.7 MB) is downloaded once from Google's model
hosting and cached in `~/.cache/rppg-heartrate/` (override with
`RPPG_CACHE_DIR`). If the download fails (offline), the app falls back to
the Haar tracker, which needs no download.

## Run

```bash
python main.py
```

Options:

```
--camera N        webcam index (default 0)
--width / --height  capture resolution (default 640x480)
--method pos|green  initial pulse method (default pos)
--no-mediapipe    force the OpenCV Haar fallback
```

### Keys
| key | action |
|-----|--------|
| `m` | toggle pulse method (POS ↔ green-channel baseline) |
| `e` | toggle the Eulerian magnification view (stretch feature) |
| `q` / `Esc` | quit |

The window shows your webcam with the tracked ROI drawn on your face, a
live scrolling plot of the filtered pulse waveform, a large BPM readout,
and a small SNR/confidence bar.

---

## Getting a clean reading

- **Even, steady lighting.** Avoid flickering lights (some indoor lighting
  flickers at 50/60 Hz or its subharmonics) and strong changing shadows.
  Diffuse daylight or a steady lamp works best.
- **Hold still.** Motion is the biggest enemy. Face the camera and keep
  your head steady. POS tolerates some motion; the green baseline does not.
- **Face the camera** so the forehead and cheeks are clearly visible and
  well lit. Remove glasses glare if it covers the ROI.
- **Wait a few seconds.** The estimate needs ~5–10 s of buffered signal
  before it locks on; it shows "acquiring…" until the SNR is high enough.

---

## Tests

Two layers, neither needing a camera:

- **Signal-processing unit tests** (`tests/test_signal_processing.py`) —
  synthetic RGB signals with a known pulse frequency go through the
  estimator; the recovered BPM must be within ±2 bpm. Needs only
  numpy/scipy/pytest.
- **Vision integration tests** (`tests/test_integration_vision.py`) — a
  synthetic pulsing "video" built from a real face photo runs through the
  *actual* face trackers, rolling buffer, and estimator end-to-end, and
  must also recover the pulse within ±2 bpm. These skip automatically if
  OpenCV/MediaPipe or the network are unavailable.

```bash
pip install -r requirements-dev.txt   # minimal: signal tests only
python -m pytest
# with the full requirements.txt installed, the integration tests run too
```

---

## Limitations — please read

This is a **demonstration, not a medical device.**

- **Sensitive to motion.** Head movement, talking, and expression changes
  inject noise that can dominate the pulse.
- **Sensitive to lighting.** Flickering or changing light can create false
  peaks; uneven light reduces signal quality.
- **Accuracy varies with skin tone and lighting.** rPPG relies on light
  reflected from the skin; signal strength and reliability differ across
  skin tones and lighting conditions, and the method can be less reliable
  for darker skin tones under poor lighting. Treat all readings as
  approximate.
- **Not diagnostic.** Do not use this for any medical decision. If you have
  concerns about your heart rate or rhythm, use a validated medical device
  and consult a professional.

The SNR gate is there specifically so the app prefers to say "acquiring…"
rather than show a confident wrong number — but a plausible-looking number
is still only as good as your lighting and stillness.

---

## Project layout

```
main.py                     desktop app: capture -> ROI -> estimate -> UI
rppg/
  signal_processing.py      pure numpy/scipy core (resample, POS, FFT, SNR)
  roi.py                    MediaPipe (Tasks + legacy) ROI, OpenCV Haar fallback
  assets.py                 model download/cache + vendored cascade lookup
  capture.py                webcam wrapper + rolling (t, RGB) buffer
  ui.py                     self-drawn OpenCV panel + scrolling waveform
  eulerian.py               stretch: real-time Eulerian magnification view
  data/                     vendored Haar cascade (OpenCV project, Apache-2.0)
tests/
  test_signal_processing.py synthetic, camera-free math correctness tests
  test_integration_vision.py end-to-end tracker+pipeline tests (no camera)
```

The signal-processing core has no camera/OpenCV/MediaPipe dependency, so
the math is testable and portable on its own.

## License

MIT — see [LICENSE](LICENSE). The vendored Haar cascade is from the
OpenCV project (Apache-2.0). The MediaPipe face-landmark model downloaded
at first run is provided by Google under its own terms.
