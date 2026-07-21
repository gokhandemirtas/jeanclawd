"""Wake-word diagnostic — verify mic capture + openwakeword scoring.

Prints:
  - All sounddevice input devices
  - The default input device
  - Real-time mic RMS (proves the mic is capturing)
  - Real-time openwakeword scores for the configured model
  - Any threshold crossings

Run:
    uv run python -m stt.diagnose

Or to override the model:
    WAKE_WORD_MODEL=hey_jarvis WAKE_WORD_THRESHOLD=0.5 uv run python -m stt.diagnose

Stops after ~12 seconds.
"""

import os
import time

import numpy as np
import sounddevice as sd

_SAMPLE_RATE = 16000
_CHUNK_FRAMES = 1280
_THRESHOLD = float(os.getenv("WAKE_WORD_THRESHOLD", "0.5"))
_MODEL_NAME = os.getenv("WAKE_WORD_MODEL", "hey_jarvis")
_DURATION = float(os.getenv("DIAGNOSE_DURATION", "12"))


def _print_devices() -> None:
	print("=" * 60)
	print("Input devices:")
	try:
		devices = sd.query_devices()
	except Exception as e:
		print(f"  (failed to enumerate: {e})")
		return
	default_in, default_out = sd.default.device
	for i, d in enumerate(devices):
		if d.get("max_input_channels", 0) <= 0:
			continue
		marker = "  → DEFAULT" if i == default_in else ""
		name = d.get("name", "<unnamed>")
		host = sd.query_hostapis(d["hostapi"])["name"]
		print(f"  [{i}] {name}  ({host}, {d['max_input_channels']} ch, "
			  f"{int(d['default_samplerate'])} Hz){marker}")
	print(f"Default input index: {default_in}")
	print("=" * 60)


def _rms(int16_chunk: np.ndarray) -> float:
	# int16 → float32 → RMS in [0, 1]
	samples = int16_chunk.astype(np.float32) / 32768.0
	return float(np.sqrt(np.mean(samples * samples)))


def main() -> None:
	_print_devices()

	print(f"\nLoading openwakeword model: {_MODEL_NAME!r} (threshold={_THRESHOLD})…")
	from openwakeword.model import Model
	model = Model(wakeword_models=[_MODEL_NAME], inference_framework="onnx")

	# Resolve the device the wake-word loop will use (it omits `device=`, so sounddevice picks the default).
	default_in, _ = sd.default.device
	print(f"Opening InputStream on default device [{default_in}] @ {_SAMPLE_RATE} Hz, mono, int16…\n")
	print(f"{'t (s)':>6}  {'rms':>7}  {'score':>6}  bar")
	print("-" * 60)

	peak_rms = 0.0
	peak_score = 0.0
	crossings = 0
	start = time.time()

	with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="int16") as stream:
		while time.time() - start < _DURATION:
			data, _ = stream.read(_CHUNK_FRAMES)

			rms = _rms(data)
			scores = model.predict(data.flatten())
			score = float(scores.get(_MODEL_NAME, 0.0))
			peak_rms = max(peak_rms, rms)
			peak_score = max(peak_score, score)

			t = time.time() - start
			bar = "█" * int(min(rms * 200, 40))
			print(f"{t:6.2f}  {rms:7.4f}  {score:6.3f}  {bar}", end="\r")
			if score >= _THRESHOLD:
				crossings += 1
				print(f"\n  🔔 THRESHOLD CROSSED at t={t:.2f}s (score={score:.3f})")

	print()
	print("-" * 60)
	print(f"Duration:       {_DURATION}s")
	print(f"Peak mic RMS:   {peak_rms:.4f}  (>0.005 = picking up audio; >0.05 = loud)")
	print(f"Peak wakeword:  {peak_score:.3f}   (threshold={_THRESHOLD})")
	print(f"Threshold hits: {crossings}")
	if peak_rms < 0.005:
		print("\n⚠️  Mic RMS near zero — wrong device, muted, or no permission.")
		print("   • macOS: System Settings → Privacy & Security → Microphone → enable for your terminal.")
		print("   • Check the default input device above matches your mic.")
		print("   • Try setting the default in System Settings → Sound → Input.")
	elif peak_score < 0.05:
		print(f"\n⚠️  Mic is capturing but wake-word model never responded (peak {peak_score:.3f}).")
		print("   • Try saying 'hey jarvis' louder / closer to the mic.")
		print(f"   • Lower threshold: WAKE_WORD_THRESHOLD=0.2  (current: {_THRESHOLD})")
		print(f"   • Try a different model: WAKE_WORD_MODEL=alexa  (current: {_MODEL_NAME})")
	elif crossings == 0 and peak_score < _THRESHOLD:
		print(f"\n⚠️  Wake-word picked up *something* (peak {peak_score:.3f}) but never crossed {_THRESHOLD}.")
		print(f"   • Try: WAKE_WORD_THRESHOLD={max(0.2, peak_score - 0.05):.2f}")
		print("   • Or speak louder / with clearer pronunciation.")
	else:
		print("\n✅ Wake-word is working.")


if __name__ == "__main__":
	main()
