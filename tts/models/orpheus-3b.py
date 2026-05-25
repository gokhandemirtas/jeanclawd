# Orpheus-3B-0.1-FT adapter for mlx-audio server
# https://huggingface.co/mlx-community/orpheus-3b-0.1-ft-bf16
#
# Start server: mlx_audio.server --host 0.0.0.0 --port 8000
# .env: TTS_PROVIDER=remote, TTS_HOST=http://localhost:8000/v1, TTS_MODEL=orpheus-3b

MODEL_ID = "mlx-community/orpheus-3b-0.1-ft-bf16"

SAMPLE_RATE = 24000  # Hz, used for PCM streaming

DEFAULT_VOICE = "tara"

VOICES = {
	"tara": "English female (Tara) — warm, natural",
	"leah": "English female (Leah) — clear, expressive",
	"jess": "English female (Jess) — conversational",
	"mia":  "English female (Mia) — soft, friendly",
	"zoe":  "English female (Zoe) — bright, upbeat",
	"leo":  "English male (Leo) — neutral, clear",
	"dan":  "English male (Dan) — deep, authoritative",
	"zac":  "English male (Zac) — casual, relaxed",
}
