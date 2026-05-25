# VibeVoice-Realtime-0.5B adapter for mlx-audio server
# https://huggingface.co/mlx-community/VibeVoice-Realtime-0.5B-fp16
#
# Start server: mlx_audio.server --host 0.0.0.0 --port 8000
# .env: TTS_PROVIDER=remote, TTS_HOST=http://localhost:8000/v1, TTS_MODEL=vibevoice-realtime-0.5b

MODEL_ID = "mlx-community/VibeVoice-Realtime-0.5B-fp16"

DEFAULT_VOICE = "en-Frank_man"

VOICES = {
	# English
	"en-Emma_woman": "English female (Emma)",
	"en-Grace_woman": "English female (Grace)",
	"en-Carter_man": "English male (Carter)",
	"en-Davis_man": "English male (Davis)",
	"en-Frank_man": "English male (Frank)",
	"en-Mike_man": "English male (Mike)",
	# German
	"de-Spk0_man": "German male",
	"de-Spk1_woman": "German female",
	# French
	"fr-Spk0_man": "French male",
	"fr-Spk1_woman": "French female",
	# Spanish
	"sp-Spk0_woman": "Spanish female",
	"sp-Spk1_man": "Spanish male",
	# Portuguese
	"pt-Spk0_woman": "Portuguese female",
	"pt-Spk1_man": "Portuguese male",
	# Italian
	"it-Spk0_woman": "Italian female",
	"it-Spk1_man": "Italian male",
	# Dutch
	"nl-Spk0_man": "Dutch male",
	"nl-Spk1_woman": "Dutch female",
	# Polish
	"pl-Spk0_man": "Polish male",
	"pl-Spk1_woman": "Polish female",
	# Indian English
	"in-Samuel_man": "Indian English male (Samuel)",
	# Japanese
	"jp-Spk0_man": "Japanese male",
	"jp-Spk1_woman": "Japanese female",
	# Korean
	"kr-Spk0_woman": "Korean female",
	"kr-Spk1_man": "Korean male",
}
