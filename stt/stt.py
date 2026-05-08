import whisper

_model = None


def _get_model() -> whisper.Whisper:
    global _model
    if _model is None:
        _model = whisper.load_model("base")
    return _model


def transcribe(audio_path: str) -> str:
    """Transcribe an audio file to text using Whisper base model.

    Args:
        audio_path: Path to the audio file (any format Whisper supports)

    Returns:
        Transcribed text string
    """
    model = _get_model()
    result = model.transcribe(audio_path)
    return result["text"].strip()
