import logging

from groq import Groq

from config import settings

logger = logging.getLogger(__name__)

_MIME_TO_EXT: dict[str, str] = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-wav": ".wav",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/aac": ".aac",
    "audio/flac": ".flac",
}


class WhisperError(Exception):
    pass


class WhisperClient:
    def __init__(self) -> None:
        self._client: Groq | None = None

    def _get_client(self) -> Groq:
        if self._client is None:
            if not settings.GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY is not set in .env")
            self._client = Groq(api_key=settings.GROQ_API_KEY)
        return self._client

    def transcribe(self, audio_bytes: bytes, mime_type: str) -> str:
        try:
            client = self._get_client()
            base_mime = mime_type.split(";")[0].strip().lower()
            ext = _MIME_TO_EXT.get(base_mime, ".webm")
            filename = f"audio{ext}"

            result = client.audio.transcriptions.create(
                file=(filename, audio_bytes, base_mime),
                model="whisper-large-v3",
                response_format="json",
            )
            text = (result.text or "").strip()
            if not text:
                raise WhisperError("Whisper returned an empty transcript")
            return text
        except WhisperError:
            raise
        except Exception as exc:
            logger.error("Whisper transcription failed: %s", exc)
            raise WhisperError(f"Transcription failed: {exc}") from exc


whisper_client = WhisperClient()
