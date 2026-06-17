import json
import logging

from google import genai
from google.genai import types

from config import settings

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"


class GeminiParseError(Exception):
    pass


class GeminiClient:
    def __init__(self):
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            if not settings.GEMINI_API_KEY:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Add it to your .env file."
                )
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return self._client

    def _call(self, system_prompt: str, content: str, *, json_mode: bool = False) -> str:
        client = self._get_client()
        config_kwargs: dict = {"system_instruction": system_prompt}
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=content,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text or ""

    def json_completion(self, system_prompt: str, user_payload: dict | str) -> dict:
        content = (
            json.dumps(user_payload) if isinstance(user_payload, dict) else user_payload
        )
        try:
            raw = self._call(system_prompt, content, json_mode=True)
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON; attempting repair")
            try:
                repaired = self._call(
                    "You are a JSON repair assistant. "
                    "Return ONLY valid JSON — no markdown fences, no explanation.",
                    f"The following is invalid JSON. Fix it:\n{raw}",
                    json_mode=True,
                )
                return json.loads(repaired)
            except json.JSONDecodeError as exc:
                raise GeminiParseError(
                    f"LLM output could not be parsed as JSON after repair: {exc}"
                ) from exc

    def text_completion(self, system_prompt: str, user_payload: str) -> str:
        # PHASE 3+: used for OCR prompts and fusion
        try:
            return self._call(system_prompt, user_payload, json_mode=False)
        except Exception as exc:
            raise RuntimeError(f"Gemini text completion failed: {exc}") from exc


gemini_client = GeminiClient()
