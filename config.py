from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    GEMINI_API_KEY: str = ""       # PHASE 2: required for interview/OCR
    GROQ_API_KEY: str = ""         # PHASE 2: required for STT
    DATABASE_URL: str = "sqlite:///./meditrace.db"
    MAX_PDF_MB: int = 20
    UPLOAD_DIR: str = "./uploads"


settings = Settings()
