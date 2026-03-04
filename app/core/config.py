import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Vector Retail Finance Agent"
    VERSION: str = "1.0.0"
    
    # Security
    OPENAI_API_KEY: str = "default_key_override_in_env"
    
    # Model Config
    MODEL_NAME: str = "gpt-4o-2024-05-13"
    MIN_REL_SCORE: float = 0.85
    MAX_RETRIES: int = 3
    FISCAL_YEAR_FILTER: int = 2024

    class Config:
        env_file = ".env"

settings = Settings()
