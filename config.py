import os

try:
    from api_key import OPENAI_API_KEY as API_KEY_FILE, NCBI_API_KEY as NCBI_KEY_FILE
except ImportError:
    API_KEY_FILE = ""
    NCBI_KEY_FILE = ""


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", API_KEY_FILE)
    NCBI_API_KEY: str = os.getenv("NCBI_API_KEY", NCBI_KEY_FILE)
    STORAGE_PATH: str = os.getenv("STORAGE_PATH", "./data")


settings = Settings()
