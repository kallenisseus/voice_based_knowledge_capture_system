from pathlib import Path


# Central path constants used across AI modules.
AI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AI_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
# Legacy JSON paths are still referenced by import helpers.
DATABASE_FILE = DATA_DIR / "database.json"
CATEGORIES_FILE = DATA_DIR / "categories.json"
