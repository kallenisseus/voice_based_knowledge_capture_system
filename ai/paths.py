import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DATABASE = os.path.join(DATA_ROOT, "database.json")