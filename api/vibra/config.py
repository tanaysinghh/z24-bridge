import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(os.environ.get("VIBRA_ARTIFACTS", ROOT / "artifacts"))

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_CHANNELS = 32
MAX_DURATION_S = 600.0
MIN_FS = 50.0
MAX_FS = 2000.0
PLOT_POINTS = 1500

DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://vibradiagnose.vercel.app",
]
ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", ",".join(DEFAULT_ORIGINS)).split(",") if o.strip()]
ORIGIN_REGEX = os.environ.get("ALLOWED_ORIGIN_REGEX", r"https://vibradiagnose(-web)?(-[a-z0-9-]+)?\.vercel\.app")
