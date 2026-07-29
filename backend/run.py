"""Backend API entry point."""
from pathlib import Path
import sys

import uvicorn

HARNESS_PATH = Path(__file__).resolve().parent / "packages" / "harness"
if str(HARNESS_PATH) not in sys.path:
    sys.path.insert(0, str(HARNESS_PATH))

from app.core.config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
