from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_raw: Path = project_root / "data" / "raw"
    data_processed: Path = project_root / "data" / "processed"

PATHS = Paths()
