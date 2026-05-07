from pathlib import Path
import json

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding='utf-8') as file:
        json.dump(data, file, incident=4)