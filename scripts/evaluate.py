
# scripts/evaluate.py

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import yaml

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from src.data import load_dataset


def load_cfg(path="config/train.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():

    # Charger la configuration
    cfg = load_cfg()

    # Récupérer le dossier des artefacts
    art_dir = Path(cfg.get("artifacts_dir", "artifacts"))

    # Charger le modèle entraîné
    model = joblib.load(art_dir / "model.joblib")

    X, y = load_dataset(cfg)

    _, Xte, _, yte = train_test_split(
        X,
        y,
        test_size=float(cfg["split"]["test_size"]),
        random_state=int(cfg["split"]["random_state"]),
        stratify=y,
    )

    pred = model.predict(Xte)

    # Générer le rapport
    report = classification_report(
    yte,
    pred,
    output_dict=True
    )

    # Enregistrer le rapport
    (art_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("Evaluate OK: artifacts/report.json")


if __name__ == "__main__":
    main()
