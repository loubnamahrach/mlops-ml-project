
import json
import sys
from pathlib import Path

# Rendre le package src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import yaml

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_dataset
from src.features import build_numeric_preprocess
from src.model import build_model


def load_cfg(path="config/train.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center"
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():

    # 1. Charger la configuration
    cfg = load_cfg()

    # 2. Créer le dossier des artefacts
    art_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    art_dir.mkdir(exist_ok=True)

    # 3. Charger les données
    X, y = load_dataset(cfg)

    # 4. Diviser les données
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=float(cfg["split"]["test_size"]),
        random_state=int(cfg["split"]["random_state"]),
        stratify=y,
    )

    # 5. Construire le pipeline
    pipe = Pipeline(steps=[
        ("preprocess", build_numeric_preprocess()),
        ("model", build_model(cfg)),
    ])

    # 6. Entraîner le modèle
    pipe.fit(Xtr, ytr)

    # 7. Effectuer les prédictions
    pred = pipe.predict(Xte)

    # 8. Calculer les métriques
    acc = float(accuracy_score(yte, pred))
    f1 = float(f1_score(yte, pred, average="macro"))

    # 9. Sauvegarder le modèle
    joblib.dump(pipe, art_dir / "model.joblib")

    # 10. Sauvegarder les métriques
    metrics = {
        "accuracy": acc,
        "f1_macro": f1
    }

    (art_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    # 11. Sauvegarder la matrice de confusion
    save_confusion_matrix(
        yte,
        pred,
        art_dir / "confusion_matrix.png"
    )

    print("Train OK:", metrics)


if __name__ == "__main__":
    main()
