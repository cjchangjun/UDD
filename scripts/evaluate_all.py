"""Re-evaluate the saved Phase 1 models on the validation split and print the Phase 1 results table.

Usage (from the repository root):
    python scripts/evaluate_all.py                 # evaluates checkpoints/
    python scripts/evaluate_all.py --root runs     # evaluates models retrained with scripts/train_all.sh

Each per-seed result is compared with checkpoints/<model>/summary.json (the numbers reported in Phase 1).
The official test split is never read.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mamba_starter"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from data import load_har, subject_split  # noqa: E402
from evaluate import load_checkpoint, evaluation_data  # noqa: E402
from signal_utils import ema_features  # noqa: E402
from train import make_loader, evaluate_model, metrics  # noqa: E402

DATA = ROOT / "data" / "har"
SPLIT = json.loads((ROOT / "splits" / "har_split.json").read_text())
MODELS = ["ema_logreg", "cnn", "cnn_width59", "mamba_selective", "mamba_lti"]
SEEDS = [0, 1, 2]


def find(root, model, seed, name):
    hits = sorted((root / model).glob(f"**/seed_{seed}{name}"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one {model} seed {seed} file under {root / model}, found {len(hits)}")
    return hits[0]


def eval_torch(path, device):
    model, ckpt = load_checkpoint(path, device)
    if ckpt["split"]["val_indices"] != SPLIT["val_indices"]:
        raise ValueError(f"{path}: validation split differs from splits/har_split.json")
    x, y = evaluation_data(ckpt, data_root=str(DATA), split="validation")
    loader = make_loader(x, y, ckpt["config"], validation=True)
    r = evaluate_model(model, loader, device, ckpt["model_kwargs"]["num_classes"])
    return r, sum(p.numel() for p in model.parameters())


def eval_ema(path):
    obj = joblib.load(path)
    x, y, subjects = load_har(DATA)
    _, va = subject_split(subjects, 0.2, 42, None)
    if va.tolist() != SPLIT["val_indices"]:
        raise ValueError("validation split differs from splits/har_split.json")
    return metrics(y[va], obj["pipeline"].predict(ema_features(x[va], obj["alphas"])), 6), None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(ROOT / "checkpoints"), help="folder with <model>/.../seed_<s>/best.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    root = Path(args.root)
    ok = True
    print(f"{'model':16s} {'params':>7s}  {'accuracy':>17s}  {'macro-F1':>17s}  match")
    for m in MODELS:
        if not (root / m).exists():
            print(f"{m:16s} (not found under {root}, skipped)")
            continue
        reported = {r["seed"]: r for r in json.loads((ROOT / "checkpoints" / m / "summary.json").read_text())["runs"]}
        acc, f1, match, params = [], [], True, None
        for s in SEEDS:
            if m == "ema_logreg":
                r, params = eval_ema(find(root, m, s, ".joblib"))
            else:
                r, params = eval_torch(find(root, m, s, "/best.pt"), args.device)
            acc.append(r["accuracy"])
            f1.append(r["macro_f1"])
            match &= (r["confusion_matrix"] == reported[s]["confusion_matrix"])
        ok &= match
        fmt = lambda v: f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}"
        print(f"{m:16s} {params if params else '-':>7}  {fmt(acc):>17s}  {fmt(f1):>17s}  {'OK' if match else 'DIFFERENT'}")
    print("\nall per-seed confusion matrices match the reported results" if ok
          else "\nWARNING: some results differ from checkpoints/<model>/summary.json")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
