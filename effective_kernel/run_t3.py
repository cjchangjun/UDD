"""T3 analysis: class-conditional kernel spectra of the trained Mamba models on the validation windows.

Usage (from the repository root, after downloading data/har):
    python effective_kernel/run_t3.py                      # seeds 0 1 2, all validation windows (about 8 min per seed)
    python effective_kernel/run_t3.py --seeds 0            # one seed; run the seeds in parallel processes if you like
    python effective_kernel/run_t3.py --plots_only         # figures and summary from the saved results/t3/*.npz

For every seed it computes (spectra.py, effective_kernel.py):
  selective_D                   hidden-attention kernel K of mamba_selective
  lti_D                         K of mamba_lti (independent of the input: one curve), drawn next to the selective classes
Only validation windows are used; the official test set is never read.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "mamba_starter"), str(ROOT / "effective_kernel")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
from evaluate import load_checkpoint, evaluation_data  # noqa: E402
import spectra  # noqa: E402
import plots  # noqa: E402

DATA = ROOT / "data" / "har"


def load(name, seed, device):
    model, ckpt = load_checkpoint(ROOT / "checkpoints" / name / f"seed_{seed}" / "best.pt", device)
    x, y = evaluation_data(ckpt, data_root=str(DATA))                           # validation windows only
    return model, x, np.asarray(y)


def save(out, name, seed, spec):
    np.savez_compressed(out / f"{name}_seed{seed}.npz", **spec)


def compute_seed(seed, args, out):
    t0 = time.time()
    sel, x, y = load("mamba_selective", seed, args.device)
    lti, _, _ = load("mamba_lti", seed, args.device)
    say = lambda msg: print(f"[seed {seed}] {msg} ({time.time() - t0:.0f}s)", flush=True)

    save(out, "selective_D", seed, spectra.class_mean_spectra(spectra.hidden_attention(sel), x, y, max_per_class=args.max_per_class))
    say("selective")
    save(out, "lti_D", seed, spectra.class_mean_spectra(spectra.hidden_attention(lti), x, y, max_per_class=args.lti_per_class))
    say("LTI")


def load_specs(out, name, seeds):
    files = [out / f"{name}_seed{s}.npz" for s in seeds]
    return [dict(np.load(f)) for f in files if f.exists()]


def make_figures(args, out):
    seeds = args.seeds
    sel, lti = (load_specs(out, n, seeds) for n in ["selective_D", "lti_D"])
    plots.fig_class_spectra(sel, lti).savefig(out / "t3_class_spectra.png", dpi=150)
    summary = {"seeds": [s for s in seeds if (out / f"selective_D_seed{s}.npz").exists()],
               "rows_averaged": "all output times i = 0..127", "spectrum": "magnitude |rfft_j K[i, :]|, FFT length 128 -> 65 frequencies (0-25 Hz), no window",
               "windows_per_class": sel[0]["count"].tolist(),
               "class_distance_percent_selective": [np.mean([spectra.relative_distance(spectra.channel_mean(s)[:, l]) for s in sel], 0).round(2).tolist() for l in range(2)]}
    (out / "t3_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max_per_class", type=int, help="validation windows per class (default: all)")
    p.add_argument("--lti_per_class", type=int, default=5, help="windows per class for the LTI curve (K does not depend on the input)")
    p.add_argument("--plots_only", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--out", default=str(ROOT / "results" / "t3"))
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.plots_only:
        for seed in args.seeds:
            compute_seed(seed, args, out)
    make_figures(args, out)


if __name__ == "__main__":
    main()
