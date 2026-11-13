"""Checks effective_kernel.py on validation windows of the saved Mamba checkpoints.

Usage (from the repository root, after downloading data/har):
    python effective_kernel/check_effective_kernel.py

For every seed of mamba_selective and mamba_lti (models run in float64, so that the differences
reflect the implementation and not float32 rounding):
  * logits returned by effective_kernels equal model(x);
  * K u reproduces the starter ssm() output for every layer, channel and sample;
  * LTI only: every K[s, l, c] is the Toeplitz matrix of the starter lti_kernel(T), with D
    (include_skip=True) and without D (include_skip=False);
  * K[s, l, c] equals K.dense()[s, l, c] bit for bit.
Jacobian kernels (jacobian.py), one window per seed, in float64 on --device (default: cuda if available):
  * the block Jacobian matches a central finite difference of the block in a random direction;
  * LTI only: the channel-diagonal Jacobian of the SSM sublayer equals K exactly (for selective they differ: J also
    differentiates through Delta, B, C).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mamba_starter"))

import torch  # noqa: E402
from evaluate import load_checkpoint, evaluation_data  # noqa: E402
from effective_kernel import effective_kernels  # noqa: E402
import jacobian  # noqa: E402

DATA = ROOT / "data" / "har"
TOL = 1e-10
FD_TOL = 1e-7                  # central finite difference with step 1e-6


def capture_ssm(model):
    """Record the input u and output y of every layer's starter ssm() during the next forward pass."""
    io = {}
    for l, layer in enumerate(model.layers):
        def ssm(u, _f=layer.mixer.ssm, _l=l, **kwargs):
            y = _f(u, **kwargs)
            io[_l] = (u.detach(), y.detach())
            return y
        layer.mixer.ssm = ssm
    return io


def toeplitz(lti, T):
    """[C, T, T] lower-triangular Toeplitz matrices from a lag-indexed kernel lti [T, C]."""
    i, j = torch.arange(T)[:, None], torch.arange(T)[None, :]
    return torch.where((i >= j)[..., None], lti[(i - j).clamp(min=0)], 0.0).permute(2, 0, 1)


def check(name, seed):
    model, ckpt = load_checkpoint(ROOT / "checkpoints" / name / f"seed_{seed}" / "best.pt")
    model = model.double()
    x, _ = evaluation_data(ckpt, data_root=str(DATA))
    x = torch.as_tensor(x[::62], dtype=torch.float64)          # 32 windows spread over the validation set
    with torch.no_grad():
        ref = model(x)
    io = capture_ssm(model)
    logits, K = effective_kernels(model, x)
    _, K0 = effective_kernels(model, x, include_skip=False)
    T = x.shape[1]
    err = {"logits": (logits - ref).abs().max().item()}
    for l, layer in enumerate(model.layers):
        u, y = io[l]
        Kl = K[:, l]                                               # [S,C,T,T]
        y_hat = torch.einsum("scij,sjc->sic", Kl, u)
        err[f"Ku L{l}"] = ((y_hat - y).abs().max() / y.abs().max()).item()
        if name == "mamba_lti":
            mixer = layer.mixer
            with torch.no_grad():
                err[f"lti L{l}"] = (Kl - toeplitz(mixer.lti_kernel(T), T)).abs().max().item()
                err[f"lti noD L{l}"] = (K0[:, l] - toeplitz(mixer.lti_kernel(T, include_skip=False), T)).abs().max().item()
    _, small = effective_kernels(model, x[:4])                     # 4 windows, to keep dense() small
    dense = small.dense()
    same = all(torch.equal(small[s, l, c], dense[s, l, c]) for s in range(4) for l in range(2) for c in (0, 17, 63))
    ok = same and all(v < TOL for v in err.values())
    print(f"{name:16s} seed {seed}  " + "  ".join(f"{k}={v:.1e}" for k, v in err.items())
          + f"  dense==index={same}  {'OK' if ok else 'FAIL'}")
    return ok


def check_jacobian(name, seed, device):
    model, ckpt = load_checkpoint(ROOT / "checkpoints" / name / f"seed_{seed}" / "best.pt", device)
    model = model.double()
    for p in model.parameters():
        p.requires_grad_(False)
    x, _ = evaluation_data(ckpt, data_root=str(DATA))
    xb = torch.as_tensor(x[:1], dtype=torch.float64, device=device)
    err = {}
    for l, z in enumerate(jacobian.block_inputs(model, xb)):
        fn = jacobian.layer_function(model, l, "block")
        J = jacobian.full_jacobian(fn, z[0])
        d, eps = torch.randn_like(z[0]), 1e-6
        fd = (fn((z[0] + eps * d)[None]) - fn((z[0] - eps * d)[None]))[0] / (2 * eps)
        err[f"FD L{l}"] = ((torch.einsum("iojc,jc->io", J, d) - fd).abs().max() / fd.abs().max()).item()
    ok = all(v < FD_TOL for v in err.values())
    if name == "mamba_lti":
        K = effective_kernels(model, xb)[1][:]
        err["J_ssm-K"] = ((jacobian.jacobian_kernels(model, xb, "ssm").to(device) - K.to(device)).abs().max() / K.abs().max()).item()
        ok = ok and err["J_ssm-K"] < TOL
    print(f"{name:16s} seed {seed}  jacobian  " + "  ".join(f"{k}={v:.1e}" for k, v in err.items()) + f"  {'OK' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device of the Jacobian checks")
    args = p.parse_args()
    results = [check(name, seed) for name in ["mamba_selective", "mamba_lti"] for seed in [0, 1, 2]]
    results += [check_jacobian(name, seed, args.device) for name in ["mamba_selective", "mamba_lti"] for seed in [0, 1, 2]]
    print("all checks passed" if all(results) else "SOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
