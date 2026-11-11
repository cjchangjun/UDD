"""Phase 1 frequency-response analyses (Task 1.3 EMA filters, Task 1.5 finite-length LTI kernels).

Usage (from the repository root):  python scripts/frequency_response.py
Output: results/ema_filters.json, results/ema_responses.png, results/lti_kernel_responses.png

* EMA  h_t = a h_{t-1} + (1-a) x_t : impulse response (1-a) a^k, frequency response (1-a) / (1 - a e^{-jw}),
  half-life ln2 / (-ln a), half-power cutoff, and Delta for A = -1 (a = exp(Delta A)).
* LTI Mamba (checkpoints/mamba_lti/seed_0, layer 0): kernel K[k] = sum_n C_n exp(k Delta A_n) Delta B_n
  (same formula as lti_kernel, skip D excluded). The exact response of the infinite kernel is compared
  with the FFT of the kernel truncated to L = 16, 32, 128 taps, for the channels with the shortest,
  median and longest memory.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mamba_starter"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from evaluate import load_checkpoint  # noqa: E402

FS = 50.0                                   # UCI HAR sampling rate [Hz]
ALPHAS = [0.5, 0.8, 0.9, 0.95, 0.99]        # EMA coefficients used in Phase 1
OUT = ROOT / "results"
COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


# ---------------------------------------------------------------- Task 1.3: EMA filters
def ema_cutoff_hz(a):
    """Half-power cutoff: |H|^2 = 1/2 at cos(w) = 1 - (1-a)^2 / (2a)."""
    c = 1 - (1 - a) ** 2 / (2 * a)
    return float(np.arccos(c) / (2 * np.pi) * FS) if (1 - a) / (1 + a) < np.sqrt(0.5) else None


def ema_analysis():
    rows = [{"a": a, "half_life_steps": float(np.log(2) / -np.log(a)),
             "half_life_ms": float(np.log(2) / -np.log(a) / FS * 1000),
             "cutoff_hz": ema_cutoff_hz(a), "delta_A": float(np.log(a)), "delta_for_A_minus1": float(-np.log(a))}
            for a in ALPHAS]
    (OUT / "ema_filters.json").write_text(json.dumps(rows, indent=2) + "\n")
    for r in rows:
        print(f"a={r['a']:<5} half-life {r['half_life_steps']:6.1f} steps ({r['half_life_ms']:6.0f} ms)  "
              f"cutoff {r['cutoff_hz']:.2f} Hz  Delta(A=-1) {r['delta_for_A_minus1']:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    k = np.arange(0, 129)
    f = np.linspace(0, FS / 2, 801)
    w = 2 * np.pi * f / FS
    for a, c in zip(ALPHAS, COL):
        ax[0].semilogy(k / FS, (1 - a) * a ** k, color=c, label=f"a={a}")
        ax[1].semilogy(f, np.abs((1 - a) / (1 - a * np.exp(-1j * w))) ** 2, color=c)
        fc = ema_cutoff_hz(a)
        if fc is not None:
            ax[1].plot([fc], [0.5], "o", color=c, ms=4)
    ax[0].set(ylim=(1e-4, 0.6), xlabel="lag [s] (128 samples = 2.56 s)", ylabel="impulse response",
              title="(a) impulse response")
    ax[0].legend(ncol=2)
    ax[1].axhline(0.5, color="gray", ls=":", lw=0.8)
    ax[1].set(xlim=(0, 25), ylim=(2e-5, 1.6), xlabel="frequency [Hz]", ylabel="passed power ratio",
              title="(b) frequency response (dots: half-power cutoff)")
    for a_ in ax:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ema_responses.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- Task 1.5: finite-length LTI kernels
def lti_params(block):
    with torch.no_grad():
        dt = (torch.nn.functional.softplus(block.dt_bias) * block.delta_scale).double().numpy()   # [h]
        A = block.A.double().numpy()                                                             # [h, n]
        B = block.B_const.double().numpy()                                                       # [n]
        C = block.C_const.double().numpy()                                                       # [n]
    return dt, A, B, C


def kernel(dt, A, B, C, L):
    """[L, h] kernel K[k] = sum_n C_n exp(k dt A_n) dt B_n (lti_kernel without D)."""
    k = np.arange(L)[:, None, None]
    return (np.exp(k * dt[None, :, None] * A[None]) * dt[None, :, None] * B * C).sum(-1)


def exact_response(dt, A, B, C, f):
    """[h, f] response of the untruncated kernel: sum_n C_n dt B_n / (1 - exp(dt A_n) e^{-jw})."""
    w = 2 * np.pi * f / FS
    a = np.exp(dt[:, None] * A)
    gain = C[None, :] * dt[:, None] * B[None, :]
    return (gain[:, :, None] / (1 - a[:, :, None] * np.exp(-1j * w)[None, None, :])).sum(1)


def lti_kernel_analysis():
    model, _ = load_checkpoint(ROOT / "checkpoints" / "mamba_lti" / "seed_0" / "best.pt")
    block = model.layers[0].mixer
    dt, A, B, C = lti_params(block)
    # sanity check: same kernel as the starter's lti_kernel (D excluded)
    ref = block.lti_kernel(128, include_skip=False).detach().double().numpy()
    assert np.allclose(kernel(dt, A, B, C, 128), ref, rtol=1e-5, atol=1e-7), "kernel differs from lti_kernel"

    slowest_half_life = (np.log(2) / (-dt[:, None] * A)).max(1)          # per channel, in steps
    order = np.argsort(slowest_half_life)
    picks = [("short", order[0]), ("median", order[len(order) // 2]), ("long", order[-1])]
    f = np.linspace(0, FS / 2, 513)
    nfft = 4096
    fk = np.fft.rfftfreq(nfft, 1 / FS)
    H = exact_response(dt, A, B, C, f)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
    for ax, (lab, h) in zip(axes, picks):
        ax.semilogy(f, np.abs(H[h]) ** 2, color="k", lw=1.8, label="exact (no truncation)")
        for L, c in [(16, "#d62728"), (32, "#ff7f0e"), (128, "#2ca02c")]:
            Hk = np.fft.rfft(kernel(dt, A, B, C, L)[:, h], n=nfft)      # zero-padded FFT of the cut kernel
            ax.semilogy(fk, np.abs(Hk) ** 2 + 1e-20, color=c, lw=0.9, label=f"FFT, L={L}")
        ax.set(xlim=(0, 25), xlabel="frequency [Hz]",
               title=f"{lab} memory (channel {h}, {slowest_half_life[h]:.0f}-step half-life)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("passed power ratio")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "lti_kernel_responses.png", dpi=160)
    plt.close(fig)
    print("LTI kernel channels (layer 0, seed 0):", {lab: int(h) for lab, h in picks})


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    ema_analysis()
    lti_kernel_analysis()
