"""Row-wise spectra of effective kernels and their class-conditional summary (Phase 2, T3).

K[i, :] is the kernel the layer applies at output time i (a "frozen-time" filter over the input
times j). Its magnitude spectrum is |rfft_j K[i, :]|: an FFT of length L = 128, i.e. 65 frequencies from 0 to 25 Hz
in steps of 50/128 = 0.39 Hz (no window). Row i is cut short by the window start (it has i + 1 nonzero entries).
A conditional summary of a time-varying, input-conditioned kernel is built in this order:

    rows i in `rows` (default: all L output times)  ->  mean of the magnitudes (not power)
    ->  windows of one class  ->  channels.

Everything here works on any kernel source `kernel_fn(x) -> [S, layers, C, L, L]`, e.g. `hidden_attention(model)`
below (effective_kernel.py).
"""
import numpy as np
import torch

from effective_kernel import effective_kernels

FS = 50.0                       # UCI HAR sampling rate [Hz]


def freqs(L=128, fs=FS):
    return np.fft.rfftfreq(L, d=1.0 / fs)


def default_rows(L=128):
    return slice(0, L)


def hidden_attention(model, include_skip=True):
    """kernel_fn for the hidden-attention kernel K of effective_kernel.py (D on the diagonal by default)."""
    device = next(model.parameters()).device

    def kernel_fn(x):
        return effective_kernels(model, x.to(device), include_skip=include_skip)[1][:]
    return kernel_fn


def row_spectra(K, rows=slice(None)):
    """|rfft over the input time j| of the rows `rows` of K [..., L, L] -> [..., n_rows, L // 2 + 1]."""
    return torch.fft.rfft(K[..., rows, :], n=K.shape[-1], dim=-1).abs()


def select_windows(y, num_classes, max_per_class=None):
    """Indices of the windows used per class: all of them, or `max_per_class` evenly spread ones."""
    chosen = []
    for c in range(num_classes):
        idx = np.flatnonzero(np.asarray(y) == c)
        if max_per_class is not None and len(idx) > max_per_class:
            idx = idx[np.linspace(0, len(idx) - 1, max_per_class).round().astype(int)]
        chosen.append(idx)
    return chosen


@torch.no_grad()
def class_mean_spectra(kernel_fn, x, y, num_classes=6, rows=None, max_per_class=None, chunk=8):
    """Class-conditional mean row spectra of the kernels given by `kernel_fn` (x: [N, L, 9], y: [N]).

    Returns a dict of numpy arrays (F = L // 2 + 1 frequency bins):
      freq    [F]                      bin frequencies in Hz
      count   [classes]                windows per class
      mean    [classes, layers, C, F]  mean over the rows in `rows` and over the windows of the class
      time    [classes, layers, L, F]  mean over windows and channels, for every row i (time-resolved view)
      window  [N_used, layers, F]      per-window spectrum (mean over rows and channels)
      label   [N_used]                 class of each row of `window`
    """
    chosen = select_windows(y, num_classes, max_per_class)
    order = np.concatenate(chosen)
    labels = np.asarray(y)[order]
    L = x.shape[1]
    rows = default_rows(L) if rows is None else rows
    mean = time = window = None
    for start in range(0, len(order), chunk):
        sel = order[start:start + chunk]
        K = kernel_fn(torch.as_tensor(np.asarray(x)[sel]))                      # [S, layers, C, L, L]
        sp = row_spectra(K.double())                                           # [S, layers, C, L, F]
        in_rows = sp[..., rows, :].mean(3)                                     # [S, layers, C, F]
        if mean is None:
            S, nl, C, _, F = sp.shape
            mean = np.zeros((num_classes, nl, C, F))
            time = np.zeros((num_classes, nl, L, F))
            window = np.zeros((len(order), nl, F))
        for s in range(len(sel)):
            c = labels[start + s]
            mean[c] += in_rows[s].numpy()
            time[c] += sp[s].mean(1).numpy()
            window[start + s] = in_rows[s].mean(1).numpy()
    count = np.array([len(i) for i in chosen])
    return dict(freq=freqs(L), count=count, mean=mean / count[:, None, None, None],
                time=time / count[:, None, None, None], window=window, label=labels)


def channel_mean(spec):
    """[classes, layers, F]: class-conditional spectrum averaged over the channels of each layer."""
    return spec["mean"].mean(2)


def relative_distance(S):
    """[classes, classes] distance between class spectra S [classes, F] of one layer, in percent:
    ||S_a - S_b|| / ||(S_a + S_b) / 2||."""
    d = np.linalg.norm(S[:, None] - S[None], axis=-1)
    return 100 * d / np.linalg.norm((S[:, None] + S[None]) / 2, axis=-1)

