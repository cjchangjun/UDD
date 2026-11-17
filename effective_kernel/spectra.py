"""Row-wise spectra of effective kernels and their class-conditional summary (Phase 2, T3).

K[i, :] is the kernel the layer applies at output time i (a "frozen-time" filter over the input
times). Row i is written over the lag j: K[i, i - j], j = 0..LAGS-1 (j = 0 is the diagonal K[i, i], j = 63 is
K[i, i - 63]). Every row has the same length 64, so only the rows i = 63..127 are used (row i < 63 has fewer
than 64 past inputs). Its power spectrum is |rfft_j K[i, i - j]|^2: an FFT of length 64, i.e. 33 frequencies
from 0 to 25 Hz in steps of 50/64 = 0.78 Hz (no window). In the LTI case K[i, i - j] = k[j] (lti_kernel).
A conditional summary of a time-varying, input-conditioned kernel is built in this order:

    rows i = 63..127  ->  mean of the powers |.|^2  ->  windows of one class  ->  channels.

Everything here works on any kernel source `kernel_fn(x) -> [S, layers, C, L, L]`, e.g. `hidden_attention(model)`
below (effective_kernel.py).
"""
import numpy as np
import torch

from effective_kernel import effective_kernels

FS = 50.0                       # UCI HAR sampling rate [Hz]
LAGS = 64                       # lags j = 0..63 per row, also the FFT length


def freqs(n=LAGS, fs=FS):
    return np.fft.rfftfreq(n, d=1.0 / fs)


def hidden_attention(model, include_skip=True):
    """kernel_fn for the hidden-attention kernel K of effective_kernel.py (D on the diagonal by default)."""
    device = next(model.parameters()).device

    def kernel_fn(x):
        return effective_kernels(model, x.to(device), include_skip=include_skip)[1][:]
    return kernel_fn


def lag_rows(K, lags=LAGS):
    """K [..., L, L] -> [..., L - lags + 1, lags]: entry [r, j] = K[i, i - j] with i = lags - 1 + r."""
    L = K.shape[-1]
    i = torch.arange(lags - 1, L, device=K.device)[:, None]
    return K[..., i, i - torch.arange(lags, device=K.device)]


def row_spectra(K, lags=LAGS):
    """|rfft over the lag j|^2 of K[i, i - j], rows i = lags-1..L-1 -> [..., L - lags + 1, lags // 2 + 1]."""
    return torch.fft.rfft(lag_rows(K, lags), n=lags, dim=-1).abs() ** 2


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
def class_mean_spectra(kernel_fn, x, y, num_classes=6, lags=LAGS, max_per_class=None, chunk=8):
    """Class-conditional mean row spectra of the kernels given by `kernel_fn` (x: [N, L, 9], y: [N]).

    Returns a dict of numpy arrays (F = lags // 2 + 1 frequency bins):
      freq    [F]                      bin frequencies in Hz
      count   [classes]                windows per class
      mean    [classes, layers, C, F]  mean over the rows i = lags-1..L-1 and over the windows of the class
      window  [N_used, layers, F]      per-window spectrum (mean over rows and channels)
      label   [N_used]                 class of each row of `window`
    """
    chosen = select_windows(y, num_classes, max_per_class)
    order = np.concatenate(chosen)
    labels = np.asarray(y)[order]
    mean = window = None
    for start in range(0, len(order), chunk):
        sel = order[start:start + chunk]
        K = kernel_fn(torch.as_tensor(np.asarray(x)[sel]))                      # [S, layers, C, L, L]
        in_rows = row_spectra(K.double(), lags).mean(3)                        # [S, layers, C, F]
        if mean is None:
            S, nl, C, F = in_rows.shape
            mean = np.zeros((num_classes, nl, C, F))
            window = np.zeros((len(order), nl, F))
        for s in range(len(sel)):
            c = labels[start + s]
            mean[c] += in_rows[s].cpu().numpy()
            window[start + s] = in_rows[s].mean(1).cpu().numpy()
    count = np.array([len(i) for i in chosen])
    return dict(freq=freqs(lags), count=count, mean=mean / count[:, None, None, None], window=window, label=labels)


def channel_mean(spec):
    """[classes, layers, F]: class-conditional spectrum averaged over the channels of each layer."""
    return spec["mean"].mean(2)


def relative_distance(S):
    """[classes, classes] distance between class spectra S [classes, F] of one layer, in percent:
    ||S_a - S_b|| / ||(S_a + S_b) / 2||."""
    d = np.linalg.norm(S[:, None] - S[None], axis=-1)
    return 100 * d / np.linalg.norm((S[:, None] + S[None]) / 2, axis=-1)

