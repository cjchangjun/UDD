"""Effective SSM kernels of a trained starter MambaClassifier (Phase 2, T3).

For each sample, Mamba layer and SSM channel, the starter SSM (mamba_block.py, ssm()) is a causal
linear map y = K u over the window. K is the hidden-attention matrix of Ali et al. (2024), Eq. 12,
plus the starter's skip term D on the diagonal:

    K[i, j] = sum_n C_i[n] * exp(A[n] * (Delta_{j+1} + ... + Delta_i)) * Delta_j * B_j[n]   (j <= i)
              + D                                                                            (i == j)
    K[i, j] = 0                                                                              (j >  i)

u is the SSM input (after in_proj, conv1d and SiLU), y the SSM output before the SiLU(z) gate.
Rows i are output times, columns j input times. The starter is not modified: Delta, A, B, C come
from its set_record() and D from the model parameters.

Usage:
    logits, K = effective_kernels(model, x)     # x: normalized input [batch, time, 9]
    K.shape                                     # (batch, n_layers, d_inner, time, time)
    K[s, l, c]                                  # [time, time] matrix, computed when indexed
    K[s, l]                                     # [d_inner, time, time]
    K.dense()                                   # the whole array (large; computed sample by sample)
"""
import torch


@torch.no_grad()
def effective_kernels(model, x, include_skip=True, dt_mult=None, mask=None):
    """Run one forward pass of a MambaClassifier and return (logits, EffectiveKernels).

    include_skip=False leaves out D, i.e. the hidden-attention matrix of the paper alone.
    The logits are those of model(x); recording is switched off again afterwards.
    """
    model.set_record(True)
    try:
        logits = model(x, dt_mult=dt_mult, mask=mask)
        records = [dict(r) for r in model.records]
    finally:
        model.set_record(False)
    D = [layer.mixer.D.detach().cpu() for layer in model.layers]
    return logits, EffectiveKernels(records, D, include_skip)


class EffectiveKernels:
    """Lazy array of shape (sample, layer, channel, time, time).

    Only Delta, A, B, C (per layer) and D are stored; K[s, l, c] builds the requested matrices
    when indexed, in float64 on the CPU. Each index may be an int, slice, list or bool mask.
    """

    def __init__(self, records, D, include_skip=True):
        self.records, self.D, self.include_skip = records, D, include_skip
        batch, time, channels = records[0]["delta"].shape
        self.shape = (batch, len(records), channels, time, time)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        index = index if isinstance(index, tuple) else (index,)
        if len(index) > 3:
            raise IndexError("index as K[sample, layer, channel]")
        index += (slice(None),) * (3 - len(index))
        s, l, c = (torch.arange(n)[i] for n, i in zip(self.shape[:3], index))
        out = torch.stack([self._layer(layer, s.reshape(-1), c.reshape(-1))
                           for layer in l.reshape(-1).tolist()], dim=1)
        return out.reshape(*s.shape, *l.shape, *c.shape, *self.shape[3:])  # int indices drop their axis

    def dense(self):
        return torch.stack([self[s] for s in range(len(self))])

    def _layer(self, layer, s, c):
        """[len(s), len(c), time, time] kernels of one layer."""
        r = self.records[layer]
        delta = r["delta"][s][:, :, c].double()                   # [S,T,C]
        A = r["A"][c].double()                                     # [C,N]
        B, C = r["B"][s].double(), r["C"][s].double()              # [S,T,N]
        cum = delta.cumsum(1)
        decay = (cum[:, :, None] - cum[:, None]).clamp(min=0)      # [S,i,j,C]: Delta_{j+1} + ... + Delta_i
        delta_B = delta[..., None] * B[:, :, None]                 # [S,j,C,N]
        K = 0
        for n in range(A.shape[1]):
            K = K + C[:, :, None, None, n] * torch.exp(decay * A[:, n]) * delta_B[:, None, :, :, n]
        time = K.shape[1]
        eye = torch.eye(time, dtype=K.dtype)[None, :, :, None]
        if self.include_skip:
            K = K + eye * self.D[layer][c].double()
        valid = r["mask"][s]
        keep = torch.ones(time, time, dtype=torch.bool).tril() & valid[:, :, None] & valid[:, None, :]
        K = torch.where(keep[..., None], K, torch.zeros_like(K))
        return K.permute(0, 3, 1, 2)                               # [S,C,i,j]
