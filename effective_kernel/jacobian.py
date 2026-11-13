"""Jacobian-based kernels of a trained starter MambaClassifier (Phase 2, T3 extension).

For one input, a block (or a sublayer) f maps x [L, C] to f(x) [L, C']. Its Jacobian
J[i, o, j, c] = d f(x)_i[o] / d x_j[c] is the exact linearization of f at that input:
f(x + dx) ~ f(x) + J dx. Unlike the hidden-attention K, which freezes Delta, B, C at their values
for this input, J also differentiates through them; and it covers the whole block (in_proj, causal
conv1d, SiLU, SSM, gate, out_proj, norm and residual).

`jacobian_kernels(model, x)` keeps the channel-diagonal part J[i, c, j, c] and returns it in the layout of
effective_kernel.EffectiveKernels, [S, layers, C, L, L] (row i = output time, column j = input time); the
cross-channel entries are dropped. target="block": one residual block of the model
(input = block input, C = d_model); target="ssm": the SSM sublayer only (input = u, C = d_inner), where for
an LTI model J equals K exactly.
"""
import torch
from torch.func import jvp


def full_jacobian(fn, x, chunk=2048):
    """J [L, C', L, C] of fn at x [L, C]; fn maps a batch [b, L, C] to [b, L, C'] with independent batch items.

    Forward mode: batch item k carries the unit tangent of input entry (j, c), and its output tangent is
    the column J[:, :, j, c]. One pass handles `chunk` input entries.
    """
    L, C = x.shape
    n = L * C
    cols = []
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        tangent = torch.zeros(m, n, dtype=x.dtype, device=x.device)
        tangent[torch.arange(m), torch.arange(start, start + m)] = 1
        _, out = jvp(fn, (x.expand(m, L, C).clone(),), (tangent.view(m, L, C),))      # [m, L, C']
        cols.append(out)
    cols = torch.cat(cols).view(L, C, L, -1)                                   # [j, c, i, o]
    return cols.permute(2, 3, 0, 1)                                            # [i, o, j, c]


def block_inputs(model, x):
    """Inputs of every residual block during model(x): list of [S, L, d_model]."""
    got, hooks = {}, []
    for l, layer in enumerate(model.layers):
        hooks.append(layer.register_forward_pre_hook(lambda m, args, l=l: got.__setitem__(l, args[0].detach())))
    try:
        model(x)
    finally:
        for h in hooks:
            h.remove()
    return [got[l] for l in range(len(model.layers))]


def ssm_inputs(model, x):
    """Inputs u of every layer's SSM sublayer during model(x): list of [S, L, d_inner]."""
    got = {}
    for l, layer in enumerate(model.layers):
        def ssm(u, _f=layer.mixer.ssm, _l=l, **kwargs):
            got[_l] = u.detach()
            return _f(u, **kwargs)
        layer.mixer.ssm = ssm                                                  # instance attribute shadows the method
    try:
        model(x)
    finally:
        for layer in model.layers:
            del layer.mixer.ssm
    return [got[l] for l in range(len(model.layers))]


def layer_function(model, layer, target):
    """The map whose Jacobian is taken: the residual block (starter forward) or its SSM sublayer."""
    block = model.layers[layer]
    if target == "block":
        return lambda z: block(z, mask=torch.ones(z.shape[:2], dtype=torch.bool, device=z.device))
    if target == "ssm":
        return lambda z: block.mixer.ssm(z)
    raise ValueError("target must be 'block' or 'ssm'")


@torch.no_grad()
def jacobian_kernels(model, x, target="block", chunk=2048):
    """Channel-diagonal Jacobians of every layer for the windows x [S, L, 9]: [S, layers, C, L, L] (CPU float64)."""
    model.eval()
    params = list(model.parameters())
    x = x.to(params[0].device, params[0].dtype)
    flags = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)                                                # forward mode only: no backward graph
    try:
        inputs = (block_inputs if target == "block" else ssm_inputs)(model, x)
        out = []
        for s in range(len(x)):
            per_layer = []
            for l in range(len(model.layers)):
                J = full_jacobian(layer_function(model, l, target), inputs[l][s], chunk)          # [i, o, j, c]
                per_layer.append(torch.diagonal(J, dim1=1, dim2=3).permute(2, 0, 1).double().cpu())  # [c, i, j]
            out.append(torch.stack(per_layer))
    finally:
        for p, f in zip(params, flags):
            p.requires_grad_(f)
    return torch.stack(out)

