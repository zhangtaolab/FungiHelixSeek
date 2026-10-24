#!/usr/bin/env python3
"""Apply the macOS/MPS workarounds to a downloaded HelixSeek model.

The shipped ``attention.py`` has two issues that only surface on Apple Silicon:

1. On torch <= 2.11 the MPS backend returns a query-shaped output from
   ``scaled_dot_product_attention`` when ``value.head_dim < query.head_dim``.
   HelixSeek MLA layers hit this with ``mla_use_nope=True`` (q/k head_dim 96,
   v head_dim 64), so the forward pass dies inside ``o_proj``.
2. Without Triton (macOS has no Triton wheels) ``FLA_AVAILABLE`` is ``False``
   and the KDA layers take the pure-PyTorch fallback, whose gate and delta-rule
   recurrence do not match FLA semantics. It runs, but mask recovery drops to
   chance level. This one is opt-in via ``--kda_fallback``.

Both patches are source-level, idempotent, and keep a pristine
``attention.py.orig`` next to the patched file. The MPS patch is gated on
``device.type == "mps"``, so CUDA/NPU behaviour is unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER_MPS = "HELIXSEEK_MPS_SDPA_PAD"
MARKER_KDA = "HELIXSEEK_KDA_FALLBACK"

SDPA_ORIGINAL = """        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
"""

SDPA_PATCHED = """        # HELIXSEEK_MPS_SDPA_PAD: torch <= 2.11 MPS SDPA returns a
        # query-shaped output when value head_dim != query head_dim.
        value_dim = value.shape[-1]
        pad = query.shape[-1] - value_dim
        pad_value = pad > 0 and query.device.type == "mps"
        if pad_value:
            value = F.pad(value, (0, pad))

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
            is_causal=False,
        )
        if pad_value:
            attn_output = attn_output[..., :value_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
"""

GATE_ORIGINAL = """        else:
            A_decay = torch.exp(-torch.exp(self.A_log.float()))
            dt = F.softplus(self.dt_bias.view(self.num_k_heads, self.head_k_dim))
            g = g * dt * A_decay"""

GATE_PATCHED = """        else:
            # HELIXSEEK_KDA_FALLBACK: match fla.ops.kda.gate.naive_kda_gate
            dt = F.softplus(
                g.float() + self.dt_bias.view(self.num_k_heads, self.head_k_dim)
            )
            g = -torch.exp(self.A_log.view(self.num_k_heads, 1).float()) * dt"""

KDA_START = "            # Pure PyTorch fallback matching FLA KDA semantics"
KDA_END = "\n\n        if past_key_values is not None:"

KDA_PATCHED = """            # HELIXSEEK_KDA_FALLBACK: exact KDA recurrence, mirrors
            # fla.ops.kda.naive.naive_recurrent_kda
            q = rearrange(q, "b h d s -> b s h d")
            k = rearrange(k, "b h d s -> b s h d")
            v = rearrange(v, "b h d s -> b s h d")

            q_norm = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
            k_norm = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            q_norm = q_norm * (q.shape[-2] ** -0.5)

            b_size, seq_len, n_heads, head_dim = q_norm.shape
            value_dim = v.shape[-1]

            state = torch.zeros(
                b_size, n_heads, head_dim, value_dim,
                dtype=q_norm.dtype, device=q_norm.device,
            )
            if recurrent_state is not None:
                state = state + recurrent_state

            outs = []
            for i in range(seq_len):
                k_i = k_norm[:, i]
                v_i = v[:, i]
                b_i = beta[:, i].unsqueeze(-1)
                state = state * g[:, i].exp().unsqueeze(-1)
                state = state + torch.einsum(
                    "bhk,bhv->bhkv", b_i * k_i, v_i - (k_i.unsqueeze(-1) * state).sum(-2)
                )
                outs.append(torch.einsum("bhk,bhkv->bhv", q_norm[:, i], state))

            o = torch.stack(outs, dim=1)
            recurrent_state = state"""


class PatchError(RuntimeError):
    pass


def patch_sdpa(source: str) -> tuple[str, bool]:
    if MARKER_MPS in source:
        return source, False
    if source.count(SDPA_ORIGINAL) != 1:
        raise PatchError(
            "expected exactly one scaled_dot_product_attention call site, "
            f"found {source.count(SDPA_ORIGINAL)}"
        )
    return source.replace(SDPA_ORIGINAL, SDPA_PATCHED), True


def patch_kda_fallback(source: str) -> tuple[str, bool]:
    if MARKER_KDA in source:
        return source, False
    if source.count(GATE_ORIGINAL) != 1:
        raise PatchError(
            f"expected exactly one KDA gate block, found {source.count(GATE_ORIGINAL)}"
        )
    if source.count(KDA_START) != 1:
        raise PatchError(
            f"expected exactly one KDA fallback block, found {source.count(KDA_START)}"
        )
    patched = source.replace(GATE_ORIGINAL, GATE_PATCHED)
    start = patched.index(KDA_START)
    end = patched.find(KDA_END, start)
    if end == -1:
        raise PatchError("could not find the end of the KDA fallback block")
    return patched[:start] + KDA_PATCHED + patched[end:], True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", "-m", default="models/FungiHelixSeek")
    parser.add_argument(
        "--kda_fallback",
        action="store_true",
        help="also fix the pure-PyTorch KDA fallback used when Triton is absent",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    target = Path(args.model_path) / "attention.py"
    if not target.is_file():
        print(f"error: {target} not found", file=sys.stderr)
        return 1

    source = target.read_text()
    try:
        patched, did_sdpa = patch_sdpa(source)
        did_kda = False
        if args.kda_fallback:
            patched, did_kda = patch_kda_fallback(patched)
    except PatchError as exc:
        print(f"error: {exc}; the model code differs from the targeted version", file=sys.stderr)
        return 1

    if not (did_sdpa or did_kda):
        print(f"already patched: {target}")
        return 0

    compile(patched, str(target), "exec")

    if args.dry_run:
        print(f"dry run: {target} would be patched (sdpa={did_sdpa}, kda_fallback={did_kda})")
        return 0

    backup = target.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(source)
    target.write_text(patched)
    print(f"patched: {target} (sdpa={did_sdpa}, kda_fallback={did_kda})")
    print(f"backup:  {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
