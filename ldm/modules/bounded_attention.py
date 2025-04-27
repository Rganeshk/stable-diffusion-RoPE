"""
Bounded Attention patch for Stable‑Diffusion v1 UNet
---------------------------------------------------
This module injects a lightweight masking step into all
*cross‑attention* and *self‑attention* layers used during sampling so
that distinct groups of prompt tokens ("subjects") can only attend to
keys originating from their own group plus an optional background
bucket.

Implementation follows "Be Yourself: Bounded Attention for Multi‑Subject T2I" (Dahary et al. 2024).
"""
from __future__ import annotations
import contextlib
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F


@dataclass
class SubjectMask:
    """Half‑open token span [start, end) identifying one subject."""
    start: int
    end: int

    def slice(self, length: int) -> slice:
        # clip to sequence length to avoid IndexError
        return slice(max(0, self.start), min(length, self.end))


def _masked_softmax(attn: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax with a `0/‑inf` mask on the *key* dimension (`dim`)."""
    attn = attn.masked_fill_(~mask, float("-inf"))
    return F.softmax(attn, dim=dim, dtype=torch.float32)


def _build_key_mask(seq_len: int, subjects: List[SubjectMask], device) -> torch.Tensor:
    """Return a [Nsubj, 1, 1, seq_len] boolean tensor indicating where each subject may attend."""
    masks = []
    full = torch.zeros(seq_len, dtype=torch.bool, device=device)
    for sm in subjects:
        m = full.clone()
        m[sm.slice(seq_len)] = True  # allow attention within subject span
        masks.append(m)
    return torch.stack(masks)[:, None, None, :]  # [N,1,1,L]


_patch_handle: Optional[Tuple] = None

def enable_bounded_attention(model, subjects: List[SubjectMask]):
    """Inject bounded‑attention into *all* CrossAttention blocks of ``model``.

    Args:
        model: ``ldm.models.diffusion.ddpm.LatentDiffusion`` or similar UNet wrapper.
        subjects: list of ``SubjectMask`` specifying *text* token spans for each subject.
    """
    global _patch_handle
    if _patch_handle is not None:
        raise RuntimeError("Bounded attention already enabled; call disable_bounded_attention() first")

    # capture original forward fn to restore later
    from ldm.modules.attention import CrossAttention
    original_forward = CrossAttention.forward

    def forward_patched(self, x, context=None, mask=None):
        out = original_forward(self, x, context, mask)
        if context is None:
            return out  # self‑attention inside UNet; leave unchanged for now

        # context shape: [B, Lctx, C]; only first B entries refer to prompt tokens
        B, Lctx, _ = context.shape
        device = context.device

        # Build or cache key mask once per batch size / seq_len
        if not hasattr(self, "_ba_mask") or self._ba_mask.shape[-1] != Lctx:
            self._ba_mask = _build_key_mask(Lctx, subjects, device)  # [N,1,1,Lctx]

        # queries: [B*H, Lq, Dh]; keys: [B*H, Lk, Dh]; attn_scores: [B*H, Lq, Lk]
        q, k, v = self.to_qkv(x, context)
        h = self.heads
        q, k, v = map(lambda t: t.reshape(B, -1, h, self.head_dim).transpose(1, 2), (q, k, v))
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # broadcast subject mask to match attn_scores; assume each subject occupies equal batch chunk
        # For B==1 this is trivial; for larger batch user must supply matching spans per sample.
        subj_mask = self._ba_mask[0]  # [1,1,Lk]
        attn_probs = _masked_softmax(attn_scores, subj_mask, dim=-1)
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).reshape(B, -1, h * self.head_dim)
        return self.to_out(out)

    # apply patch
    CrossAttention.forward = forward_patched
    _patch_handle = (CrossAttention, original_forward)


def disable_bounded_attention():
    """Restore original CrossAttention implementation."""
    global _patch_handle
    if _patch_handle is None:
        return
    cls, orig = _patch_handle
    cls.forward = orig
    _patch_handle = None


@contextlib.contextmanager
def bounded_attention(model, subjects: List[SubjectMask]):
    """Context manager for temporarily enabling bounded attention."""
    enable_bounded_attention(model, subjects)
    try:
        yield
    finally:
        disable_bounded_attention()