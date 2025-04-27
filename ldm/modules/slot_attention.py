# ldm/modules/slot_attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """
    Locatello et al., 2020 – re-implemented in PyTorch, batched.
    Inputs
    ------
    x           : (B, N, D) — sequence of N tokens (e.g. flattened feature map)
    mask        : (B, N) or None — bool/int mask; 1 = keep token
    Returns
    -------
    slots       : (B, S, D) — learned object-centric slots
    attn_logits : (B, S, N) — before softmax (debugging only)
    """
    def __init__(
        self,
        num_slots: int,
        dim: int,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128,
        slot_init_scale: float = 0.5,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps

        # Slot initialisation parameters (μ, σ)
        self.slot_mu = nn.Parameter(torch.randn(1, 1, dim) * 0.01)
        self.slot_log_sigma = nn.Parameter(
            torch.zeros(1, 1, dim) + math.log(slot_init_scale)
        )

        # Projectors
        self.lin_q = nn.Linear(dim, dim, bias=False)
        self.lin_k = nn.Linear(dim, dim, bias=False)
        self.lin_v = nn.Linear(dim, dim, bias=False)

        # GRU & MLP for slot update
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, dim)
        )

        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, x, mask=None, return_logits=False):
        B, N, D = x.shape
        assert D == self.dim, f"dim mismatch: {D}!={self.dim}"
        if mask is not None:
            x = x * mask.unsqueeze(-1)  # zero out masked tokens

        # Initialise slots
        mu = self.slot_mu.expand(B, self.num_slots, -1)
        sigma = self.slot_log_sigma.exp().expand_as(mu)
        slots = mu + sigma * torch.randn_like(mu)

        # Pre-norm
        x = self.norm_inputs(x)

        for _ in range(self.iters):
            slots_prev = slots

            ## Attention -----------------------------------------------------
            # (B, S, D)·Wq  -> (B, S, D)
            q = self.lin_q(self.norm_slots(slots))
            k = self.lin_k(x)                   # (B, N, D)
            v = self.lin_v(x)                   # (B, N, D)

            attn_logits = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(D)  # (B,S,N)
            if mask is not None:
                attn_logits = attn_logits.masked_fill(~mask[:, None, :], -1e9)

            attn = attn_logits.softmax(dim=-1) + self.eps  # (B, S, N)
            attn = attn / attn.sum(dim=-1, keepdim=True)   # normalise again

            updates = torch.einsum("bjn,bnd->bjd", attn, v)  # (B,S,D)

            ## Slot-wise GRU --------------------------------------------------
            slots = self.gru(
                updates.reshape(-1, D), slots_prev.reshape(-1, D)
            ).view(B, self.num_slots, D)

            ## MLP -----------------------------------------------------------
            slots = slots + self.mlp(self.norm_mlp(slots))

        return (slots, attn_logits) if return_logits else slots
