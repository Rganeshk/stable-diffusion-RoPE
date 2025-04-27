import torch
import math

# === ALiBi utilities ===

def build_alibi_bias(seq_len, num_heads, device, max_bias=8.0):
    slopes = get_alibi_slopes(num_heads).to(device)
    pos = torch.arange(seq_len, device=device).unsqueeze(0)
    bias = pos.unsqueeze(1) - pos.unsqueeze(0)
    bias = bias.unsqueeze(0) * slopes.unsqueeze(1).unsqueeze(1)
    bias = torch.clamp(bias, max=max_bias)
    return bias

def get_alibi_slopes(n_heads):
    def get_slopes(n):
        start = 2.0 ** -(2 ** -(math.log2(n) - 3))
        ratio = start
        return torch.tensor([start * (ratio ** i) for i in range(n)])

    if math.log2(n_heads).is_integer():
        return get_slopes(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        return torch.cat([get_slopes(closest_power_of_2), get_slopes(2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2]])

class AlibiAttentionWrapper(torch.nn.Module):
    def __init__(self, attn_layer):
        super().__init__()
        self.attn = attn_layer
        self.alibi_bias = None

    def forward(self, x, *args, **kwargs):
        B, S, C = x.shape
        device = x.device
        num_heads = self.attn.num_heads
        head_dim = C // num_heads

        qkv = torch.nn.functional.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias)
        qkv = qkv.view(B, S, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.alibi_bias is None or self.alibi_bias.shape[-1] != S:
            self.alibi_bias = build_alibi_bias(S, num_heads, device)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn_weights += self.alibi_bias.unsqueeze(0)
        attn_weights = attn_weights.softmax(dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, C)
        output = self.attn.out_proj(attn_output)

        return output

def apply_alibi(transformer_model):
    """
    Apply ALiBi wrapper to all MultiheadAttention layers inside the transformer model.
    """
    for name, module in transformer_model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            print(f"[ALiBi] Wrapping attention module: {name}")
            parent_module = transformer_model
            subnames = name.split(".")
            for subname in subnames[:-1]:
                parent_module = getattr(parent_module, subname)
            setattr(parent_module, subnames[-1], AlibiAttentionWrapper(module))
