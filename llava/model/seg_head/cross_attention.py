# import torch.nn as nn

# from torch import FloatTensor
# from typing import Optional

# from llava.model.seg_head.flash_attn_wrapper import MultiheadAttention


# class CrossAttentionLayer(nn.Module):
#     def __init__(self, hidden_dims, num_heads, dropout=0.1):
#         super().__init__()
#         self.attn = MultiheadAttention(
#             input_dim=hidden_dims,
#             hidden_dim=hidden_dims,
#             num_heads=num_heads,
#             dropout=dropout
#         )

#         self.norm = nn.LayerNorm(hidden_dims)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, src: FloatTensor, tgt: FloatTensor, q_embed: Optional[FloatTensor] = None, k_embed: Optional[FloatTensor] = None):
#         q = self.norm(src)
#         q = q + q_embed if q_embed is not None else y

#         k = tgt + k_embed if k_embed is not None else tgt
#         v = tgt

#         y = self.attn(q=q, k=k, v=v)
#         y = src + self.dropout(y)
#         return y
    