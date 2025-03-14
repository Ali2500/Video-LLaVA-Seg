# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

# import einops
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MultiheadAttention(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_heads, bias=False, dropout=0.0):
#         super().__init__()

#         self.proj_k = nn.Linear(input_dim, hidden_dim, bias=bias)
#         self.proj_q = nn.Linear(input_dim, hidden_dim, bias=bias)
#         self.proj_v = nn.Linear(input_dim, hidden_dim, bias=bias)
#         self.proj_out = nn.Linear(input_dim, hidden_dim, bias=bias)

#         self.dropout = dropout
#         self.num_heads = num_heads
#         assert hidden_dim % num_heads == 0, f"Hidden dim {hidden_dim} should be exactly divisible by number of heads {num_heads}."

#     def forward(self, q, k, v):
#         # q,k,v : [B, N, C]
#         q = self.proj_q(q)
#         k = self.proj_k(k)
#         v = self.proj_v(v)
#         dropout = self.dropout if self.training else 0.0

#         if q.shape == k.shape: # use more efficient packed function
#             qkv = torch.stack((q, k, v), dim=2)  # [B, N, 3, C]
#             qkv = einops.rearrange(qkv, "B N three (Nh Ch) -> B N three Nh Ch", Nh=self.num_heads)

#             out = flash_attn_qkvpacked_func(
#                 qkv=qkv,
#                 dropout_p=dropout
#             )
#         else:
#             q = einops.rearrange(q, "B N (Nh Ch) -> B N Nh Ch", Nh=self.num_heads)
#             k = einops.rearrange(k, "B N (Nh Ch) -> B N Nh Ch", Nh=self.num_heads)
#             v = einops.rearrange(v, "B N (Nh Ch) -> B N Nh Ch", Nh=self.num_heads)

#             out = flash_attn_func(
#                 q=q,
#                 k=k,
#                 v=v,
#                 dropout_p=dropout
#             )

#         out = einops.rearrange(out, "B N Nh Ch -> B N (Nh Ch)")
#         return self.proj_out(out)


# def _test():
#     x = torch.randn((1, 1234, 256)).cuda().to(torch.bfloat16)
#     attn = MultiheadAttention(256, 256, 8).cuda().to(torch.bfloat16)

#     y = attn(x, x, x)
#     print(x.shape, y.shape)


# if __name__ == '__main__':
#     _test()
