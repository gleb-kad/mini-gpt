import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiHeadAttention(nn.Module):

    def __init__(self, n_embd, len_context, n_head):
        super().__init__()

        # B - batch size
        # T - context length
        # C - embedding size

        assert n_embd % n_head == 0
        self.n_heads = n_head
        self.head_size = n_embd // n_head

        self.register_buffer("tril", torch.tril(torch.ones(1, 1, len_context, len_context)))

        self.query = nn.Linear(n_embd, n_embd, bias=False) # (B, T, C) x (C, C) = (B, T, C)
        self.key = nn.Linear(n_embd, n_embd, bias=False) # (B, T, C)
        self.value = nn.Linear(n_embd, n_embd, bias=False) # (B, T, C)

        self.projection = nn.Linear(n_embd, n_embd, bias=False)

        self.dropout_softmax = nn.Dropout(p=0.1)
        self.dropout_linear = nn.Dropout(p=0.1)

    def forward(self, x):

        B,T,C = x.shape
        # x (B, T, C)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = (q.view(B, T, self.n_heads, self.head_size)).transpose(1, 2)
        k = (k.view(B, T, self.n_heads, self.head_size)).transpose(1, 2)
        v = (v.view(B, T, self.n_heads, self.head_size)).transpose(1, 2)


        att_scores = q @ k.transpose(-2, -1) # (B, n, T, C) x (B, n, C, T) = (B, n, T, T)

        tril = self.tril[:, :, :T, :T]

        att_scores = att_scores.masked_fill(tril==0, float('-inf'))
        att_scores = F.softmax(att_scores * (self.head_size ** -0.5), dim=-1)

        att_scores = self.dropout_softmax(att_scores)

        out = att_scores @ v # (B, n, T, T) x (B, n, T, h) = (B, n, T, h)

        out = out.transpose(1, 2) # (B, T, n, h)

        out = out.contiguous().view(B, T, C) # (B, T, C)
        out = self.projection(out)

        out = self.dropout_linear(out)

        return out

    



model = CausalMultiHeadAttention(n_embd=64, n_head=4, len_context=128)
x = torch.randn(2, 10, 64)
y = model(x)
print(y.shape)  # должно быть torch.Size([2, 10, 64])