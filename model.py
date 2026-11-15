import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiHeadAttention(nn.Module):
    """
    Multi Head Attention block

    parameters:
    x - input Tensor of B, T, C dimensions where
        B - batch_size,
        T - len_context - maximum number of tokens in input prompt
        C - n_embd, inner size of data representation
    n_head - numbers of attention heads

    output:
    out - matrix B, T, C - logits of attention block
    """

    def __init__(self, n_embd: int=64, len_context: int=128, n_head: int=4):
        super().__init__()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B,T,C = x.shape

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


class FeedForward(nn.Module):
    """
    Fully-connected network that adds complexity and non-linearity to attention head

    parameters:
    x - input Tensor of B, T, C dimensions where:
        B - batch_size,
        T - len_context - maximum number of tokens in input prompt
        C - n_embd, inner size of data representation. The output of attention block
    n_embd - inner size of data representation
    dropout - percentage of neurons to be shutted down to generalize the model

    output:
    out - tensor of B, T, C dimension
    """
    def __init__(self, n_embd: int=64, dropout: int=0.1):
        super().__init__()

        self.feed = nn.Sequential(nn.Linear(n_embd, n_embd*4), 
                                   nn.GELU(approximate='tanh'), 
                                   nn.Linear(n_embd*4, n_embd), 
                                   nn.Dropout(p=dropout))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.feed(x)
        return out

class TransformerBlock(nn.Module):
    """
    Transformer proccesing unit. Consists of MultiHead Attention block and linear MLP network.

    parameters:
    x - input Tensor of B, T, C dimensions where:
        B - batch_size,
        T - len_context - maximum number of tokens in input prompt
        C - n_embd, inner size of data representation
    n_head - numbers of attention heads
    dropout - percentage of neurons to be shutted down to generalize the model

    output:
    out - tensor of B, T, C dimension
    """
    def __init__(self, 
                n_embd: int=64,
                n_heads: int=4, 
                dropout: int=0.1, 
                len_context: int=128):
        
        super().__init__()

        self.attention_block = CausalMultiHeadAttention(n_embd=n_embd, 
                                                        n_head=n_heads, 
                                                        len_context=len_context)
        
        self.ffn = FeedForward(n_embd=n_embd, dropout=dropout)

        self.layer_norm1 = nn.LayerNorm(n_embd, 1e-5)
        self.layer_norm2 = nn.LayerNorm(n_embd, 1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention_block(self.layer_norm1(x))
        x = x + self.ffn(self.layer_norm2(x))

        return x
    
class MiniGPT(nn.Module):
    """
    Neural Network consists of n-Transformer blocks, 
    normalized with LayerNormalization and put through linear layer 
    to mix and aggregate all n-Transformer blocks outputs. 
    Uses CrossEntropyLoss function

    parameters:
    x - input Tensor of B, T, C dimensions where:
        B - batch_size,
        T - len_context - maximum number of tokens in input prompt
        C - n_embd, inner size of data representation
    num_heads - numbers of attention heads
    vocab_size - number of seen tokens in Embedding Table
    dropout - percentage of neurons to be shutted down to generalize the model
    num_blocks - number of Transformer blocks in architecture

    output:
    logits - probabilities of Transformer that need to be transformed from n_embed back to vocab_size
    loss - CrossEntropyLoss value used in training loop for reduction
    """
    def __init__(self, 
                 len_context: int=128, 
                 num_heads: int=4, 
                 vocab_size: int=128, 
                 dropout: int=0.1, 
                 n_embd: int=64, 
                 num_blocks: int=6):
        
        super().__init__()
        self.len_context = len_context
        self.head_size = n_embd//num_heads

        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.position_embed = nn.Embedding(len_context, n_embd)

        self.cluster = nn.Sequential(*[TransformerBlock(n_embd=n_embd, 
                                                  n_heads=num_heads, 
                                                  dropout=dropout, 
                                                  len_context=len_context) 
                                                  for _ in range(num_blocks)])

        self.layernorm = nn.LayerNorm(n_embd, 1e-5)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, 
                targets: torch.Tensor | None=None) -> torch.Tuple[torch.Tensor, torch.Tensor | None]:

        B, T = x.shape

        x_embd = self.token_embed(x)
        x_pos =  self.position_embed(torch.arange(T, device=x.device))
        x = x_embd + x_pos

        x = self.cluster(x)
        x = self.layernorm(x)

        logits = self.lm_head(x)

        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        else:
            loss = None
        return logits, loss