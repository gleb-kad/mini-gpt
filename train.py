import torch

from model import MiniGPT
from load_data import load_dataset

vocab_size = 65
n_embed = 64
num_heads = 4
num_blocks = 6
len_context = 128
dropout=0.1
batch_size=16
eval_iters = 200
eval_interval=500
max_iters=5000
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_batch(split):
    data = train_data if split == 'train' else val_data
    
    # Случайно выбираем начальные позиции для батча
    ix = torch.randint(len(data) - len_context, (batch_size,))
    
    # Создаем x и y (сдвиг на 1 токен)
    x = torch.stack([data[i:i+len_context] for i in ix])  # [B, T]
    y = torch.stack([data[i+1:i+len_context+1] for i in ix])  # [B, T]
    
    x, y = x.to(device), y.to(device)
    return x, y

data, vocab_size, encode, decode = load_dataset('data/input.txt')
print(f"Dataset size: {len(data)} characters")
print(f"Vocabulary size: {vocab_size} unique characters")

@torch.no_grad()
def estimate_loss():
    """Оценивает средний loss на train и val"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

model = MiniGPT(len_context, 
                num_heads, 
                vocab_size, 
                dropout,
                n_embed,
                num_blocks)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"### Model has {params:,} parameters ###")

optimizer = torch.optim.AdamW(model.parameters(), 3e-3)

for step in range(max_iters):

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    

    xb, yb = get_batch('train')

    logits,loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'model.pth')
print("Модель сохранена в model.pth")