import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def generate(model, x, context_length, temperature=1.0):
    # X - (B, T)
    model.eval()

    for _ in range(context_length):
        x_context = x[:, -model.len_context:]
        logits, _ = model(x_context)

        logits = logits[:,-1,:] / temperature

        probs = F.softmax(logits, dim=-1)
        x_next = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, x_next), dim=1)

    return x


def generate_text(model, encode, decode, prompt="", max_new_tokens=500, temperature=1.0, device='cpu'):
    if prompt:
        encoded_prompt = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    else:
        encoded_prompt = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated = generate(model, encoded_prompt, max_new_tokens, temperature)
    text = decode(generated[0].tolist())
    
    return text


if __name__ == "__main__":
    from model import MiniGPT
    from load_data import load_dataset
    
    vocab_size = 65
    n_embed = 64
    num_heads = 4
    num_blocks = 6
    len_context = 128
    dropout = 0.1
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data, vocab_size, encode, decode = load_dataset('data/input.txt') 

    model = MiniGPT(len_context, num_heads, vocab_size, dropout, n_embed, num_blocks)  

    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Модель загружена с устройства {device}")
    print("=" * 60)
    
    prompts = ["\n", "ROMEO:", "JULIET:", "KING RICHARD:"]
    temperatures = [0.8, 1.0, 1.2]
    
    for prompt in prompts:
        for temp in temperatures:
            print(f"\nПромпт: '{prompt}' | Температура: {temp}")
            print("-" * 60)
            text = generate_text(
                model, encode, decode, 
                prompt=prompt, 
                max_new_tokens=500, 
                temperature=temp, 
                device=device
            )
            print(text)
            print("=" * 60)