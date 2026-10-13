import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)

c = a @ b

if device.type == "cuda":
    torch.cuda.synchronize()

print("Result:", c.shape)