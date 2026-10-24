import torch

dtype = torch.float
ddevice = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

a = torch.randn(2, 3, device=ddevice, dtype=dtype)
b = torch.randn(1, 3, device=ddevice, dtype=dtype)

print("张量 a:")
print(a)
print("张量 b:")
print(b)

c = a + b
print("a 和 b 的和:")
print(c)