import torch
weights = torch.empty(2 << 30, device='cuda', dtype=torch.uint8)   # 2 GiB held, never touched by the kernels
x = torch.ones(16 << 20, device='cuda', dtype=torch.bfloat16)       # 32 MiB
a = torch.ones(1024, 1024, device='cuda', dtype=torch.bfloat16)
torch.cuda.synchronize()
for _ in range(40):
    y = x + 1
    c = a @ a
    x = y * 0.5
torch.cuda.synchronize()
print('work done')
