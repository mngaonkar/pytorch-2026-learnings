# overlap_host_to_device.py
import torch
from torch.utils.data import DataLoader, TensorDataset

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
data = TensorDataset(torch.randn(1024, 128), torch.randint(0, 10, (1024,)))
loader = DataLoader(data, batch_size=64, pin_memory=torch.accelerator.is_available())

model = torch.nn.Linear(128, 10).to(device)
for x, y in loader:
    # with pinned memory, non_blocking=True lets the copy overlap with compute
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    loss = torch.nn.functional.cross_entropy(model(x), y)
print("last loss", loss.item())
