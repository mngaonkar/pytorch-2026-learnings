# bench_num_workers.py
import time
import torch
from torch.utils.data import DataLoader, Dataset

class SlowDataset(Dataset):
    def __len__(self):
        return 512
    def __getitem__(self, i):
        time.sleep(0.002)                       # stands in for file read + decode
        return torch.randn(3, 64, 64), i % 10

if __name__ == "__main__":
    for workers in [0, 2, 4, 8]:
        loader = DataLoader(SlowDataset(), batch_size=32, num_workers=workers,
                            pin_memory=torch.accelerator.is_available(),  # page-locked host memory
                            persistent_workers=workers > 0,
                            prefetch_factor=2 if workers > 0 else None)
        t0 = time.perf_counter()
        for x, y in loader:
            pass
        print(f"num_workers={workers}: {time.perf_counter() - t0:.2f} s/epoch")
