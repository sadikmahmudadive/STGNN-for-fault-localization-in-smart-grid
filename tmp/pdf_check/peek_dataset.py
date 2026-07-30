from pathlib import Path
p = Path('data/fault_dataset_huge_complex_imbalanced.npz')
print(p.exists(), p.stat().st_size)
with p.open('rb') as f:
    b = f.read(64)
print(b)
