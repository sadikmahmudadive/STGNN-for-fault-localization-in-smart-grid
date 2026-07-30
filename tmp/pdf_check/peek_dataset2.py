from pathlib import Path
p = Path('data/fault_dataset_huge_complex_imbalanced.npz')
with p.open('rb') as f:
    b = f.read(8)
print(b)
