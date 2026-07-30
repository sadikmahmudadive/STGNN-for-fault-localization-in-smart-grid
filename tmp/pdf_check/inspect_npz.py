import numpy as np
from pathlib import Path
p = Path('data/fault_dataset_huge_complex_imbalanced.npz')
try:
    data = np.load(p)
    print('loaded_without_pickle')
    print(data.files)
    for k in data.files:
        arr = data[k]
        print(k, arr.dtype, arr.shape)
except Exception as e:
    print(type(e).__name__, e)
