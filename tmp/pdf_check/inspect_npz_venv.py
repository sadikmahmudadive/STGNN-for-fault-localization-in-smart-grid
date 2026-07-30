import numpy as np
from pathlib import Path
p = Path('data/fault_dataset_huge_complex_imbalanced.npz')
for allow in (False, True):
    try:
        d = np.load(p, allow_pickle=allow)
        print('ALLOW_PICKLE=', allow, 'FILES=', d.files)
        for k in d.files:
            arr = d[k]
            print(k, arr.dtype, arr.shape)
    except Exception as e:
        print('ALLOW_PICKLE=', allow, type(e).__name__, e)
