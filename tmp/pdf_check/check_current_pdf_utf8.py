import sys
from pathlib import Path
sys.path.insert(0, str(Path('tmp/pydeps').resolve()))
from pypdf import PdfReader
pdf = Path(r"Reports/Graph_Neural_Networks__GNN__for_Fault_Detection_and_Localization_in_Power_Grids.pdf")
reader = PdfReader(str(pdf))
text = '\n'.join((page.extract_text() or '') for page in reader.pages)
patterns = [
    'matches or slightly exceeds ST-GNN',
    'better fit for practical fault diagnosis',
    'topology-aware pipeline',
    'The results show that the model meets the project targets',
    'mean absolute bus-index error',
    'confusion matrices in Fig. 12',
    'Noise sensitivity, ablations, and N-1 contingency robustness',
]
for p in patterns:
    print(f'=== {p} ===')
    idx = text.find(p)
    print('FOUND' if idx != -1 else 'NOT FOUND')
    if idx != -1:
        start = max(0, idx - 140)
        end = min(len(text), idx + 260)
        print(text[start:end].replace('\n', ' '))
        print()
