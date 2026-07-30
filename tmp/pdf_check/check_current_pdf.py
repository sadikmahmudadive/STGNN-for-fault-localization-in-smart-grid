import sys
from pathlib import Path
sys.path.insert(0, str(Path('tmp/pydeps').resolve()))
from pypdf import PdfReader
pdf = Path(r"Reports/Graph_Neural_Networks__GNN__for_Fault_Detection_and_Localization_in_Power_Grids.pdf")
reader = PdfReader(str(pdf))
print(f'PAGES={len(reader.pages)}')
full = []
for i, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ''
    full.append(text)
    head = text.replace('\n', ' ')[:420]
    print(f'PAGE_{i}_HEAD={head}')
text = '\n'.join(full)
patterns = [
    'matches or slightly exceeds ST-GNN',
    'better fit for practical fault diagnosis',
    'multi-task prediction, and robustness analysis',
    'topology-aware pipeline',
    '27 distinct sliding windows',
    'The proposed model comfortably meets the project targets',
    'The results show that the model meets the project targets',
]
for p in patterns:
    print(f'\n=== {p} ===')
    idx = text.find(p)
    print(idx)
    if idx != -1:
        start = max(0, idx - 120)
        end = min(len(text), idx + 260)
        print(text[start:end].replace('\n',' '))
