import json
import sys

notebook_path = sys.argv[1]

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        print("".join(cell.get('source', [])))
