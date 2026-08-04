import json
import base64
import os
import sys

notebook_path = sys.argv[1]
output_dir = sys.argv[2]

os.makedirs(output_dir, exist_ok=True)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

img_count = 0
for i, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'code':
        for output in cell.get('outputs', []):
            if 'data' in output and 'image/png' in output['data']:
                img_data = output['data']['image/png']
                # Sometimes image data is a list of strings, sometimes a single string
                if isinstance(img_data, list):
                    img_data = "".join(img_data)
                
                img_bytes = base64.b64decode(img_data)
                filename = os.path.join(output_dir, f'plot_{img_count}.png')
                with open(filename, 'wb') as img_file:
                    img_file.write(img_bytes)
                print(f"Extracted {filename}")
                img_count += 1
