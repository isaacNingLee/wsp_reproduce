"""Render the public math results without dependencies or model downloads."""
import json
from pathlib import Path

snapshot = json.loads((Path(__file__).resolve().parents[2] / 'results-snapshot.json').read_text())
print('| Evaluated Qwen3.5-0.8B intervention | MATH-500 (%) | GSM8K (%) |')
print('|---|---:|---:|')
rows = {}
for result in snapshot['math_results']:
    if result['thinking_enabled']:
        rows.setdefault(result['intervention'], {})[result['benchmark']] = result['acc'] * 100
for name, scores in rows.items():
    print(f"| {name} | {scores['MATH-500']:.1f} | {scores['GSM8K']:.1f} |")
for name in snapshot['missing_math_results']:
    print(f'| {name} | Not available | Not available |')
