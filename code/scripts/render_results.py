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

if 'model_merging' in snapshot:
    merging = snapshot['model_merging']
    print('\n| Merging method | CIFAR-100 (%) | CUB (%) | Mean (%) |')
    print('|---|---:|---:|---:|')
    for key, label in [('average', 'Plain averaging'), ('task_arithmetic', 'Task arithmetic tuned on practice')]:
        scores = merging['baselines']['holdout'][key]
        print(f"| {label} | {100*scores['cifar100']:.2f} | {100*scores['cub']:.2f} | {100*scores['mean']:.2f} |")
    for editor in merging['editors']:
        scores = editor['holdout']
        label = editor['editor'] + (' (submission failure)' if not editor['beats_average'] else '')
        print(f"| {label} | {100*scores['cifar100']:.2f} | {100*scores['cub']:.2f} | {100*scores['mean']:.2f} |")
