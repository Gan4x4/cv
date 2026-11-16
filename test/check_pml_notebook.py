"""Run with the notebook's dependencies: python test/check_pml_notebook.py."""
import ast
import json
from pathlib import Path

import torch
from torch import nn
from pytorch_metric_learning.losses import ArcFaceLoss

notebook = Path(__file__).resolve().parents[1] / 'Representation_learning/PML_example.ipynb'
cells = json.loads(notebook.read_text())['cells']
namespace = {'torch': torch, 'nn': nn}
for cell in cells:
    if cell.get('metadata', {}).get('id') in {
        'IxHjskbI-Xgx', 'gkvtRQYZ6JST', 'kU8cC9sA9Z9K'
    }:
        tree = ast.parse(''.join(cell['source']))
        # Skip the example call; keep imports, definitions and device selection.
        tree.body = [node for node in tree.body if not isinstance(node, ast.Expr)]
        exec(compile(tree, str(notebook), 'exec'), namespace)

metric = namespace['PMLMetricWrapper']()
embeddings = torch.zeros(4, 64)
embeddings[:, 0] = torch.tensor([0., 1., 10., 11.])
labels = torch.tensor([0, 0, 1, 1])
metric.update(embeddings[:2], labels[:2])
metric.update(embeddings[2:], labels[2:])
assert metric.compute()['mean_average_precision'] == 1.0, 'Check neighbor search and labels'
metric.reset()
assert metric.labels.numel() == 0 and metric.embeddings.shape == (0, 64)
assert metric.labels.dtype == torch.long
metric.update(embeddings, torch.tensor([0, 1, 0, 1]))
assert abs(metric.compute()['mean_average_precision'] - 5 / 12) < 1e-6, 'Exclude each query from its own neighbors'

criterion = ArcFaceLoss(num_classes=2, embedding_size=64)
module = namespace['Lit'](nn.Linear(2, 64), criterion)
optimizer = module.configure_optimizers()
optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
assert all(id(p) in optimized for p in criterion.parameters()), 'ArcFace weights must be optimized'
loss = criterion(module.model(torch.randn(4, 2)), labels)
loss.backward()
optimizer.step()
assert torch.isfinite(loss), 'ArcFace training must produce a finite loss'

x0 = torch.tensor([[1., 0.]]).repeat(3, 1)
x1 = torch.tensor([[1., 0.], [-1., 0.], [0., 1.]])
batch = [(x0, x1, torch.tensor([1, -1, 1]))]
assert namespace['validation'](nn.Identity(), batch, threshold=0.) == 1.0, 'Check equality at the threshold'
single = [(x0[:1], x1[:1], torch.tensor([1]))]
assert namespace['validation'](nn.Identity(), single) == 1.0, 'One pair must remain a 1-D target'
print('PML notebook regression checks: PASS')
