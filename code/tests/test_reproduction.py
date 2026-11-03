import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from wsp.checkpoints import load_tensors
from wsp.mathbench import last_boxed, answer_part, grade
from wsp.kernel import Kernel, _sandbox_env
from wsp import qwen


class ReproductionTests(unittest.TestCase):
    def test_grading(self):
        text = r'<think>incorrect draft \boxed{7}</think>Answer: \boxed{\frac{1}{2}}'
        self.assertEqual(last_boxed(answer_part(text)), r'\frac{1}{2}')
        self.assertTrue(grade(last_boxed(answer_part(text)), '0.5'))
        self.assertFalse(grade(None, '0.5'))
        self.assertIsNone(last_boxed(r'\boxed{unfinished'))

    def test_sharded_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); base = root/'base'; base.mkdir()
            save_file({'a': torch.ones(2)}, str(base/'part1.safetensors'))
            save_file({'b': torch.zeros(2)}, str(base/'part2.safetensors'))
            (base/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a':'part1.safetensors','b':'part2.safetensors'}}))
            (base/'config.json').write_text('{}')
            tensors = load_tensors(base)
            with patch.object(qwen, 'base_dir', return_value=str(base)):
                target = qwen.write_ckpt(tensors, {'a':torch.full((2,), 3.)}, str(root/'edited'))
            self.assertFalse((Path(target)/'model.safetensors.index.json').exists())
            actual = load_tensors(target)
            self.assertTrue(torch.equal(actual['a'], torch.full((2,), 3., dtype=torch.bfloat16)))
            self.assertTrue(torch.equal(actual['b'], tensors['b']))

    def test_control_checkpoint_export(self):
        from scripts.make_control_edits import edit, write_ckpt
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)/"base"; base.mkdir()
            (base/"config.json").write_text("{}")
            (base/"model.safetensors.index.json").write_text("{}")
            original = torch.eye(3)
            changed = edit(original, torch.tensor([1., 0., 0.]), [0], 6.)
            self.assertEqual(float(changed[0, 0]), -5.)
            target = Path(temp)/"output"
            write_ckpt(str(base), {"weight":original}, {"weight":changed}, str(target), {})
            self.assertFalse((target/"model.safetensors.index.json").exists())
            self.assertTrue(torch.equal(load_tensors(target)["weight"], changed))

    def test_sandbox(self):
        # No model/API calls: execute code through the real bubblewrap kernel.
        with tempfile.TemporaryDirectory(prefix='wsp-test-', dir=str(Path.cwd())) as temp:
            k = Kernel('wsp.tasks.sandbox_check', {'run_dir':temp}, setup_timeout=30, timeout=10,
                       visible=[sys.prefix, str(Path.cwd()/'wsp')])
            try:
                output, restarted = k.run('value = 40\nvalue + 2')
                self.assertIn('42', output); self.assertFalse(restarted)
                output, _ = k.run('value + 3')
                self.assertIn('43', output)
                output, _ = k.run('x.backward()')
                self.assertIn('REFUSED', output)
            finally:
                k.close()

    def test_credentials_not_passed_to_kernel(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY':'dummy','HF_TOKEN':'dummy'}):
            env = _sandbox_env('/tmp/unused')
        self.assertNotIn('OPENAI_API_KEY', env)
        self.assertNotIn('HF_TOKEN', env)

if __name__ == '__main__':
    unittest.main()
