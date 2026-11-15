import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from transformers import Cohere2Config, Cohere2ForCausalLM, BatchEncoding
from scripts.global_refusal import (
    GlobalConfig, split_prompts, check_model, temporary_weight_orthogonalization,
    SteeringVectorExtractor, VectorExtractionConfig, fit_and_select, generate_responses,
)


class TinyTokenizer:
    chat_template = 'test-only'
    padding_side = 'right'
    def apply_chat_template(self, messages, **kwargs):
        text = messages[0]['content'] + '!'
        if kwargs.get('tokenize', False):
            return self([text])
        return text
    pad_token_id = 0
    def decode(self, tokens, **kwargs):
        return 'Test response'
    def __call__(self, texts, **kwargs):
        ids = [[2 + ord(c) % 20 for c in s] for s in texts]
        length = max(map(len, ids))
        return BatchEncoding({'input_ids': torch.tensor([[0]*(length-len(s))+s for s in ids]),
                              'attention_mask': torch.tensor([[0]*(length-len(s))+[1]*len(s) for s in ids])})


class GlobalRefusalTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = Cohere2ForCausalLM(Cohere2Config(
            vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=64, tie_word_embeddings=False, eos_token_id=1, pad_token_id=0,
        )).cpu().eval()

    def test_stratified_split_and_duplicates(self):
        data = pd.DataFrame({'prompt': [f'p{i}' for i in range(12)],
                             'label': ['safe']*6+['harmful']*6})
        cfg = GlobalConfig(train_per_class=2, val_per_class=2)
        splits = split_prompts(data, cfg)
        self.assertEqual(splits.groupby(['split','label']).size().tolist(), [2]*6)
        pd.testing.assert_frame_equal(splits, split_prompts(data, cfg))
        data.loc[1,'prompt'] = data.loc[0,'prompt']
        with self.assertRaises(ValueError): split_prompts(data, cfg)

    def test_real_cohere2_extraction_batch_invariance(self):
        tok = TinyTokenizer()
        def extract(batch_size):
            return SteeringVectorExtractor(self.model, tok, VectorExtractionConfig(
                target_layers=[0,1], batch_size=batch_size,
                pool_method='last_token', normalize=False,
            )).extract(['alpha','b'], ['gamma','delta'])
        one, two = extract(1), extract(2)
        self.assertEqual(check_model(self.model), 2)
        for layer in one:
            torch.testing.assert_close(one[layer], two[layer], atol=1e-6, rtol=1e-4)
            self.assertGreater(one[layer].norm().item(), 0)
        self.assertEqual(tok.padding_side, 'right')
        self.assertFalse(any(m._forward_hooks for m in self.model.modules()))

    def test_weight_projection_and_exception_restoration(self):
        originals = {k:v.clone() for k,v in self.model.state_dict().items()}
        d = torch.randn(16); d /= d.norm()
        with self.assertRaisesRegex(RuntimeError,'test interruption'):
            with temporary_weight_orthogonalization(self.model, d):
                for layer in self.model.model.layers:
                    for w in (layer.self_attn.o_proj.weight, layer.mlp.down_proj.weight):
                        torch.testing.assert_close(d @ w, torch.zeros(w.shape[1]), atol=1e-6, rtol=0)
                torch.testing.assert_close(self.model.get_input_embeddings().weight @ d,
                                           torch.zeros(32), atol=1e-6, rtol=0)
                raise RuntimeError('test interruption')
        for k,v in self.model.state_dict().items():
            self.assertTrue(torch.equal(v, originals[k]), k)

    def test_end_to_end_selection_artifacts_and_no_fallback(self):
        data = pd.DataFrame({'prompt': [f'p{i}' for i in range(12)],
                             'label': ['safe']*6+['harmful']*6})
        cfg = GlobalConfig(train_per_class=2, val_per_class=2,
                           layer_fractions=(.25,), max_new_tokens=2)
        splits = split_prompts(data, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            direction, scores = fit_and_select(self.model, TinyTokenizer(), splits, tmp, cfg)
            torch.testing.assert_close(direction.norm(), torch.tensor(1.))
            self.assertEqual(scores.layer.tolist(), [0])
            manifest = json.loads((Path(tmp)/'manifest.json').read_text())
            self.assertIn('selection', manifest)
            scored = pd.read_csv(Path(tmp)/'validation_baseline.csv')
            self.assertEqual(set(scored.prompt), set(splits.loc[splits.split.eq('validation'), 'prompt']))
            self.assertTrue(set(scored.prompt).isdisjoint(splits.loc[splits.split.eq('test'), 'prompt']))
            with self.assertRaises(FileExistsError):
                fit_and_select(self.model, TinyTokenizer(), splits, tmp, cfg)
        bad = GlobalConfig(train_per_class=2, val_per_class=2,
                           layer_fractions=(.25,), max_new_tokens=2, kl_threshold=-1.)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'No candidate'):
                fit_and_select(self.model, TinyTokenizer(), splits, tmp, bad)
            self.assertFalse((Path(tmp)/'selected_direction.pt').exists())

    def test_batched_generation_preserves_order_padding_and_stops(self):
        tok = TinyTokenizer()
        data = pd.DataFrame({'prompt': ['a', 'longer', 'z'], 'label': ['safe']*3})
        one = generate_responses(self.model, tok, data, max_new_tokens=3, batch_size=1)
        batched = generate_responses(self.model, tok, data, max_new_tokens=3, batch_size=2)
        pd.testing.assert_frame_equal(one, batched)
        self.assertEqual(tok.padding_side, 'right')
        # EOS positions vary within a padded output batch; padding is not output.
        output = torch.tensor([[2, 1, 0], [2, 3, 1]])
        with patch.object(self.model, 'generate', return_value=torch.cat((torch.zeros(2, 7, dtype=torch.long), output), dim=1)):
            stopped = generate_responses(self.model, tok, data.iloc[:2], max_new_tokens=3, batch_size=2)
        self.assertEqual(stopped.generated_tokens.tolist(), [2, 3])
        self.assertEqual(stopped.hit_token_limit.tolist(), [False, False])

    def test_partial_library_failure_restored(self):
        original = self.model.get_input_embeddings().weight.detach().clone()
        def fail(model, direction):
            with torch.no_grad(): model.get_input_embeddings().weight.zero_()
            raise RuntimeError('partial failure')
        with patch('scripts.global_refusal.orthogonalize_weights', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'partial failure'):
                with temporary_weight_orthogonalization(self.model, torch.ones(16)): pass
        self.assertTrue(torch.equal(original, self.model.get_input_embeddings().weight))


if __name__ == '__main__': unittest.main()
