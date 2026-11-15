import unittest
import pandas as pd
from scripts.refusal_eval import evaluate_refusals, REFUSAL_SUBSTRINGS


class RefusalEvalTest(unittest.TestCase):
    def test_official_rule_and_evidence(self):
        for phrase in REFUSAL_SUBSTRINGS:
            annotated, _ = evaluate_refusals(pd.DataFrame({
                'label': ['harmful'], 'response': ['Prefix: ' + phrase.upper() + ' suffix']}))
            self.assertEqual(annotated.iloc[0]['refusal_status'], 'refusal')
            self.assertIn(phrase, annotated.iloc[0]['matched_refusal_phrases'])

    def test_counts_missing_and_input_preservation(self):
        original = pd.DataFrame({'label': ['safe'] * 3 + ['harmful'] * 2,
                                 'response': ['I cannot help.', 'Hello!', ' ', None, 'Answer']})
        annotated, summary = evaluate_refusals(original)
        safe = summary.set_index('label').loc['safe']
        self.assertEqual((safe.refusals, safe.non_refusals, safe['empty']), (1, 1, 1))
        self.assertEqual(safe.refusal_rate, .5)
        self.assertEqual(annotated.iloc[3]['refusal_status'], 'empty')
        self.assertNotIn('refusal_status', original)

    def test_mixed_runs_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_refusals(pd.DataFrame({'label': ['safe', 'safe'],
                'response': ['Hello', 'Hello'], 'config_json': ['a', 'b']}))

    def test_absent_class_has_undefined_rate(self):
        _, summary = evaluate_refusals(pd.DataFrame({'label': ['safe'], 'response': ['Hello']}))
        self.assertTrue(pd.isna(summary.set_index('label').loc['harmful', 'refusal_rate']))


if __name__ == '__main__':
    unittest.main()
