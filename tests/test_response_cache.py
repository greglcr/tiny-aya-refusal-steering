import tempfile
import unittest
from pathlib import Path

import pandas as pd
from scripts.response_cache import cached_responses


class ResponseCacheTest(unittest.TestCase):
    def test_resume_and_invalidation(self):
        # Test atomic replacement on the same filesystem as the real cache.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / 'responses.csv'
            prompts = pd.DataFrame({'prompt': ['first', 'second'], 'example_id': ['a', 'b']})
            config = {'model_id': 'fake', 'max_new_tokens': 256}
            def interrupted(prompt):
                if prompt == 'second':
                    raise RuntimeError('interruption')
                return ''
            with self.assertRaises(RuntimeError):
                cached_responses(prompts, interrupted, path, config)
            calls = []
            def generate(prompt):
                calls.append(prompt)
                return 'réponse, avec "guillemets"\net retour ligne'
            result = cached_responses(prompts, generate, path, config)
            self.assertEqual(calls, ['second'])
            self.assertEqual(result.iloc[0]['response'], '')
            calls.clear()
            reread = cached_responses(prompts.iloc[::-1], generate, path, config)
            self.assertEqual(calls, [])
            self.assertEqual(reread.iloc[0]['response'], result.iloc[1]['response'])
            cached_responses(prompts, generate, path, {**config, 'max_new_tokens': 128})
            self.assertEqual(calls, ['first', 'second'])
            self.assertEqual(len(pd.read_csv(path)), 4)


if __name__ == '__main__':
    unittest.main()
