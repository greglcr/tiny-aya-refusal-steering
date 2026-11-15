"""Resumable CSV cache. One writer at a time per cache file."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd
from tqdm.auto import tqdm


def cached_responses(prompts, generate, cache_path, config):
    """Cache successful responses by exact prompt + full experiment config.

    Persist after each answer using an atomic replacement. Exceptions propagate;
    previously saved answers remain available on the next invocation.
    """
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config_json = json.dumps(config, sort_keys=True, ensure_ascii=False)
    records = []
    if path.exists():
        records = pd.read_csv(path, keep_default_na=False, dtype=str).to_dict('records')
        required = {'cache_key', 'prompt', 'config_json', 'response'}
        if not required.issubset(pd.read_csv(path, nrows=0).columns):
            raise ValueError(f'Cache incompatible : {path}. Utiliser un autre fichier.')
    cache = {}
    for record in records:
        expected = hashlib.sha256((record['config_json'] + '\n' + record['prompt']).encode()).hexdigest()
        if record['cache_key'] != expected or expected in cache:
            raise ValueError(f'Cache corrompu ou dupliqué : {path}')
        cache[expected] = record
    results = []
    hits = 0
    for row in tqdm(prompts.to_dict('records'), desc='Réponses (cache inclus)'):
        prompt = row['prompt']
        key = hashlib.sha256((config_json + '\n' + prompt).encode()).hexdigest()
        if key not in cache:
            response = generate(prompt)
            if not isinstance(response, str):
                raise TypeError('La génération doit retourner une chaîne.')
            record = {**row, 'cache_key': key, 'config_json': config_json,
                      'model_id': config['model_id'], 'response': response}
            updated = records + [record]
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                                 dir=path.parent, suffix='.tmp', delete=False) as handle:
                    temporary = Path(handle.name)
                    pd.DataFrame(updated).to_csv(handle, index=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            records = updated
            cache[key] = record
        else:
            hits += 1
        results.append({**cache[key], **row})
    print(f'{hits}/{len(results)} réponses réutilisées. Cache : {path.resolve()}')
    return pd.DataFrame(results)
