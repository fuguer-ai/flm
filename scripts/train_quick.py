"""Train a tiny proof-of-concept fly readout from bundled examples.

This is intentionally a smoke-test checkpoint for talking to FLM quickly, not a
benchmark or a replacement for scripts/train_conversation.py.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from safetensors.torch import save_file

from flm import conversation_config as cfg
from flm.graph import NativeBatchKernel, ReservoirBatch, sha256
from flm.model import FLM, FlyAdapter
from flm.paths import CACHE, GRAPH, ROOT


def make_record(model, user_text, assistant_text, key, max_answer_tokens):
    context = [{'role': 'user', 'content': user_text}]
    prefix = model.prompt_ids(context, max_context=100000)
    complete = model.tokenizer.apply_chat_template(
        [model.system] + context + [{'role': 'assistant', 'content': assistant_text}],
        tokenize=True,
        add_generation_prompt=False,
        **model.chat_kwargs,
    )
    if complete[:len(prefix)] != prefix:
        raise ValueError('Training/inference prefix mismatch.')
    answer = complete[len(prefix):][:max_answer_tokens]
    if len(answer) < 2:
        return None
    ids = np.asarray(prefix + answer, dtype=np.int64)
    mask = np.arange(len(ids) - 1) >= len(prefix) - 1
    return {'key': key, 'ids': ids[:-1], 'labels': ids[1:], 'mask': mask}


def batches(hidden, features, labels, batch_size, order):
    for offset in range(0, len(order), batch_size):
        ix = order[offset:offset + batch_size]
        yield hidden[ix], features[ix], labels[ix]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/quick-chat')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--examples', type=int, default=8)
    parser.add_argument('--answer-tokens', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if not 2 <= args.examples <= 32:
        parser.error('--examples must be between 2 and 32')
    if not 4 <= args.answer_tokens <= 64:
        parser.error('--answer-tokens must be between 4 and 64')
    if not 1 <= args.epochs <= 100:
        parser.error('--epochs must be between 1 and 100')
    if not 1 <= args.batch_size <= 256:
        parser.error('--batch-size must be between 1 and 256')

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'run.json').exists():
        raise SystemExit('Completed run exists; use a new --output directory.')

    started = time.monotonic()
    torch.manual_seed(27)
    np.random.seed(27)
    model = FLM(checkpoint=None, profile='conversation', device=args.device)
    kernel = NativeBatchKernel(CACHE / 'native') if (CACHE / 'native/manifest.json').exists() else None
    print(json.dumps({
        'stage': 'loaded',
        'device': model.device,
        'graph_kernel': 'native' if kernel else 'scipy',
        'base_parameters': sum(p.numel() for p in model.base.parameters()),
    }), flush=True)

    source = json.loads((ROOT / 'training/conversation-style.json').read_text())['conversations']
    records = []
    for i, lines in enumerate(source):
        record = make_record(model, lines[0], lines[1], f'conversation-style:{i}', args.answer_tokens)
        if record is not None:
            records.append(record)
        if len(records) == args.examples:
            break
    if len(records) != args.examples:
        raise ValueError('Not enough bundled examples for requested quick run.')

    # Reuse the exact shared system-prefix reservoir state for every example.
    all_ids = [r['ids'] for r in records]
    shared = 0
    while shared < min(map(len, all_ids)) and all(ids[shared] == all_ids[0][shared] for ids in all_ids):
        shared += 1
    reservoir = model.reservoir()
    for token in all_ids[0][:shared]:
        reservoir.step(model.embeddings[int(token)], 'intact')
    prefix_state = reservoir.state.copy()

    # Gather model embeddings once, then advance the full 166,700-node graph in parallel columns.
    token_ids = np.unique(np.concatenate(all_ids))
    embedding_values = model.embeddings[token_ids]
    lookup = {int(token): i for i, token in enumerate(token_ids)}
    embedded = [embedding_values[[lookup[int(token)] for token in r['ids'][shared:]]] for r in records]

    hidden_rows = []
    feature_rows = [[] for _ in records]
    label_rows = []
    with torch.no_grad():
        for r in records:
            h = model.base.model(
                torch.as_tensor(r['ids'][None, :], device=model.device), use_cache=False
            ).last_hidden_state[0]
            target_mask = torch.as_tensor(r['mask'], device=model.device)
            hidden_rows.append(h[target_mask].float().cpu().numpy())
            label_rows.append(r['labels'][r['mask']])

        for offset in range(0, len(records), 8):
            group = records[offset:offset + 8]
            vectors = embedded[offset:offset + 8]
            batch = ReservoirBatch(model.reservoir(), len(group), prefix_state, kernel)
            for t in range(max(map(len, vectors))):
                emb = np.stack([
                    v[t] if t < len(v) else np.zeros(model.hidden_size, np.float32)
                    for v in vectors
                ])
                fly = batch.step(emb, 'intact')
                for j, r in enumerate(group):
                    absolute_t = shared + t
                    if t < len(vectors[j]) and r['mask'][absolute_t]:
                        feature_rows[offset + j].append(fly[j].copy())
            print(json.dumps({
                'stage': 'graph',
                'examples': min(offset + 8, len(records)),
                'total': len(records),
                'elapsed': round(time.monotonic() - started, 1),
            }), flush=True)

    hidden = torch.as_tensor(np.concatenate(hidden_rows), device=model.device)
    features = torch.as_tensor(np.concatenate([np.asarray(x) for x in feature_rows]), device=model.device)
    labels = torch.as_tensor(np.concatenate(label_rows), device=model.device)
    if not (len(hidden) == len(features) == len(labels)):
        raise ValueError('Feature/target alignment failure.')

    adapter = FlyAdapter(model.hidden_size, scale=cfg.INTERFACE['adapter_scale']).to(model.device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=3e-4, weight_decay=.01)

    @torch.no_grad()
    def nll(current_adapter=None):
        total = 0.0
        count = 0
        order = np.arange(len(labels))
        for h, f, y in batches(hidden, features, labels, args.batch_size, order):
            logits, _, _ = model.scores(h, f, 'base' if current_adapter is None else 'intact', current_adapter)
            total += float(torch.nn.functional.cross_entropy(logits, y, reduction='sum'))
            count += len(y)
        return total / count

    base_nll = nll(None)
    curve = []
    rng = np.random.default_rng(2701)
    for epoch in range(args.epochs):
        adapter.train()
        order = rng.permutation(len(labels))
        total = 0.0
        count = 0
        for h, f, y in batches(hidden, features, labels, args.batch_size, order):
            optimizer.zero_grad(set_to_none=True)
            logits, base, _ = model.scores(h, f, adapter=adapter)
            ce = torch.nn.functional.cross_entropy(logits, y)
            # Keep the tiny quick adapter close to the already-capable frozen LM.
            divergence = torch.nn.functional.kl_div(
                torch.log_softmax(logits, -1), torch.softmax(base, -1), reduction='batchmean'
            )
            loss = ce + 0.5 * divergence
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite objective.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            total += float(ce.detach()) * len(y)
            count += len(y)
        item = {'epoch': epoch + 1, 'train_nll': total / count}
        curve.append(item)
        print(json.dumps({'stage': 'epoch', **item}), flush=True)

    adapter.eval()
    trained_nll = nll(adapter)
    weights = {k: v.detach().cpu() for k, v in adapter.state_dict().items()}
    checkpoint = args.output / 'adapter.safetensors'
    save_file(weights, str(checkpoint))

    weights_hash = sha256(model.model_dir / 'model.safetensors')
    manifest = {
        'schema_version': 2,
        'model_id': cfg.MODEL_ID,
        'model_revision': cfg.MODEL_REVISION,
        'model_weights_sha256': weights_hash,
        'graph_sha256': sha256(GRAPH / 'manifest.json'),
        'adapter_sha256': sha256(checkpoint),
        'system_text': cfg.SYSTEM_TEXT,
        'interface': cfg.INTERFACE,
        'context_tokens': cfg.CONTEXT_TOKENS,
        'sampling': cfg.SAMPLING,
        'training': {
            'kind': 'quick_smoke_test',
            'examples': len(records),
            'target_tokens': int(len(labels)),
            'epochs': args.epochs,
            'answer_tokens_per_example_max': args.answer_tokens,
            'source_sha256': sha256(ROOT / 'training/conversation-style.json'),
            'selection_sha256': hashlib.sha256(json.dumps([r['key'] for r in records]).encode()).hexdigest(),
        },
        'trainable_parameters': sum(p.numel() for p in adapter.parameters()),
        'base_parameters': sum(p.numel() for p in model.base.parameters()),
        'device': model.device,
        'claim': 'Quick local smoke-test readout trained on bundled synthetic examples; not a benchmark or biological-learning claim.',
    }
    report = {
        'base_nll': base_nll,
        'trained_nll': trained_nll,
        'base_perplexity': math.exp(base_nll),
        'trained_perplexity': math.exp(trained_nll),
        'target_tokens': int(len(labels)),
        'examples': len(records),
        'shared_prefix_tokens': shared,
        'graph_kernel': 'native' if kernel else 'scipy',
        'elapsed_seconds': round(time.monotonic() - started, 1),
        'note': 'Training-set smoke-test metrics only; use scripts/train_conversation.py for the controlled experiment.',
    }
    (args.output / 'run.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'training.json').write_text(json.dumps(curve, indent=2) + '\n')
    print(json.dumps({'stage': 'complete', 'output': str(args.output), 'report': report}), flush=True)


if __name__ == '__main__':
    main()
