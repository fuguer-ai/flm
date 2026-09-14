# Quick fly-chat validation notes

The quick-start path is deliberately separated from the controlled conversation experiment.

Validated without external downloads:

- `scripts/train_quick.py` parses as Python.
- `scripts/download.py` exposes `--model-only` and skips corpus downloads when selected.
- The quick trainer writes the schema/version, graph hash, adapter hash, system text, interface, context, and sampling fields required by `FLM` checkpoint loading.
- The quick trainer reuses the same `ReservoirBatch`/`NativeBatchKernel` graph path as the controlled trainer and never modifies the connectome weights or base-model weights.

Full integration still requires the pinned language-model weights and prepared MaleCNS graph, so the end-to-end run must be performed in an environment that can download those artifacts.
