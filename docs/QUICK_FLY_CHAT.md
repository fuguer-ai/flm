# Quick fly chat

This path exists only to get a first local conversation through the full MaleCNS graph with minimal training work.

It uses the real 166,700-node, 25,582,938-edge prepared connectome and the pinned LFM2.5-1.2B-Instruct backbone. It trains only the small fly readout from a handful of bundled synthetic examples. Treat the result as a smoke test, not as evidence that fly topology improves language modeling.

## Windows / WSL2

Run the project inside WSL2/Ubuntu rather than native Windows. The optional fast graph kernel is built as a Linux shared object and substantially improves offline graph feature extraction.

```sh
git clone https://github.com/fuguer-ai/flm.git
cd flm
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/download.py --model-only
python scripts/prepare_graph.py
python scripts/build_graph_kernel.py
python scripts/train_quick.py
python scripts/chat.py --run runs/quick-chat
```

If `cc` is missing, install a normal C build toolchain or skip `build_graph_kernel.py`; the trainer falls back to SciPy.

For a deterministic one-shot sanity check after training:

```sh
python scripts/chat.py --run runs/quick-chat \
  --prompt "What do you think about being wired through a fruit-fly connectome?" \
  --seed 42
```

## What quick training changes

The language model weights and fly adjacency matrix remain frozen. Each token drives the fixed recurrent graph; its 166,700-dimensional state is pooled to 128 fly features. `train_quick.py` learns only the existing `FlyAdapter` that maps those features into a bounded correction of the language-model logits.

Defaults are deliberately small:

- 8 bundled conversation examples
- up to 24 supervised answer tokens per example
- 12 adapter epochs after feature extraction
- output in `runs/quick-chat/`

The expensive step is graph feature extraction. Adapter optimization itself is tiny. For the controlled development experiment with held-out validation/test data and a matched direct-input control, run `scripts/train_conversation.py` instead.
