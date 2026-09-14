# Greentext Fly — Research and Implementation Plan

## Objective

Build a system in which a real Drosophila connectome is not merely a decorative residual attached to a language model, but the principal dynamical substrate used to judge, rank, and eventually characterize greentexts.

The target architecture is:

```text
greentext
  -> frozen tokenizer / semantic embedding frontend
  -> fixed MaleCNS connectome dynamics
  -> fly state / trajectory
  -> very small trainable ranking head
  -> scalar quality score
```

The current conversational FLM path remains useful as a smoke test and demonstration, but it is not the final scientific target. In chat mode, most linguistic competence resides in the 1.2B-parameter pretrained LFM backbone and the fly supplies a bounded residual. For ranking, we can make the fly substantially more central by using the language model only as a sensory/semantic frontend and discarding its autoregressive generation head.

## Current baseline in this fork

The repository already provides the critical infrastructure:

- MaleCNS v1.0 graph preparation.
- 166,700 retained neurons.
- 25,582,938 directed connections.
- recurrent token-by-token dynamics in `flm/graph.py`.
- seeded embedding-to-fly and fly-to-128D projections.
- a matched direct-input control.
- shuffled-wiring and no-edge controls.
- an optional native sparse graph kernel.
- a quick chat smoke-test path in `scripts/train_quick.py`.

Do not alter the graph implementation merely to make the first greentext experiment more sophisticated. The initial ranking experiment should reuse the existing graph exactly, so the first result is attributable to the task rather than simultaneous architecture changes.

## Experimental question

The primary question is not "can an LLM classify greentexts?" It obviously can.

The useful question is:

> Does the real fly connectome transform semantic input into a representation that is useful for greentext preference/ranking, beyond matched non-fly controls?

For a greentext `i`, let the embedding sequence be `e_{i,1:T}`. The current fly recurrence is

```text
x_t = tanh(W_fly @ (0.6 x_{t-1} + 0.4 B e_t))
```

and the fixed projection produces `f_t in R^128`.

A document representation can initially be one of:

```text
last:      z = f_T
mean:      z = mean_t(f_t)
mean+last: z = concat(mean_t(f_t), f_T)
```

The ranker should be deliberately small, e.g.

```text
z -> Linear -> GELU -> Linear -> scalar score
```

For preference pair `A > B`, train with Bradley-Terry / logistic pairwise loss:

```text
L(A,B) = softplus(score(B) - score(A))
```

The core metric is held-out pairwise accuracy; secondary metrics are Kendall tau, Spearman rho, and NDCG when ordinal or score labels are available.

## Phase 0 — Validate the current quick-chat path

Purpose: prove the complete graph/model stack runs correctly on the local workstation before adding ranking code.

Run:

```sh
python scripts/download.py --model-only
python scripts/prepare_graph.py
python scripts/build_graph_kernel.py
python scripts/train_quick.py
python scripts/chat.py --run runs/quick-chat
```

Record:

- graph preparation success and manifest counts;
- whether the native kernel loads;
- quick-train graph extraction elapsed time;
- GPU/CPU device used by the LFM backbone;
- base NLL vs trained NLL;
- approximate interactive token latency.

This is only a systems smoke test. Do not interpret chat quality as evidence about fly topology.

## Phase 1 — Build a clean greentext corpus

### Seed corpus

Use `DarwinAnim8or/greentext` on Hugging Face as the first clean text-only corpus. It contains about 1.88k examples and provides `greentexts.jsonl` / `greentexts.txt` in a roughly 1–2 MB dataset. Its dataset license is currently listed as unknown, so treat it as a research input and do not redistribute it inside this repository without clarifying terms.

Resource:

- https://huggingface.co/datasets/DarwinAnim8or/greentext

The first data loader should download/cache externally and keep corpus files out of Git.

### Larger corpora

After the pipeline works, expand with one or more of:

- public 4chan archive exports;
- Reddit `/r/greentext` metadata for weak preference labels;
- manually curated greentext collections;
- user-created pairwise preference labels.

The preferred long-term training target is not raw post frequency. It is preference data.

### Canonical local schema

Normalize every item into a repository-independent JSONL schema such as:

```json
{"id":"...","text":">be me\n>...","source":"...","source_score":null,"created_utc":null}
```

Optional preference file:

```json
{"winner_id":"a","loser_id":"b","source":"human"}
```

Do not bake Reddit score or any other weak label into the text itself.

## Phase 2 — First fly-centric ranker

Add, approximately:

```text
flm/ranking.py
scripts/cache_greentext_features.py
scripts/train_greentext_ranker.py
scripts/rank_greentexts.py
tests/test_ranking.py
```

### Feature extraction

For each greentext:

1. tokenize with the frozen LFM tokenizer;
2. gather token embeddings from the frozen LFM embedding matrix;
3. run every token through the full fly reservoir;
4. save compact fly features and metadata;
5. never recompute graph trajectories during tiny-head hyperparameter sweeps.

Cache format should include strong provenance:

- corpus item hash;
- tokenizer/model revision;
- model-weights hash;
- graph manifest hash;
- graph-code hash;
- interface seed/configuration;
- pooling mode;
- feature format version.

This makes later comparisons reproducible and cheap.

### Keep the LLM out of the ranking head

The first ranker should not use LFM hidden states after embedding extraction. The desired experiment is:

```text
LFM embedding lookup -> fly -> rank head
```

not:

```text
LFM transformer hidden state + fly -> rank head
```

This distinction makes the fly the dominant sequence-processing substrate while retaining a semantic sensory interface.

## Phase 3 — Controls are mandatory

Every reported result should compare at least these matched representations:

1. **Fly intact** — real MaleCNS topology.
2. **Direct input** — same fixed 128D projection without graph propagation.
3. **Shuffled interface/topology alignment** — existing shuffled control.
4. **No edges** — exact disconnection sanity control.
5. **Random graph** — later addition, matched for node/edge count or degree distribution.
6. **Degree-preserving rewiring** — later and more informative topology control.

The important scientific comparison is not "fly versus chance". It is:

```text
real fly topology versus equally expressive non-fly transformations
```

If direct input beats the fly, report it plainly. That is still a valid result.

## Phase 4 — Labels and ranking strategies

### A. Personal pairwise preferences — preferred gold target

Build a tiny local UI that repeatedly presents two greentexts and records:

```text
A > B
B > A
skip / incomparable
```

Advantages:

- directly represents the intended notion of "good";
- no exposure/popularity confound;
- pairwise ranking has good statistical efficiency;
- active learning can later choose the most informative pairs.

Start with 300–1000 human comparisons. That is enough for a first personal-ranker experiment when the representation is fixed and the learned head is tiny.

### B. Reddit score as weak supervision

Reddit score can bootstrap a larger noisy dataset. Prefer pairwise construction within similar time windows and require a substantial score gap rather than regressing directly on raw karma.

For example, create `A > B` only when:

```text
log1p(score_A) - log1p(score_B) > delta
```

and both posts have roughly comparable exposure conditions.

Use weak labels for training scale, but keep a manually judged holdout for the final evaluation.

### C. Synthetic/LLM labels

Use only as a bootstrap/debugging device. They should never be the sole evidence that the fly learned human taste, because the evaluation would collapse into imitation of the labeling model.

## Phase 5 — State-trajectory analysis

The fly representation is more interesting than a single score. Preserve per-token or per-line trajectories for a diagnostic subset:

```text
f_1, f_2, ..., f_T
```

Analyze:

- PCA / UMAP trajectories;
- `||f_t - f_{t-1}||` state-change magnitude;
- sentence/line boundaries with largest state changes;
- trajectory length;
- terminal-state clusters;
- relationship between state geometry and rank score;
- differences between intact and shuffled fly trajectories.

A particularly useful visualization is line-by-line greentext trajectory in 2D fly-state space. Annotate setup, escalation, twist, and punchline lines when available. This may reveal whether high-quality greentexts produce characteristic dynamical transitions rather than merely separable terminal embeddings.

## Phase 6 — Make the fly itself learn more

Do this only after the fixed-connectome baseline is measured.

There are several increasingly invasive options.

### 6.1 Train input/output interfaces

Keep `W_fly` fixed but train `B`, `P`, or constrained low-rank variants. This asks whether the biological topology becomes more useful when the sensory/readout mapping adapts to the task.

### 6.2 Per-neuron gain

Introduce a learnable gain `g_i`:

```text
x_{t+1} = tanh(g * (W_fly @ u_t))
```

This adds only about 166,700 trainable scalars while preserving every anatomical edge.

### 6.3 Per-neuron time constants / leak

Learn `lambda_i in (0,1)`:

```text
x_{t+1} = (1-lambda) * x_t + lambda * tanh(W_fly @ u_t)
```

This is attractive because neuronal timescale is biologically plausible and changes dynamics without destroying topology.

### 6.4 Cell-type or neurotransmitter gains

Use MaleCNS annotations/neurotransmitter predictions to share trainable parameters across biological classes rather than assigning arbitrary edge-specific weights.

### 6.5 Edge-weight training — avoid initially

Training millions of independent synaptic weights would largely destroy the meaning of "use the fly connectome". If ever explored, preserve anatomical sparsity and compare against equally parameterized random topologies.

## Phase 7 — More biological dynamics

The current FLM recurrence is intentionally abstract and unsigned. More biological simulation is a separate research branch, not a prerequisite for greentext ranking.

Useful reference projects/resources:

- upstream FLM: https://github.com/nftechie/flm
- MaleCNS/FlyWire access tutorials: https://github.com/seung-lab/FlyConnectome
- FlyWire LIF/embodied simulation reference: https://github.com/pusulamkendim/flywire-neuro
- connectome preprocessing examples: https://github.com/YijieYin/connectome_data_prep
- GPU-oriented whole-fly simulation work: https://github.com/eonfathom/FastFly
- FlyWire whole-brain simulation example: https://github.com/theflyRH/thefly-brain

Potential future changes include neurotransmitter sign, membrane/leak dynamics, delays, and biologically grouped parameters. These should be evaluated as separate model families because they change the hypothesis being tested.

## Suggested experiment matrix

The smallest useful study after feature extraction is:

| Representation | Trainable head | Real fly topology | Purpose |
|---|---:|---:|---|
| Intact fly | yes | yes | primary model |
| Direct 128D | yes | no | matched input baseline |
| Shuffled fly | yes | topology preserved but misaligned | topology/interface control |
| No edges | yes | disconnected | sanity control |

Then extend to:

| Representation | Purpose |
|---|---|
| degree-preserving rewired fly | test topology specificity |
| random graph, same density | generic recurrent-network baseline |
| frozen LFM pooled embeddings | strong semantic baseline |
| full LFM hidden-state ranker | practical upper baseline, not fly-centric |

Use identical train/validation/test item splits for all models. Never split generated pairs independently if the same greentext can occur in both train and test; split by document first, then construct pairs within each split.

## Evaluation

Primary:

- pairwise preference accuracy on held-out greentexts.

Secondary:

- Kendall tau against ordinal/manual ranking;
- Spearman rho;
- NDCG@K;
- top-K overlap;
- bootstrap confidence intervals for Fly minus Direct and Fly minus Rewired;
- calibration of pairwise probability if useful.

For personal preference data, repeated comparisons can estimate human self-consistency. Model performance above the user's own repeat-consistency ceiling should be treated skeptically and checked for leakage.

## Performance priorities

The graph traversal is the expensive component. Optimize in this order:

1. build/use the existing native sparse kernel;
2. cache fly features once;
3. batch independent documents as graph columns;
4. reuse common prefix state when applicable;
5. avoid recomputing the LFM transformer when only embeddings are required;
6. only then consider a CUDA sparse implementation.

For the ranker, the LFM transformer may not need to run at all after tokenizer/embedding loading. This can make the fly-centric ranking path considerably lighter than conversational FLM.

## Immediate implementation order

1. Run and benchmark `quick-chat` locally.
2. Add a downloader/normalizer for the 1.88k greentext seed corpus, without committing corpus data.
3. Add fly feature extraction with `last`, `mean`, and `mean+last` pooling.
4. Add a tiny binary/pairwise ranking head.
5. Create an initial weak-label or manually labeled preference set.
6. Run Intact vs Direct vs Shuffled vs No-edge on identical held-out documents.
7. Only after that result, add degree-preserving rewiring.
8. Add trajectory visualizations.
9. If the topology shows signal, test learnable per-neuron gain/time constants.
10. Explore more biological LIF/neurotransmitter dynamics only as a separate branch.

## Success criteria

The first meaningful milestone is not high absolute accuracy. It is a reproducible result with clean controls.

A compelling positive result would be a statistically stable ordering such as:

```text
intact fly > degree-preserving rewired ~= direct > no-edge
```

on unseen greentexts.

An equally useful negative result would be:

```text
direct >= intact fly
```

which would show that the current MaleCNS reservoir adds no ranking value under this interface/dynamics. That would motivate changing the biological dynamics or trainable interface rather than tuning the ranker until the fly appears to win.

## Scientific guardrails

- Do not describe the system as a biological fly understanding English.
- Keep the distinction between anatomical connectivity, engineered dynamics, and learned task parameters explicit.
- Keep the language-model frontend frozen in the first ranking experiments.
- Preserve exact dataset/model/graph provenance in every feature cache and run manifest.
- Prefer matched controls over impressive standalone scores.
- Do not tune experimental choices on the final held-out test set.

The project becomes interesting precisely when we can state exactly which computation came from pretrained language semantics, which came from the fly topology, and which came from the learned readout.
