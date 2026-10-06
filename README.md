# llm-knowledge-base

A research-engineer-first knowledge base on large language models: the math and ML foundations, deep learning, LLM internals, pre-training, distributed training, post-training and RL, inference and serving, agents and RAG, productionizing ML, research methods, a guided paper spine, ML system design, GPU kernels, and communicating research.

## How to use it

Open `index.html` in any browser and read the volumes in order. Every page is self-contained: styles, scripts, and images are inlined, so the files work from `file://` with no network connection and no build step. Volumes marked below close with interview questions worked from their own chapters.

## Contents

### Base volumes (16)

The spine. Read in order; each volume builds on the earlier ones.

| Vol | File | Topic |
|-----|------|-------|
| 0 | `00-how-to-use.html` | How to use this knowledge base: order, pacing, capstone project |
| 1 | `01-math-for-ml.html` | Math for ML: linear algebra, probability, statistics, optimization, information theory, calculus |
| 2 | `02-ml-foundations-bridge.html` | ML foundations: experiment design, eval methodology, metrics, leakage and contamination |
| 3 | `03-deep-learning-for-researchers.html` | Deep learning: normalization, initialization, optimizers, mixed precision, checkpointing |
| 4 | `04-llm-internals.html` | LLM internals: tokenization, attention with worked numerics, the transformer block, RoPE, MHA/MQA/GQA, KV cache. Closes with interview Q&A. |
| 5 | `05-pretraining.html` | Pre-training: data pipelines, scaling laws, the training loop, inside a pre-training run. Closes with interview Q&A. |
| 6 | `06-distributed-training.html` | Distributed training: DDP, FSDP/ZeRO, tensor and pipeline parallelism, checkpointing, fault tolerance, profiling. Closes with interview Q&A. |
| 7 | `07-post-training-rl.html` | Post-training and RL: SFT, RLHF, DPO, GRPO, RLVR, reward hacking. Closes with interview Q&A. |
| 8 | `08-inference-serving-bridge.html` | Inference and serving: batching, disaggregation, quantization, benchmarking rigor. Closes with interview Q&A. |
| 9 | `09-agents-rag-guide.html` | Agents, RAG, production GenAI: agents, retrieval, evals, guardrails |
| 10 | `10-productionizing-mlops.html` | Productionizing and MLOps: pipelines, CI/CD for ML, monitoring, drift, cost engineering, reliability |
| 11 | `11-research-methods.html` | Research methods: reading papers, experiment design, ablations, negative results, reproducible write-ups |
| 12 | `12-paper-spine.html` | Guided paper spine: Transformer, GPT-3, InstructGPT, scaling laws, DPO, MoE, R1, each taught from zero |
| 13 | `13-ml-system-design.html` | ML system design: LLM serving systems, training platforms, eval platforms, RAG at scale, cost/latency tradeoffs. Closes with interview Q&A. |
| 14 | `14-communicating-research.html` | Communicating research: research narratives, failure and conflict stories, practice reps, calibration |
| 15 | `15-gpu-kernels.html` | GPU architecture and kernel programming |

### S30 AI Lab (13)

Hands-on lab pages. Each one pairs with the base volumes: short theory, then guided labs with concrete tasks. Read a lab after its matching base volume.

| Lab | File | Topic |
|-----|------|-------|
| 16 | `16-s30-atlas.html` | Lab map: the full 15-week plan, week by week |
| 17 | `17-s30-python-data.html` | Python and data tooling for ML work |
| 18 | `18-s30-math-for-ml.html` | Math for ML, taught through labs |
| 19 | `19-s30-classical-ml.html` | Classical ML: regression, trees, ensembles, validation |
| 20 | `20-s30-evaluation-debugging.html` | Evaluation and debugging: metrics, error analysis, ablations |
| 21 | `21-s30-deep-learning.html` | Deep learning labs: training loops, regularization, tuning |
| 22 | `22-s30-cv-nlp-basics.html` | Computer vision and NLP foundations |
| 23 | `23-s30-transformers.html` | Transformers, built step by step |
| 24 | `24-s30-rag-retrieval.html` | RAG and retrieval: chunking, embeddings, reranking |
| 25 | `25-s30-mlops-systems.html` | MLOps and systems: deployment, monitoring, pipelines |
| 26 | `26-s30-system-design.html` | ML system design drills |
| 27 | `27-s30-llm-agentic.html` | LLM and agentic AI: prompting, tools, agents |
| 28 | `28-s30-interview-clearance.html` | Interview clearance: full loop preparation, round by round |

### Role tracks (10)

Role-focused guides built on the spine. Each track re-orders the base volumes into a reading path for that role. It then adds specialist chapters that go deeper than the base volumes on the role's core problems.

| Track | File | Focus |
|-------|------|-------|
| 1 | `track-01-llm-research-engineer.html` | LLM Research Engineer: the generalist path, end to end |
| 2 | `track-02-post-training-alignment.html` | Post-Training and Alignment Engineer: SFT, RLHF, DPO, reward modeling |
| 3 | `track-03-inference-serving.html` | Inference and Serving Engineering: low-latency, high-throughput serving |
| 4 | `track-04-distributed-training-systems.html` | Distributed Training Systems Engineer: multi-GPU and multi-node training |
| 5 | `track-05-evals-safety.html` | Evals and Safety Engineer: eval methodology, benchmarks, red-teaming, safety systems |
| 6 | `track-06-agent-systems.html` | Agent Systems Engineer: tools, planning, multi-agent patterns, evals, guardrails |
| 7 | `track-07-forward-deployed-engineer.html` | Forward Deployed Engineer / AI Consultant: shipping AI inside real organizations |
| 8 | `track-08-ai-engineer.html` | AI Engineer: applied LLM product engineering, RAG, prompting, evals |
| 9 | `track-09-ml-engineer.html` | ML Engineer: production ML systems, pipelines, monitoring, reliability |
| 10 | `track-10-software-engineer.html` | Software Engineer, Backend and Systems: APIs, data systems, distributed services |

### Crash track (11)

Condensed interview preparation. Each page states decision rules, must-memorize numbers and formulas, classic traps, and rapid-fire Q&A with worked answers.

| File | Topic |
|------|-------|
| `crash-base.html` | Base curriculum: the condensed spine |
| `crash-track-01.html` | LLM Research Engineer |
| `crash-track-02.html` | Post-training and alignment |
| `crash-track-03.html` | Inference and serving |
| `crash-track-04.html` | Distributed training |
| `crash-track-05.html` | Evals and safety |
| `crash-track-06.html` | Agent systems |
| `crash-track-07.html` | Forward deployed engineering |
| `crash-track-08.html` | AI Engineer |
| `crash-track-09.html` | ML Engineer |
| `crash-track-10.html` | Software engineer, backend and systems |

### Parallel track

- `dsa-track-300.html` - algorithms track: 300 problems organized pattern by pattern, runs alongside the spine.

## Notes

- The `build/` directory holds the Markdown-first build pipeline (sources, canonical stylesheet and script, QA gates). Built HTML at the repo root is the published artifact.
- Every page carries the full site chrome: header navigation across all volumes and tracks, breadcrumbs, previous/next volume links, and a footer.
