# llm-knowledge-base

A research-engineer-first knowledge base on large language models: the math and ML foundations, deep learning, LLM internals, pre-training, distributed training, post-training and RL, inference and serving, agents and RAG, productionizing ML, research methods, a guided paper spine, ML system design, GPU kernels, and communicating research.

## How to use it

Open `index.html` in any browser and read the volumes in order. Every page is self-contained: styles, scripts, and images are inlined, so the files work from `file://` with no network connection and no build step.

## Contents

### Base volumes (16)

The spine. Read in order; each volume builds on the earlier ones.

| Vol | File | Topic |
|-----|------|-------|
| 0 | `00-how-to-use.html` | How to use this knowledge base: order, pacing, capstone project |
| 1 | `01-math-for-ml.html` | Math for ML: linear algebra, probability, statistics, optimization, information theory, calculus |
| 2 | `02-ml-foundations-bridge.html` | ML foundations: experiment design, eval methodology, metrics, leakage and contamination |
| 3 | `03-deep-learning-for-researchers.html` | Deep learning: normalization, initialization, optimizers, mixed precision, checkpointing |
| 4 | `04-llm-internals.html` | LLM internals: tokenization, attention with worked numerics, the transformer block, RoPE, MHA/MQA/GQA, KV cache |
| 5 | `05-pretraining.html` | Pre-training: data pipelines, scaling laws, the training loop, inside a pre-training run |
| 6 | `06-distributed-training.html` | Distributed training: DDP, FSDP/ZeRO, tensor and pipeline parallelism, checkpointing, fault tolerance, profiling |
| 7 | `07-post-training-rl.html` | Post-training and RL: SFT, RLHF, DPO, GRPO, RLVR, reward hacking |
| 8 | `08-inference-serving-bridge.html` | Inference and serving: batching, disaggregation, quantization, benchmarking rigor |
| 9 | `09-agents-rag-guide.html` | Agents, RAG, production GenAI: agents, retrieval, evals, guardrails |
| 10 | `10-productionizing-mlops.html` | Productionizing and MLOps: pipelines, CI/CD for ML, monitoring, drift, cost engineering, reliability |
| 11 | `11-research-methods.html` | Research methods: reading papers, experiment design, ablations, negative results, reproducible write-ups |
| 12 | `12-paper-spine.html` | Guided paper spine: Transformer, GPT-3, InstructGPT, scaling laws, DPO, MoE, R1, each taught from zero |
| 13 | `13-ml-system-design.html` | ML system design: LLM serving systems, training platforms, eval platforms, RAG at scale, cost/latency tradeoffs |
| 14 | `14-communicating-research.html` | Communicating research: research narratives, failure and conflict stories, practice reps, calibration |
| 15 | `15-gpu-kernels.html` | GPU architecture and kernel programming |

### Role tracks (10)

Role-focused guides built on the spine. Each track re-orders the base volumes into a reading path for that role, then adds specialist chapters that go deeper than the base volumes on the role's core problems.

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

### Parallel track

- `dsa-track-300.html` - algorithms track: 300 problems organized pattern by pattern, runs alongside the spine.

## Notes

- The `build/` directory holds the Markdown-first build pipeline (sources, canonical stylesheet and script, QA gates). Built HTML at the repo root is the published artifact.
