# Research-Engineer Curriculum ,  Master Map

A research-engineer-first curriculum for L5-L7 AI roles (Research Engineer, Applied AI Engineer, ML Engineer, Forward Deployed Engineer) at frontier labs. Built from a skill-universe extraction over 180 indexed roles across Anthropic, OpenAI, Google, DeepMind, NVIDIA, and Microsoft AI, diffed against the reader's existing library so nothing is rebuilt twice.

## Design principles

1. **Russian doll.** Each volume builds on the previous. Knowledge flows forward; nothing is re-explained. A concept taught in Volume 1 is referenced by name later, never re-taught.
2. **Zero prior knowledge, first principles.** Every volume assumes the reader knows nothing about its topic. Terms defined on first use, acronyms expanded, steps never skipped.
3. **Research engineer who understands everything.** The spine runs ML → math → deep learning → LLM internals → pre-training → distributed training → post-training/RL → inference → agents → productionizing → research methods → papers → system design → behavioral. The reader finishes able to explain how an LLM is built end to end: data, pre-training, distribution, alignment, serving, evaluation.
4. **Build an LLM from scratch.** A single capstone thread runs through Volumes 1-8: NumPy/JAX attention → BPE tokenizer → tiny GPT on a toy corpus → distributed toy run → SFT + DPO → serve with KV cache and benchmark.
5. **Existing library as modules.** The reader already owns deep materials on agents/RAG/evals (RRK v2), ML/DL basics (refresher + 300 Q&A), and DSA (fieldbook v3, 500 atlas, 300-problem curriculum). Those are slotted in as modules; new volumes are written only for gaps.
6. **Potent, not padded.** Visual learner, ADHD, 4+ hours/day. High signal density, diagrams everywhere, zero fluff, depth never sacrificed.
7. **Generic.** No personal identifiers anywhere. Office-laptop safe.

## Skill universe (from 180 roles)

Most demanded across the index: Python, distributed training, experiment design, evaluation/evals, safety/alignment, JAX/PyTorch, pre-training, RL (RLHF/DPO/GRPO), inference optimization (CUDA/Triton/vLLM), retrieval/RAG, profiling, TPU/GPU systems, Kubernetes, C++/Rust/Go, system design, statistics/optimization. Research Engineer rows additionally demand: research depth (design experiments, ablations, read/write papers), large-scale training, responsible-AI evaluations, agent task suites and autoraters.

## The spine (read in order)

| # | Volume | File | Status |
|---|--------|------|--------|
| 0 | How to use this curriculum | `00-how-to-use.html` | NEW (coordinator) |
| 1 | Math for ML | `01-math-for-ml.html` | NEW |
| 2 | ML foundations bridge | `02-ml-foundations-bridge.html` | NEW (gaps only; basics live in existing refresher) |
| 3 | Deep learning for researchers | `03-deep-learning-for-researchers.html` | NEW (gaps only; basics live in existing refresher) |
| 4 | LLM internals | `04-llm-internals.html` | NEW |
| 5 | Pre-training | `05-pretraining.html` | NEW |
| 6 | Distributed training | `06-distributed-training.html` | NEW |
| 7 | Post-training and RL | `07-post-training-rl.html` | NEW |
| 8 | Inference and serving | `08-inference-serving-bridge.html` | NEW (gaps only; reader is already deep here) |
| 9 | Agents, RAG, production GenAI | `09-agents-rag-guide.html` | GUIDE to existing RRK v2 |
| 10 | Productionizing and MLOps | `10-productionizing-mlops.html` | NEW |
| 11 | Research methods | `11-research-methods.html` | NEW |
| 12 | Guided paper spine | `12-paper-spine.html` | NEW |
| 13 | ML system design | `13-ml-system-design.html` | NEW |
| 14 | Behavioral for research roles | `14-behavioral-research.html` | NEW |
| 16 | S30 Atlas: the 15-week lab map | `16-s30-atlas.html` | NEW (S30 integration) |
| 17 | Python and Data: the research engineer's toolkit | `17-s30-python-data.html` | NEW (S30 integration) |
| 18 | Math for ML: the machinery underneath | `18-s30-math-for-ml.html` | NEW (S30 integration) |
| 19 | Classical ML: models that still win | `19-s30-classical-ml.html` | NEW (S30 integration) |
| 20 | Evaluation and Debugging: knowing what works | `20-s30-evaluation-debugging.html` | NEW (S30 integration) |
| 21 | Deep Learning: networks that learn representations | `21-s30-deep-learning.html` | NEW (S30 integration) |
| 22 | CV and NLP Basics: seeing and reading | `22-s30-cv-nlp-basics.html` | NEW (S30 integration) |
| 23 | NLP and Transformers: the attention era | `23-s30-transformers.html` | NEW (S30 integration) |
| 24 | RAG and Retrieval: knowledge on demand | `24-s30-rag-retrieval.html` | NEW (S30 integration) |
| 25 | MLOps and Systems: from notebook to production | `25-s30-mlops-systems.html` | NEW (S30 integration) |
| 26 | System Design: designing ML systems | `26-s30-system-design.html` | NEW (S30 integration) |
| 27 | LLM and Agentic AI: production intelligence | `27-s30-llm-agentic.html` | NEW (S30 integration) |
| 28 | Interview Clearance: research-engineer loops | `28-s30-interview-clearance.html` | NEW (S30 integration) |

**Parallel track:** `dsa-track-300.html` ,  patterns-based top-300 LeetCode mastery path, deep-linking into the existing 500 atlas for full treatments. Studied alongside the spine, not inside it.

**Existing modules referenced (not rebuilt):**
- `~/workspace/your_files/rrk-prep/v2/RRK_Agentic_AI_Textbook_v2.html` + `RRK_Interview_QA_Bank_v2.html` ,  agents, RAG, eval, guardrails, GPU infra basics
- `~/workspace/your_files/ml-dl-fundamentals-refresher/ml-dl-fundamentals-refresher.html` (+300 Q&A) ,  classical ML and DL basics
- `~/workspace/your_files/dsa-interview-prep/` ,  fieldbook v3, 500 atlas, 300-problem curriculum
- `~/workspace/user/files/GOOGLE_FDE_INTERVIEW_BIBLE.md` ,  FDE interview prep

## Volume dependency graph

```
Vol 0 (how to use)
  └─ Vol 1 (math) ── Vol 2 (ML foundations) ── Vol 3 (DL for researchers)
        └─ Vol 4 (LLM internals) ── Vol 5 (pre-training) ── Vol 6 (distributed)
              └─ Vol 7 (post-training/RL) ── Vol 8 (inference bridge)
                    └─ Vol 9 (agents/RAG → RRK) ── Vol 10 (productionizing/MLOps)
                          └─ Vol 11 (research methods) ── Vol 12 (paper spine)
                                └─ Vol 13 (system design) ── Vol 14 (behavioral) ── Vol 15 (GPU kernels)
                                      └─ Vol 16 (S30 atlas) ── Vols 17-27 (lab volumes, week order) ── Vol 28 (interview clearance)
DSA track runs parallel to all of the above.
```

## Capstone thread: build an LLM from scratch

| Volume | Capstone lab |
|--------|--------------|
| 1 | NumPy linear algebra + autograd-from-scratch lab |
| 4 | Attention in NumPy/JAX; BPE tokenizer from scratch |
| 5 | Train a tiny GPT on a toy corpus; plot loss curves, scaling knobs |
| 6 | Data-parallel toy run; FSDP concepts on the tiny model |
| 7 | SFT then DPO on the tiny model; measure the behavior shift |
| 8 | Serve the tiny model with KV cache; benchmark latency/throughput |

## Pacing (4+ hours/day)

- Volumes 1-3: ~2 weeks (math is the big one; do not rush it)
- Volumes 4-8: ~3 weeks (the core LLM build; labs daily)
- Volumes 9-10: ~1 week (much is reference to owned material)
- Volumes 11-12: ~2 weeks (papers are slow reading; that is the point)
- Volumes 13-14: ~1 week (interview reps)
- DSA track: 1-2 hours/day alongside, pattern by pattern

Total: roughly 9-10 weeks at 4+ hours/day, DSA running in parallel throughout.


## S30 AI Lab integration (volumes 16-28, October 2026)

Source: S30 AI Lab (https://ai.thes30.com), "LeetCode for AI/ML": 73 labs, 474 coding
tasks, 527 tests, 15-week curriculum, scraped October 2026 into
`~/workspace/s30-build/labs/` (raw JSON), indexed in `~/workspace/s30-build/index.json`,
week structure in `~/workspace/s30-build/atlas.json`. Every lab's tasks, starter code,
hints, solutions, checkpoints, interview angles, gotchas, and interview signals are
preserved verbatim in volumes 17-27. Volume 16 is the 15-week atlas with difficulty
progression and day-by-day links into the lab volumes. Volume 28 maps every interview
round at Anthropic, Google DeepMind, and OpenAI research-engineer loops (consolidated
from seven public guides in `~/workspace/s30-build/roles/`) to the labs and to spine
volumes, with worked whiteboard-ready answers. Raw site category "ML System Design"
was normalized into "System Design" (11 category volumes). All 13 pages follow the
design-system retrofit, use inline SVG figures only, carry zero external dependencies,
and passed wm_clean.py Layer A.

Reading order: volumes 16-28 sit after volume 15 as the hands-on companion to the
spine. Volume 16 first, then 17-27 (each self-contained per category; week order
1-15 is the recommended path), volume 28 last as interview prep. Pacing: at 4+
hours/day, roughly 3-4 weeks for all 73 labs alongside the spine.

## Build log

See `BUILD-LOG.md`. All HTML passed through `wm_clean.py` Layer A before delivery.
