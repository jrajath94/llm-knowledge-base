---
title: Volume 7 Appendix A - RL Systems Engineering
eyebrow: Volume 7 · Appendix A - RL Systems Engineering
---

# Appendix A: RL Systems Engineering

Volume 7 taught the algorithms of post-training: SFT, RLHF, PPO, DPO, GRPO. This appendix is the machinery those algorithms run on. It answers the question every RL practitioner hits within a week: why is my training loop spending all its time generating text instead of learning from it, and what do I build about it?

Three chapters, from zero:

- **A.1** Rollout generation throughput for online RL: why generation dominates the step, and how to budget it.
- **A.2** Inference engines inside the RL loop: how vLLM and SGLang serve as rollout workers.
- **A.3** Async RL architectures and memory layout: decoupled rollout fleets, weight sync, and what a rollout worker carries in GPU memory.

Prerequisites, assumed cold: the online RL loop from Volume 7 (generate, score, update), KV cache from Volume 4, data parallelism from Volume 6. Forward link: Volume 8 serves the aligned model with the same engines you meet here.

## A.1 Rollout generation throughput for online RL

### A.1.1 What a rollout is

A %%rollout%% is one full model-generated trajectory used as training data: a prompt goes in, the model generates tokens until it stops, and the result is scored. In online RL, every training step starts by generating fresh rollouts from the current policy. "Online" means the data is always new, always from the model you are training right now. Contrast with offline methods like DPO, which learn from a fixed dataset generated once.

One step of online RL has two phases:

1. **Generate.** Sample a batch of prompts. For each prompt, sample G completions (G = 8 is typical for GRPO-style training). Score each completion with a reward function or verifier.
2. **Update.** Run the gradient step on those completions: forward pass, backward pass, optimizer step.

The surprise, the first time you run this, is the clock. Phase 1 takes the overwhelming majority of the time. This chapter explains why, and gives you the arithmetic to budget it.

### A.1.2 Why generation dominates the step

Generation is %%autoregressive decoding%%: one token at a time, each token requiring a full pass over the model weights. Training is the opposite: hundreds of tokens processed in parallel through big matrix multiplies that keep the GPU's compute units busy.

Two different bottlenecks:

- **Decoding** is %%memory-bandwidth bound%%. To produce one token for one sequence, the GPU must read every weight from memory. A 7B model in bf16 is 14 GB of weights. At ~3 TB/s of memory bandwidth, reading all of them for one token sets a hard ceiling near 200 tokens per second per GPU. Real systems land well below that once batching overhead and the KV cache are counted.
- **Training** is %%compute bound%%. A forward-backward pass over a batch of sequences reuses each weight across hundreds of tokens at once, so the GPU's arithmetic units, not its memory bus, set the pace. Throughput per token is an order of magnitude higher.

So the same 2 million tokens cost roughly 10x more wall clock to generate than to train on. That ratio is the whole story of RL systems engineering: everything in A.2 and A.3 exists to push generation throughput up.

### A.1.3 How an online RL step works, token by token

Walk through one step with concrete shapes. Batch: 128 prompts. Group size G = 8 completions per prompt. Mean completion length: 2,000 tokens (long chain-of-thought, the R1 regime).

Total tokens to generate this step: 128 x 8 x 2,000 = 2,048,000. About 2M tokens.

The update phase replays those same 2M tokens (plus prompt tokens, a smaller addition) through forward and backward passes. Same token count, wildly different cost per token, because decode and train hit different hardware limits.

The code below is a step-time budget model. It is deliberately simple: two throughput numbers in, one fraction out. The fraction is what you use to decide where your next GPU goes.

```python
# Online RL step-time budget: where does the wall clock go?
# One step = generate rollouts, then run the gradient update. This model
# answers a single question: what fraction of the step is generation?
# Change the knobs and watch the answer move.

def step_budget(
    n_prompts,          # prompts per step (the batch)
    group_size,         # completions sampled per prompt (G in GRPO)
    avg_tokens,         # mean completion length in tokens
    rollout_tok_per_s,  # combined generation throughput of the rollout pool
    train_tok_per_s,    # combined training throughput of the trainer pool
):
    # Total tokens the step must generate: prompts x completions x length.
    # This product is the number that matters; everything else is commentary.
    gen_tokens = n_prompts * group_size * avg_tokens

    # Generation is decode-bound: tokens arrive at rollout_tok_per_s.
    # Training replays the same tokens through forward+backward at
    # train_tok_per_s, which is far higher per token because training runs
    # big matrix multiplies at high utilization while decode is choked by
    # memory bandwidth. If you adapt this model, change these two rates;
    # the structure stays the same.
    gen_seconds = gen_tokens / rollout_tok_per_s
    train_seconds = gen_tokens / train_tok_per_s

    total = gen_seconds + train_seconds
    return {
        "gen_tokens": gen_tokens,
        "gen_seconds": round(gen_seconds, 1),
        "train_seconds": round(train_seconds, 1),
        # The headline number: fraction of the step spent generating.
        "gen_fraction": round(gen_seconds / total, 3),
    }

# Worked example: 128 prompts, G=8, 2000 tokens each -> ~2.05M tokens.
# A rollout pool doing 30K tok/s needs ~68 s. The trainer replays 2M
# tokens at 300K tok/s in ~7 s. Generation is ~91% of the step.
print(step_budget(
    n_prompts=128,
    group_size=8,
    avg_tokens=2000,
    rollout_tok_per_s=30_000,
    train_tok_per_s=300_000,
))
# {'gen_tokens': 2048000, 'gen_seconds': 68.3, 'train_seconds': 6.8,
#  'gen_fraction': 0.909}
```

::: walkthrough
1. **The product first.** `gen_tokens` multiplies batch x group size x length. All three knobs multiply, so doubling any one of them doubles generation time.
2. **Two rates, one ratio.** The step split is decided by the ratio of the two throughputs, not by either one alone. Decode at 30K tok/s vs train at 300K tok/s gives a 10:1 ratio, so generation takes ~91% of the step.
3. **What moves the needle.** Raising G from 8 to 16 doubles both phases equally, so the 91% split does not change. The split only moves if you change the throughput ratio: faster rollout engines (A.2) or more rollout GPUs (A.3).
4. **What breaks if you ignore it.** Buying more trainer GPUs while the rollout pool starves leaves trainers idle 91% of the time. Budget the bottleneck first.
:::

### A.1.4 Common misunderstanding

**"Training FLOPs dominate RL cost, so I should scale the trainer."** The FLOP count of the gradient step is indeed large, but FLOPs are not wall clock. Decode is bandwidth-bound: each generated token drags all 14 GB of a 7B model's weights through memory. The gradient step reuses those weights across thousands of tokens at once. Measure your own loop with the budget above before buying hardware: in online RL with long completions, generation is routinely 70-95% of step time, and the fix is rollout throughput, not trainer FLOPs.

### A.1.5 Visual: the step-time split

![Horizontal timeline of one online RL step: a long teal segment labeled generation taking about 90 percent of the bar, and a short orange segment labeled train step taking the rest](../scratch/appendix-07a/img/a1-timeline.webp)

*Figure A1.1. One online RL step at the worked-example settings: 128 prompts, G=8, 2,000 tokens each. Generation (teal) is ~91% of wall clock; the gradient update (orange) is the remainder. Generated for this volume.*

::: walkthrough
1. **Read the bar left to right as time.** The teal segment is phase 1 (generate 2.05M tokens at 30K tok/s). The orange sliver is phase 2 (train on those tokens at 300K tok/s).
2. **The ratio is the message.** Teal is ~13x longer than orange because decode throughput per token is ~10x lower than train throughput per token.
3. **Scaling G stretches teal.** Doubling completions per prompt doubles the teal segment and barely moves the orange one in relative terms; the split stays near 90/10.
4. **What breaks the picture.** If your rollouts are short (say 128 tokens), the teal segment shrinks and the fixed costs of the update (optimizer step, communication) start to matter. The budget code lets you check your own regime.
:::

### A.1.6 Lab pointers

::: lab Lab A1.1: Budget your own loop
Copy the `step_budget` function into a file. Plug in your setting: prompts per step, group size, mean completion length, and measured tok/s for your rollout setup and trainer. Find the generation fraction. Then answer: if you doubled rollout GPUs, how much does step time fall? If you doubled trainer GPUs instead, how much? Write both numbers down; the gap is the price of optimizing the wrong phase.
:::

::: lab Lab A1.2: Feel the bandwidth bound (RunPod-ready)
On one GPU, generate 256 tokens for a single sequence with a 7B model and time it. Then generate 256 tokens for each of 32 sequences in one batch and time it. Divide tokens by seconds in both cases. The per-sequence speed barely changes with batching at small batch sizes: that flat line is the memory-bandwidth ceiling from A.1.2.
:::

::: takeaway
- A rollout is one full generated trajectory; online RL generates fresh ones every step.
- Decoding is memory-bandwidth bound, training is compute bound: the same tokens cost ~10x more to generate than to train on.
- One step's clock splits ~90/10 toward generation at R1-style settings (128 prompts, G=8, 2K tokens).
- The split is set by the throughput ratio, so the fix for a slow loop is rollout throughput: faster engines (A.2), more rollout GPUs and async overlap (A.3).
:::

::: provenance
**Last verified: September 2026.** The decode-vs-train bottleneck analysis is standard systems knowledge (memory-bandwidth-bound autoregressive decoding vs compute-bound training). The 91% figure is illustrative, computed from the stated assumptions with the budget model above, not a measured claim about any specific system.
:::

## A.2 Inference engines inside the RL loop

### A.2.1 What a rollout worker is

A %%rollout worker%% is an inference server that holds the policy weights and does nothing but generate. The RL trainer sends it prompts; it returns completions. In modern stacks this server is vLLM or SGLang, the same engines that serve chatbots in production. They are used here for one reason: they generate tokens several times faster per GPU than a naive generation loop.

The loop looks like this: the trainer finishes a gradient step, broadcasts the new weights to the rollout workers, and the workers start generating the next batch of rollouts with the fresh policy. Generation and training alternate, and the workers exist so that generation runs at inference-engine speed instead of training-script speed.

### A.2.2 Why naive generation is slow

The obvious way to generate G completions for a batch of prompts is a loop over a standard generate call. Three things make it slow:

1. **Static batching.** The naive loop pads every sequence to the longest one and waits for the whole batch to finish. If one completion runs 4,000 tokens and the rest run 200, seven-eighths of the batch's GPU slots sit idle most of the time.
2. **KV cache fragmentation.** Each sequence reserves one long contiguous block of memory for its key-value cache. Real lengths vary, so the reserved blocks are half empty on average, and the wasted memory is memory that cannot hold more concurrent sequences.
3. **No prefix reuse.** In multi-turn rollouts (agents, long CoT with shared system prompts), the same prefix is re-processed for every turn. An engine with prefix caching computes it once.

Inference engines attack all three. That is the entire reason they belong in the RL loop.

### A.2.3 How it works under the hood

**Continuous batching** (also called iteration-level scheduling) is the core trick. Instead of forming a batch and waiting for every member to finish, the scheduler works one iteration at a time. Each iteration, finished sequences leave, newly arrived sequences join, and the freed slots are refilled immediately. There is no "wait for the longest sequence." The batch is a living thing, not a fixed roster.

**PagedAttention** (vLLM's KV cache design) fixes the fragmentation. The KV cache is split into fixed-size blocks (pages), like virtual memory in an OS. A sequence's cache is a list of block pointers, not one contiguous slab. Short sequences use few blocks, long ones use many, and there is no half-empty reserved slab. More concurrent sequences fit per GPU, which directly raises tokens per second.

**RadixAttention** (SGLang's design) adds prefix sharing on top: a radix tree of cached prefixes means common prompt prefixes (system prompts, few-shot examples, repeated tool definitions) are computed once and reused across requests. For agentic rollouts with long shared prefixes, this is a large win.

**Weight sync** closes the loop with training. After each gradient step, the trainer must get fresh weights into the workers. On a real cluster this is a broadcast over NCCL (the GPU collective library) or a checkpoint-engine transfer. In verl's rollout-server mode, a checkpoint engine moves weights from trainer to rollout replicas with a staged protocol: abort in-flight requests, transfer, verify version, resume. The workers then generate with the new policy. The sync is a brief pause in generation, which is one more reason A.3 decouples the two sides.

The simulation below shows the continuous-batching win with the same requests and the same per-iteration speed. The only difference is scheduling.

```python
# Static vs continuous batching, simulated with identical requests.
# Each request: (arrival iteration, total tokens). Each iteration the GPU
# processes a fixed token budget split across the running batch.
# Static batching locks the batch: once admitted, no new request joins
# until every member finishes, so a long request holds slots hostage.
# Continuous batching refills finished slots every iteration.
# Same work, same speed: scheduling is the only difference.

def simulate(requests, tok_per_iter, max_batch, continuous):
    pending = sorted(requests)   # not yet admitted, ordered by arrival
    active = []                  # [tokens_left, arrival] currently running
    t, i = 0, 0
    done = []                    # completion times (arrival -> finished)
    batch_open = True            # static batching closes the door mid-batch
    while i < len(pending) or active:
        # Admit arrivals while there is room (and the door is open).
        while (i < len(pending) and pending[i][0] <= t
               and len(active) < max_batch and batch_open):
            arrival, tokens = pending[i]
            active.append([tokens, arrival])
            i += 1
        if not active:
            t = pending[i][0]    # idle: jump to the next arrival
            continue
        # One iteration: split the token budget across the running batch.
        share = tok_per_iter / len(active)
        still = []
        for rem, arrival in active:
            rem -= share
            if rem <= 0:
                done.append(t + 1 - arrival)
            else:
                still.append([rem, arrival])
        active = still
        # Static batching: the door stays shut until the batch fully drains.
        # Continuous batching: the door never closes.
        if not continuous:
            batch_open = len(active) == 0
        t += 1
    return sum(done) / len(done), t

import random
random.seed(3)
# 4 very long requests arrive first, then 60 short ones: the classic
# head-of-line blocking scenario for static batching.
reqs = ([(0, 4000)] * 4
        + [(random.randint(1, 10), random.randint(150, 250))
           for _ in range(60)])

for name, cont in [("static    ", False), ("continuous", True)]:
    mean_completion, makespan = simulate(reqs, 512, 8, cont)
    print(f"{name}: mean completion {mean_completion:.0f} iters, "
          f"makespan {makespan}")
# static    : mean completion 42 iters, makespan 62
# continuous: mean completion 26 iters, makespan 59
```

::: walkthrough
1. **The scenario.** Four 4,000-token requests arrive at iteration 0, then 60 short ones (150-250 tokens) trickle in. This is head-of-line blocking: the shorts queue behind the longs.
2. **Static batching.** The first batch locks in the 4 longs plus 4 shorts. The remaining 56 shorts wait until that batch fully drains at the longest member's pace. Mean completion: 42 iterations.
3. **Continuous batching.** The moment a short request finishes, its slot refills with the next waiting short. The longs still take their time, but the shorts stream through instead of queuing. Mean completion: 26 iterations, 38% lower, with identical per-iteration speed.
4. **What this proves.** Scheduling alone, with zero hardware change, moves the mean by more than a third. On real engines the gap is larger because PagedAttention also packs more sequences per GPU.
5. **What breaks if you misread it.** Makespan barely moves (62 vs 59): the total work is the same. Continuous batching wins on latency and utilization, not on total FLOPs. If your metric is "time until the last rollout finishes," batching helps less than if your metric is throughput.
:::

### A.2.4 Common misunderstanding

**"vLLM is a chatbot server; my RL script's generate loop is fine."** The generate loop in a training script is typically static batching over a fixed batch with a contiguous KV cache: exactly the slow path this chapter describes. Production RL stacks (verl, OpenRLHF, TRL's online trainers) all generate through vLLM or SGLang, as a library or as rollout-server replicas. The reason: the throughput gap is several-fold per GPU. The chatbot framing is an accident of where the engines were first adopted. The scheduling math does not care what the tokens are for.

### A.2.5 Visual: static vs continuous batching

![Two GPU timeline rows: static batching with long idle gaps while waiting for the longest sequence, continuous batching densely packed with no gaps](../scratch/appendix-07a/img/a2-continuous.webp)

*Figure A2.1. The same requests under two schedulers. Top: static batching leaves slots idle while the batch waits for its longest member. Bottom: continuous batching refills finished slots every iteration, so the GPU stays full. Generated for this volume.*

::: walkthrough
1. **Top row is static.** Each fixed batch runs to its longest member; short members' slots sit empty (the pale gaps) until the batch boundary.
2. **Bottom row is continuous.** Finished sequences are replaced the next iteration; there are no batch boundaries and almost no gaps.
3. **Count the waste.** In the top row, roughly a third of the slots are idle at any moment. That idle time is pure loss: the GPU could have been generating.
4. **Connect to the code.** The simulation's 42-vs-26 iteration gap is this picture turned into numbers.
:::

### A.2.6 Lab pointers

::: lab Lab A2.1: Sweep the scheduler
Run the `simulate` function with your own request mix: change the long/short ratio, the batch size, and the arrival spread. Find the regime where continuous batching helps most (hint: high length variance) and the regime where it barely matters (hint: uniform lengths, no queue). Write down both.
:::

::: lab Lab A2.2: Measure a real engine (RunPod-ready)
Install vLLM on a GPU machine. Generate 256 completions for a fixed prompt set twice: once with vLLM's batched `generate`, once with a naive per-prompt loop. Record tokens per second for each. The ratio you measure is the reason rollout workers exist.
:::

::: takeaway
- A rollout worker is an inference server (vLLM/SGLang) that generates for the trainer at engine speed.
- Naive generation loses to static batching, KV fragmentation, and repeated prefixes.
- Continuous batching refills slots every iteration; PagedAttention pages the KV cache; RadixAttention shares prefixes.
- Scheduling alone cut mean completion 38% in the simulation; real engines add several-fold throughput per GPU on top.
:::

::: provenance
**Last verified: September 2026.** PagedAttention/continuous batching are vLLM's documented core designs; RadixAttention is SGLang's documented prefix-caching design. verl's rollout backends (vLLM, SGLang, TensorRT-LLM) are documented in the verl project (blog v0.7, rollout-elastic design doc). The 38% figure is from the simulation above with stated inputs, illustrative, not a vendor benchmark.
:::

## A.3 Async RL architectures and memory layout for rollout workers

### A.3.1 What async RL is

In the lockstep loop from A.1, generation and training take turns: the trainers sit idle while rollouts generate, then the rollout workers sit idle while gradients compute. %%Async RL%% breaks the turn-taking. Rollout workers generate continuously on their own GPUs. Trainers update continuously on theirs. The two sides meet only at %%weight sync%%, when the trainer publishes fresh weights to the fleet.

The price of decoupling is %%staleness%%: rollouts are generated by a slightly older policy than the one being trained. Async systems bound that staleness (a few versions behind, not fifty) and correct for it in the update. The reward is utilization: no side ever waits for the other, and one straggler with a 32K-token chain-of-thought no longer stalls the whole step.

The reference design is verl's %%HybridFlow%% (Hybrid-Controller) architecture. A single-controller orchestration layer treats the RL algorithm as a dataflow graph: schedule rollouts, score, dispatch training. Internally, each worker runs standard distributed programs (FSDP/Megatron for training, vLLM/SGLang/TensorRT-LLM for rollout). On top of that, verl supports a fully-async mode. Rollout workers sample continuously into a message queue. The trainer consumes mini-batches from the queue. Weight syncs are triggered on a schedule, with a staleness threshold that pauses generation if queued data gets too old. A lighter variant, the one-step-off policy, overlaps the two phases by one step: while the trainer runs iteration k, the fleet already generates iteration k+1's rollouts.

### A.3.2 Why it exists

Three forces push RL toward async:

1. **The 90/10 split from A.1.** If generation is 90% of the step, lockstep means trainers idle 90% of the time. Decoupling lets a small trainer pool serve a large rollout fleet, each sized to its own bottleneck.
2. **Stragglers.** Completion lengths vary wildly (a short refusal vs a 32K-token proof). In lockstep, the step waits for the longest rollout. In async, long rollouts just occupy one worker while the rest of the fleet keeps producing.
3. **Fault tolerance.** A dead rollout worker in lockstep halts training. In async with a supervisor heartbeat (verl's rollout-elastic design), the dead replica is pruned and replaced while training continues on the queue's buffered batches.

### A.3.3 How it works under the hood

**Placement.** Three options, one tradeoff (memory vs sync cost):

- **Colocated:** training and rollout share the same GPUs. Memory is tight (weights + optimizer + KV cache on one card); the engine must release inference memory before the training step and reclaim it after (TensorRT-LLM's rollout in verl uses explicit release/resume for this).
- **Disaggregated:** separate GPU pools for rollout and training. Each side is memory-comfortable; the cost is the weight broadcast between pools every sync.
- **Hybrid:** some GPUs colocated, overflow on dedicated rollout nodes. Common at large scale.

**Weight sync.** The trainer's weights must reach every rollout replica. At small scale this is a checkpoint copy; at scale it is a staged transfer over NCCL with versioning (abort in-flight requests, transfer, verify, resume), so a replica never generates with half-old weights.

**Staleness control.** Two mechanisms keep old-policy data honest:

- **Bounded lag:** the trainer syncs every N updates, and generation pauses if queued batches were sampled more than K versions ago (verl's staleness threshold).
- **Algorithmic correction:** the update accounts for the policy that generated the data. verl implements this as a three-policy setup: the %%rollout policy%% (behavior policy that generated the data), the %%proximal policy%% (the clipping anchor, computed fresh each epoch), and the %%current policy%% (being optimized). PPO-style clipping against the proximal policy keeps updates safe even when the data came from a slightly older rollout policy.

The asyncio sketch below is the architecture in miniature: producers and a consumer joined by a bounded queue, rendezvousing only at weight sync. The sleeps stand in for GPU work; the structure is the real thing.

```python
# Async RL in miniature: rollout producers and a trainer consumer joined
# by a bounded queue. Neither side waits for the other; they rendezvous
# only at weight sync. Runnable anywhere: the sleeps stand in for GPU
# work, so run it and watch the overlap in the timestamps.

import asyncio
import time

t0 = time.time()

def now():
    # Elapsed seconds: makes the producer/consumer overlap visible.
    return round(time.time() - t0, 1)

async def rollout_worker(queue, worker_id, n_batches):
    # One rollout worker: generate a batch, stamp it with the weight
    # version it used, push it on the queue, repeat. It never blocks on
    # the trainer. If the queue fills, put() waits: that is backpressure
    # (a staleness guard), not lockstep.
    version = 0
    for b in range(n_batches):
        await asyncio.sleep(0.5)   # stands in for generating G completions
        await queue.put((worker_id, b, version))
        print(f"{now()}s  worker {worker_id}: pushed batch {b} "
              f"(weights v{version})")

async def trainer(queue, n_batches, sync_every):
    # The trainer: pull batches, update, and every sync_every batches
    # publish new weights. Here publishing is a version bump; on a real
    # cluster it is a staged NCCL broadcast to the rollout fleet.
    version = 0
    for b in range(n_batches):
        worker_id, batch_id, used_version = await queue.get()
        staleness = version - used_version   # how old is this data?
        await asyncio.sleep(0.3)             # stands in for the grad step
        print(f"{now()}s  trainer: batch {batch_id} from worker "
              f"{worker_id}, staleness {staleness}")
        if (b + 1) % sync_every == 0:
            version += 1
            print(f"{now()}s  trainer: published weights v{version}")
        queue.task_done()

async def main():
    # Bounded queue: the bound IS the staleness control. Unbounded would
    # let workers race arbitrarily far ahead of the trainer.
    queue = asyncio.Queue(maxsize=4)
    await asyncio.gather(
        rollout_worker(queue, 0, 6),
        rollout_worker(queue, 1, 6),
        trainer(queue, 12, sync_every=4),
    )

asyncio.run(main())
```

::: walkthrough
1. **Two producers, one consumer.** Workers 0 and 1 each push 6 batches; the trainer consumes 12. They run concurrently: while the trainer chews batch 0, workers are already generating batches 2 and 3.
2. **The version stamp.** Each batch carries the weight version used to generate it. The trainer prints staleness = current version minus batch version. Watch it stay at 0 or 1: data is never ancient.
3. **The sync point.** Every 4 batches the trainer publishes v1, v2, v3. In a real system the workers would pick up the new weights here; the sketch keeps workers on v0 to keep the code short, which is why staleness grows late in the run. That growth is exactly what the staleness threshold guards against in production.
4. **The queue bound.** `maxsize=4` means workers can run at most 4 batches ahead. Remove the bound and fast workers race ahead: unbounded staleness, unstable training. The bound is a one-line staleness controller.
5. **What breaks if you lockstep it.** Replace the queue with a direct handoff (worker generates, then waits for the trainer to finish before generating again) and total runtime becomes the sum of both phases instead of their overlap. That sum is the tax async RL refuses to pay.
:::

### A.3.4 Memory layout of a rollout worker

A rollout worker never trains, so its GPU carries no gradients and no optimizer states. Its memory is three items:

1. **Weights**, read-only: params x bytes per param.
2. **KV cache** for the running batch: 2 (K and V) x layers x KV heads x head dim x bytes per param x batch x seq len. Note KV heads, not query heads: GQA models cache far less per token, which is why they serve longer contexts per GPU.
3. **Fixed overhead**: CUDA context, engine workspace, fragmentation slack. Budget ~2 GB.

Compare with the trainer side, which carries weights + gradients + optimizer states (Adam in fp32: 8 bytes per param on top of the 2-byte weights and 2-byte grads, so ~6x the weight memory before activations). This asymmetry is why disaggregation is attractive: the rollout pool is memory-light per GPU and scales with batch x length, while the trainer pool is memory-heavy and scales with model size.

```python
# GPU memory layout of one rollout worker: weights + KV cache + overhead.
# No gradients, no optimizer states: the worker never trains. Change the
# model shape or the batch geometry and watch which term dominates.

def rollout_worker_memory_gb(
    params_b,          # model size in billions of parameters
    bytes_per_param,   # 2 for bf16/fp16, 0.5 for 4-bit
    n_layers,          # transformer layers
    n_kv_heads,        # key/value heads (GQA-aware: fewer than query heads)
    head_dim,          # dimensions per head
    batch,             # concurrent sequences in the running batch
    seq_len,           # tokens per sequence (prompt + generated so far)
    overhead_gb=2.0,   # CUDA context, engine workspace, slack
):
    # Weights: the full model, read-only during rollouts.
    weights_gb = params_b * bytes_per_param   # billions x bytes = GB

    # KV cache per token: 2 (K,V) x layers x kv_heads x head_dim elements,
    # each bytes_per_param wide. GQA shows up here directly: halving KV
    # heads halves this term, which is why GQA models pack longer contexts
    # per GPU. If you change the attention type, change this line first.
    kv_per_token_gb = 2 * n_layers * n_kv_heads * head_dim * bytes_per_param / 1e9
    # The cache must cover the worst case the scheduler admits:
    # batch x sequence length.
    kv_gb = kv_per_token_gb * batch * seq_len

    total = weights_gb + kv_gb + overhead_gb
    return {
        "weights_gb": round(weights_gb, 1),
        "kv_cache_gb": round(kv_gb, 1),
        "overhead_gb": overhead_gb,
        "total_gb": round(total, 1),
    }

# 7B bf16, 32 layers, 8 KV heads, head dim 128, batch 32 x 4096 tokens:
# weights 14.0 GB, KV per token 128 KB, cache 17.2 GB, total 33.2 GB.
# Fits a 40 GB card with room to spare.
print(rollout_worker_memory_gb(7, 2, 32, 8, 128, 32, 4096))
# {'weights_gb': 14.0, 'kv_cache_gb': 17.2, 'overhead_gb': 2.0,
#  'total_gb': 33.2}

# 70B bf16, 80 layers, 8 KV heads, batch 8 x 4096: weights 140 GB alone
# exceed any single GPU, so the worker needs tensor parallelism across
# at least 2x80 GB cards before the KV cache is even counted.
print(rollout_worker_memory_gb(70, 2, 80, 8, 128, 8, 4096))
# {'weights_gb': 140.0, 'kv_cache_gb': 10.7, 'overhead_gb': 2.0,
#  'total_gb': 152.7}
```

The 7B worker's memory as a bar (from the numbers above, 33.2 GB total):

```
0 GB          14 GB                       31.2 GB  33.2 GB
|--------------|------------------------------|-----|
  weights 14.0    KV cache 17.2             overhead 2.0
  (42%)           (52%)                     (6%)
```

::: walkthrough
1. **Read the bar left to right as gigabytes.** Weights are fixed by the model; the KV cache is set by your batch geometry; overhead is roughly constant.
2. **The KV cache is the moving part.** At batch 32 x 4096 it is the largest term (52%). Halve the batch and it halves; double the sequence length and it doubles. This is the term the scheduler manages.
3. **GQA is a memory feature.** The 8 KV heads (not 32 query heads) are why per-token cache is 128 KB and not 512 KB. An MHA model with the same shape would need ~4x the cache and would not fit this batch on 40 GB.
4. **The 70B line is the scaling lesson.** Weights alone (140 GB) exceed one GPU, so rollout workers for 70B need tensor parallelism first; only then does the KV term matter. Placement follows the weights.
:::

### A.3.5 Common misunderstanding

**"Async means training on stale garbage, so lockstep is safer."** Staleness in a well-built async system is bounded to a few versions and explicitly corrected. The update clips against the proximal policy and accounts for the rollout policy that generated the data (verl's three-policy setup). What lockstep buys in freshness it pays in utilization: trainers idle through every generation phase, and one straggler stalls everything. The production consensus has moved. Bound the staleness, correct for it, and keep both sides busy.

### A.3.6 Visual: the decoupled architecture

![Decoupled async RL: a rollout fleet generating continuously on the left, a trainer on the right, a queue buffer between them, and a dashed weight-sync arrow back to the fleet](../scratch/appendix-07a/img/a3-async.webp)

*Figure A3.1. Async RL, decoupled. The rollout fleet generates continuously into a bounded queue; the trainer consumes batches and publishes new weights on a schedule. No side waits for the other. Generated for this volume.*

::: walkthrough
1. **Left: the rollout fleet.** Replicas of the inference engine (vLLM/SGLang) generate nonstop. Each batch is stamped with its weight version.
2. **Middle: the bounded queue.** This is the staleness controller from the code sketch. Batches wait here; if the queue fills, workers pause instead of racing ahead.
3. **Right: the trainer.** Pulls mini-batches, runs gradient steps, and every N steps publishes weights back to the fleet (dashed arrow).
4. **The failure case it removes.** A 32K-token straggler occupies one worker while the rest of the fleet keeps the queue fed. In lockstep that straggler would hold the entire step hostage.
:::

### A.3.7 Lab pointers

::: lab Lab A3.1: Run the async sketch
Run the asyncio script and watch the timestamps. Then change `maxsize` from 4 to 1 and to 100, and note how total runtime and maximum staleness change. Write one sentence explaining why neither extreme is what you want.
:::

::: lab Lab A3.2: Size a rollout fleet
Use `rollout_worker_memory_gb` for a 13B model (40 layers, bf16) at batch 16 x 8192 tokens. Does it fit a 40 GB card? An 80 GB card? Now find the largest batch that fits 80 GB. That number is your per-GPU rollout capacity, and the fleet size follows from your tokens-per-step target in A.1.
:::

::: takeaway
- Async RL decouples rollout workers from trainers; they rendezvous only at weight sync.
- verl's HybridFlow: single-controller dataflow orchestration over SPMD workers (FSDP/Megatron training, vLLM/SGLang/TRT-LLM rollout); fully-async and one-step-off modes.
- Staleness is bounded (queue limits, sync schedule) and corrected (proximal clipping, three-policy accounting), not wished away.
- A rollout worker carries weights + KV cache + overhead only: 33 GB for the worked 7B example; the trainer side carries ~6x weight memory in gradients and optimizer states.
:::

::: provenance
**Last verified: September 2026.** verl's Hybrid-Controller/HybridFlow architecture, rollout engines (vLLM/SGLang/TensorRT-LLM), fully-async rollouter with message queue, one-step-off policy, staleness threshold, and the three-policy (rollout/proximal/current) framework are documented in the verl project (docs/blog/v0.7.md, rollout_elastic/DESIGN.md). Memory formulas are standard transformer accounting; the worked numbers are computed from stated shapes, illustrative.
:::
