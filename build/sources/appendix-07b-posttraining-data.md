---
title: Volume 7 Appendix B - Post-training Data and Interpretability
eyebrow: Volume 7 · Appendix B - Post-training Data and Interpretability
---

# Appendix B: Post-training Data and Interpretability

Volume 7 taught the training objectives: SFT loss, reward models, PPO, DPO, GRPO. This appendix covers the two things those objectives consume and the one thing they cannot tell you.

- **Unit 1 (B.1-B.4): post-training data pipelines.** Where SFT and preference data come from, how synthetic data is made and filtered, how verifiers turn outputs into rewards, and how to collect human judgments without fooling yourself.
- **Unit 2 (B.5-B.8): mechanistic interpretability basics.** What it means to open the model, sparse autoencoders, probing versus steering, and the honest limits.
- **Patch chapter (B.9): LoRA and QLoRA.** Efficient fine-tuning with full GPU-memory accounting drills. Volume 7's main chapters do not cover parameter-efficient methods; this patch closes the gap.

Prerequisites, assumed cold: SFT and the RLHF pipeline from Volume 7, basic probability from Volume 1. No prior exposure to interpretability or human-data operations is assumed.

## Unit 1: Post-training data pipelines

## B.1 Synthetic data generation pipelines

### B.1.1 What synthetic data is

%%Synthetic data%% is training data a model writes instead of a human: prompts, responses, step-by-step rationales, code with tests, preference pairs. The pipeline is a loop: generate candidates, filter them with a checker, keep the good ones, and sometimes feed them back to improve the generator itself.

This is not a exotic trick anymore. It is how the field makes reasoning data at scale: human experts cannot write millions of step-by-step math solutions, but a model can draft them and a verifier can check the answers.

### B.1.2 Why it exists

Three reasons, in order of importance:

1. **Humans are slow and expensive.** A careful math rationale takes an expert many minutes. A model writes thousands per hour.
2. **Verifiable domains filter themselves.** In math and code, correctness is checkable: the answer matches, or the tests pass. That turns generation into a precision instrument: keep only what is right.
3. **Bootstrapping works.** A model trained on its own correct outputs gets better at producing correct outputs, which makes the next round of synthetic data better. The loop compounds.

The catch, covered in B.1.5, is that the loop can also compound the model's blind spots. Filtering is the whole game.

### B.1.3 How it works under the hood

**The STaR loop.** STaR (Self-Taught Reasoner, Zelikman et al., 2022) is the canonical bootstrapping recipe, and every later variant is a variation on it:

1. **Generate.** Prompt the model with a few rationale examples and ask it to solve many questions, writing its reasoning.
2. **Filter.** Keep the rationales that reach the correct answer. Discard the rest.
3. **Rationalize.** For the failures, try again with the correct answer given as a hint, asking for the reasoning that leads there. Then strip the hint: the training example is (question, rationale), not (question, hint, rationale).
4. **Fine-tune.** Train on all kept rationales.
5. **Repeat.** The improved model generates better rationales next round.

The paper's headline result: this loop let a 6B model reach CommonsenseQA accuracy comparable to fine-tuning a 30x larger model. The model teaches itself reasoning by keeping its own correct work.

**Rejection sampling (RFT).** A simpler, one-shot cousin (Yuan et al., 2023): fine-tune a base model on the seed data, sample k solutions per problem, keep the correct ones, augment the dataset, fine-tune once more. No iteration. Cheaper, less compounding.

**Distillation at scale.** The industrial version: a strong teacher generates a large, carefully filtered dataset, and smaller models train on it. DeepSeek's R1 report describes distilling 800K samples (600K reasoning, 200K non-reasoning) from R1 into smaller dense models, a major driver of the R1-Distill family's strength. Same pattern as STaR, minus the loop, at massive scale.

**Quality gates.** Every serious pipeline applies four, in order.
- **Dedupe:** near-duplicate solutions teach repetition.
- **Difficulty filter:** drop problems the generator solves trivially every time; they add no signal.
- **Decontamination:** remove anything overlapping eval sets, or your "gains" are leakage.
- **Format checks:** answers must be parseable, since the verifier has to extract them.

The code below is the STaR loop with a toy noisy generator so it runs anywhere. The generator is a stand-in; the loop structure is the real thing.

```python
# STaR (Self-Taught Reasoner) in miniature: the synthetic-data loop.
# generate  -> the model writes rationales for questions
# filter    -> keep rationales that reach the right answer
# rationalize -> retry failures with the answer given as a hint, then
#                strip the hint before training (the model must learn the
#                reasoning, not the hint)
# fine-tune -> train on the kept rationales; repeat
# The generator here is a noisy toy so the loop runs anywhere. Swap in a
# real model and a real checker for the honest version.

import random

random.seed(7)

def noisy_multiply(a, b, hint=None):
    # Toy "model": a calculator that sometimes slips a digit. With a hint
    # (the true answer), its work is clean: that is rationalization.
    true = a * b
    if hint is not None:
        shown = true
    else:
        # 50% of the time it slips: the forward pass is unreliable,
        # which is exactly the regime where STaR helps.
        shown = true + random.choice([0, 0, 0, 10, -10, 100])
    rationale = f"{a}x{b}: partial products sum to {shown}."
    return rationale, shown

def star_iteration(n_questions):
    # One STaR iteration. Returns (forward_kept, rationalized_kept).
    forward_kept, rationalized_kept = 0, 0
    for _ in range(n_questions):
        a, b = random.randint(10, 99), random.randint(10, 99)
        rationale, answer = noisy_multiply(a, b)
        if answer == a * b:
            forward_kept += 1       # the model solved it alone: keep
        else:
            # Backward rationalization: give the answer, ask for the work.
            rationale2, answer2 = noisy_multiply(a, b, hint=a * b)
            if answer2 == a * b:
                rationalized_kept += 1   # new capability, banked as data
    return forward_kept, rationalized_kept

for it in range(3):
    fwd, rat = star_iteration(200)
    print(f"iter {it}: forward-kept {fwd}, rationalized-kept {rat}, "
          f"banked {fwd + rat}")
# iter 0: forward-kept 100, rationalized-kept 100, banked 200
# iter 1: forward-kept 119, rationalized-kept 81, banked 200
# iter 2: forward-kept 107, rationalized-kept 93, banked 200
```

::: walkthrough
1. **The toy model.** `noisy_multiply` slips a digit half the time without a hint, never with one. That gap between unaided and hinted performance is the raw material STaR mines.
2. **Forward keeps.** About 100 of 200 questions are solved unaided each iteration and banked directly. These are the model's existing competence, converted into training data.
3. **Rationalized keeps.** The other ~100 are solved only with the answer given, and their clean rationales are banked too. This is the interesting half: traces the model could not produce on its own.
4. **What the toy does not show.** The toy generator never learns, so the forward rate stays flat. In the real loop, step 4 (fine-tune on banked traces) is what lifts the next iteration's forward rate. The paper's compounding comes from that training step, which the demo omits on purpose to keep the data mechanics visible.
5. **What breaks if you skip rationalization.** Keeping only forward successes banks what the model already knows. Rationalization is how the dataset gains reasoning the model could not previously produce: the frontier of the data moves outward each round.
:::

### B.1.4 Worked numbers: the cost of a synthetic batch

Say you want 100K verified math rationales. Recipe: 50K seed problems, sample k=16 each at ~500 tokens per rationale.

- Generation volume: 50K x 16 x 500 = 400M tokens.
- Keep rate 25% (only a quarter reach correct answers): 100K rationales banked, 100M tokens of training data.
- The other 300M generated tokens are discarded compute: the price of filtering. A better generator (higher keep rate) is worth more than a cheaper generator here, because discarded tokens dominate the bill.

This is why keep rate is the metric synthetic-data engineers watch, and why the STaR loop's compounding matters: each iteration raises the keep rate of the next.

### B.1.5 Common misunderstanding

**"Synthetic data is free and unlimited, so just generate more."** Two failure modes. First, %%model collapse%%: training only on your own outputs narrows the distribution round after round; the model forgets the tails of the real data. Keep real human data in the mix and filter hard. Second, the generator's blind spots become the dataset's blind spots: if the model systematically mishandles a problem type, no amount of its own sampling fixes it, because the filter only keeps what the model can already verify. Synthetic data amplifies capability; it does not invent it from nothing. The rationalization step is the partial exception, and even it needs ground-truth answers from somewhere real.

### B.1.6 Visual: the STaR loop

![Circular four-node loop diagram: generate rationales, keep correct answers, fine-tune, repeat, with arrows forming a cycle](../scratch/appendix-07b/img/b1-star-loop.webp)

*Figure B1.1. The STaR loop. Each cycle banks verified rationales and the fine-tuning step makes the next cycle's generator stronger, which raises the keep rate. Generated for this volume.*

::: walkthrough
1. **Top: generate.** The current model writes rationales for a batch of questions, few-shot prompted.
2. **Right: filter.** A checker (answer match, tests) keeps only correct traces. Failures go to rationalization: retry with the answer as a hint.
3. **Bottom: fine-tune.** The banked traces become training data. This is the step that compounds.
4. **Left: repeat.** The improved model re-enters at the top with a higher keep rate. The loop stops when the keep rate plateaus.
:::

### B.1.7 Lab pointers

::: lab Lab B1.1: Run the toy loop
Run the STaR script. Change the slip probability (the `random.choice` list) and watch the forward/rationalized split move. Then add a fake "learning" step: after each iteration, shrink the slip list slightly, modeling fine-tuning. Watch the forward rate climb: that is the compounding the paper reports.
:::

::: lab Lab B1.2: Price a synthetic batch (paper exercise)
Pick a target: 50K verified code solutions. Assume k=16 samples per problem, 800 tokens per sample, keep rate 20%. Compute total generated tokens, banked tokens, and the discard ratio. Then compute how the bill changes if a better generator lifts the keep rate to 40%. Write both numbers.
:::

::: takeaway
- Synthetic data = model-generated, checker-filtered training data; the loop is generate, filter, rationalize, fine-tune, repeat.
- STaR bootstraps reasoning by banking its own correct traces, including rationalized ones it could not produce unaided; reported result: 6B model matching a 30x larger fine-tuned model on CommonsenseQA.
- Rejection sampling (RFT) is the one-shot version: sample k, keep correct, fine-tune once.
- Keep rate is the economic metric; discarded tokens dominate the bill.
- Model collapse and blind-spot amplification are the failure modes: keep real data in the mix, filter hard, decontaminate against evals.
:::

::: provenance
**Last verified: September 2026.** STaR (Zelikman et al., 2022, arXiv 2203.14465): loop description and the 30x-larger-model CommonsenseQA comparison are as reported in the paper. RFT (Yuan et al., 2023) one-shot description as reported. DeepSeek-R1 800K-sample distillation (600K reasoning + 200K non-reasoning) as reported in the R1 paper. Worked cost numbers are illustrative, computed from stated assumptions.
:::

## B.2 Verifier and reward pipelines

### B.2.1 What a verifier is

A %%verifier%% is a program that checks a model output and returns a verdict: right or wrong, tests passed or failed, proof valid or not. A %%reward pipeline%% is the code path from a raw rollout to the scalar reward the RL update consumes: normalize the output, extract the answer, run the checks, aggregate to a number, log everything.

Verifiers are what make RLVR (RL with verifiable rewards, Volume 7 Chapter 6) possible. Where RLHF needs a learned reward model trained on human preferences, RLVR needs a correct program. The verifier is the reward model, written in code instead of learned from labels.

### B.2.2 Why it exists

A reward function you cannot trust teaches the wrong thing, and the model will find the gap. This is where %%reward hacking%% is born (Volume 7 Chapter 8): not in the RL algorithm, but in the plumbing between generation and reward. A verifier that counts a test as passed when the solution crashed. A math checker that accepts "42" inside a paragraph of nonsense. An extraction regex that grabs the wrong number. Each of these becomes the behavior the model learns. The model optimizes the reward you actually compute, not the reward you meant.

So the verifier pipeline is built like infrastructure, not like a script: staged, logged, tested against adversarial outputs.

### B.2.3 How it works under the hood

A production verifier pipeline has five stages:

1. **Normalize.** Strip formatting: remove markdown fences, trailing whitespace, LaTeX wrappers. Model outputs are messy; the checker must see through the mess deterministically.
2. **Extract.** Pull the answer out: the number in `\boxed{}`, the code block, the JSON field. Extraction is where most verifier bugs live, because there are a hundred ways to write "42".
3. **Check.** Run the actual verification: unit tests in a sandbox, symbolic or numeric answer comparison, proof-checker validation. This stage must be deterministic: same output, same verdict, every time.
4. **Aggregate.** Combine check results into one scalar: fraction of tests passed, binary correct/incorrect, partial credit rules. Keep the aggregation simple enough to audit.
5. **Audit log.** Record (raw output, extracted answer, per-check results, final reward) for every rollout. When reward hacking appears, this log is the crime scene.

**Sandboxing** is non-negotiable for code: generated code runs in a subprocess with a timeout and no network, because generated code is untrusted input. A missing timeout is how one infinite loop hangs your entire rollout fleet.

**V-STaR** (Hosseini et al., 2024) adds a learned layer: train a verifier on both correct and incorrect generations, so at test time you can rank candidates by the verifier's score instead of relying on hand-written checks alone. Hand-written verifiers for training signal, learned verifiers for selection: they compose.

The code below is a complete verifier pipeline for code tasks, small enough to read in one sitting.

```python
# A verifier pipeline for code tasks: normalize, execute, score, audit.
# The verifier IS the reward function, so its bugs become the model's
# bugs. Every stage logs its decision: when the model starts gaming the
# reward, this log is where you find out how.

import subprocess
import tempfile
import os

def verify_solution(code, cases, timeout_s=10):
    # code: candidate source defining solve(...).
    # cases: list of (args_tuple, expected_value).
    # Returns (reward, audit_log). Reward is the fraction of cases passed.
    # Stages: (1) the code is written to disk as-is (normalization would
    # strip fences here in a fuller pipeline); (2) a harness calls
    # solve() per case in a subprocess; (3) per-case PASS/FAIL lines are
    # parsed; (4) the fraction passed is the scalar reward; (5) every
    # line is returned as the audit log.
    harness = "import sol\n"
    for i, (args, expected) in enumerate(cases):
        # One guarded call per case: an exception in case 3 must not kill
        # cases 4 and 5. repr() keeps values unambiguous in the log.
        harness += (
            f"try:\n"
            f"    got = sol.solve(*{args!r})\n"
            f"    print('PASS' if got == {expected!r} else 'FAIL', {i}, "
            f"          repr(got))\n"
            f"except Exception as e:\n"
            f"    print('ERROR', {i}, type(e).__name__)\n"
        )
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "sol.py"), "w").write(code)
        open(os.path.join(d, "run.py"), "w").write(harness)
        try:
            r = subprocess.run(
                ["python3", "run.py"],
                capture_output=True, text=True,
                timeout=timeout_s, cwd=d)
        except subprocess.TimeoutExpired:
            # A hang is a zero, not a crash: the pipeline must survive it.
            # Without this branch, one infinite loop stalls the rollout
            # fleet. Timeouts are load-bearing infrastructure.
            return 0.0, ["TIMEOUT"]
        log = r.stdout.strip().splitlines()
        passed = sum(1 for line in log if line.startswith("PASS"))
        reward = passed / len(cases)
        return reward, log

# Demo: sum of squares, one correct and one off-by-one candidate.
good = "def solve(n):\n    return sum(i*i for i in range(1, n+1))\n"
bad = "def solve(n):\n    return sum(i*i for i in range(n))\n"
cases = [((3,), 14), ((1,), 1), ((5,), 55)]

print(verify_solution(good, cases))
# (1.0, ['PASS 0 14', 'PASS 1 1', 'PASS 2 55'])
print(verify_solution(bad, cases))
# (0.0, ['FAIL 0 5', 'FAIL 1 0', 'FAIL 2 30'])
```

::: walkthrough
1. **The harness.** For each case, a try/except block calls `solve()` and prints one verdict line. The per-case guard means one crashing case cannot hide the others: partial credit stays honest.
2. **The sandbox.** The candidate runs in a child process with a 10-second timeout, in a temp directory. The timeout branch returns reward 0.0 with a TIMEOUT marker: hangs are data, not disasters.
3. **The aggregation.** Reward = fraction passed. Simple, auditable, no hidden weighting. If you want partial credit or difficulty weighting, it goes here, in the open.
4. **The audit log.** Every verdict line comes back to the caller. In a real pipeline these lines are written to disk per rollout: raw output, extracted answer, per-case verdicts, final reward. When the model learns to print "PASS" instead of solving (it happens), the log shows the raw output diverging from the verdicts.
5. **The demo's lesson.** The off-by-one candidate fails all three cases visibly: `range(n)` vs `range(1, n+1)`. A verifier that only checked n=1 would have passed it (both give... no: bad gives 0 for n=1, good gives 1). The point stands: case coverage is the verifier's real quality metric.
:::

### B.2.4 Worked numbers: verifier throughput

500 code problems, 8 samples each, 5 test cases per sample: 20,000 program executions per RL step. At 0.2 s per execution (small tests, warm interpreter), that is ~67 minutes single-threaded: completely unacceptable inside a training loop. Verifiers parallelize trivially (each execution is independent), so a 64-worker pool brings it to ~63 seconds. Budget verifier parallelism the same way you budget rollout GPUs: it is part of the generation phase's wall clock from A.1. And cap the timeout aggressively: one 10-second hang per thousand executions adds ~3 minutes per step at this scale if unparallelized.

### B.2.5 Common misunderstanding

**"A verifier just runs the tests."** Running the tests is the easy third of the job. The first third is normalization and extraction: model outputs arrive wrapped in markdown, apologies, and alternative phrasings, and the extractor must deterministically find the answer candidate. The last third is the audit log: without per-rollout records of (output, extracted, verdict), you cannot diagnose reward hacking when it starts. Teams that treat the verifier as "just run pytest" discover the other two thirds the hard way, usually when the model's reward curve climbs while its actual capability does not.

### B.2.6 Visual: the verifier pipeline

![Pipeline diagram: code output box into unit tests box into scalar reward gauge into training signal box](../scratch/appendix-07b/img/b2-verifier.webp)

*Figure B2.1. The five-stage verifier pipeline: normalize, extract, check, aggregate, audit. The reward that reaches the trainer has passed through all five; the audit log records every stage's decision. Generated for this volume.*

::: walkthrough
1. **Left: raw output.** The model's completion, messy and untrusted.
2. **Normalize + extract.** Formatting is stripped and the answer candidate is pulled out deterministically. Most verifier bugs live in these two boxes.
3. **Check.** The sandboxed execution: unit tests, answer comparison, proof checking. Deterministic: same input, same verdict.
4. **Aggregate.** Per-check results become one scalar reward. Simple rules, auditable.
5. **Right: two outputs.** The scalar goes to the trainer; the full audit trail goes to disk. When the reward curve and the capability curve diverge, the audit trail is where you look first.
:::

### B.2.7 Lab pointers

::: lab Lab B2.1: Break your own verifier
Take `verify_solution` and try to game it: write a candidate that prints PASS-like text, one that hangs, one that reads the test file. Watch how the pipeline handles each. Then harden one stage (e.g., run with a 2-second timeout, or compare against hidden cases not in the harness). Write down which attack each hardening blocks.
:::

::: lab Lab B2.2: Build a math verifier
Write a verifier for arithmetic answers: normalize (strip commas, whitespace, LaTeX `\boxed{}`), extract the final number with a regex, compare numerically with tolerance. Test it on ten tricky strings ("1,000", "1000.0", "\\boxed{1000}", "about 1000"). Count how many your first version gets wrong: that count is the extraction bug budget.
:::

::: takeaway
- A verifier is a program that checks outputs; the reward pipeline is normalize, extract, check, aggregate, audit.
- The verifier is the reward function: its bugs become the model's learned behaviors (reward hacking starts here).
- Sandbox untrusted code with timeouts; log every stage per rollout.
- Verifier executions parallelize trivially: budget them as part of generation wall clock.
- V-STaR adds a learned verifier trained on correct and incorrect generations for ranking candidates.
:::

::: provenance
**Last verified: September 2026.** V-STaR (Hosseini et al., COLM 2024): training a verifier on correct and incorrect generations, as described in the paper. The pipeline stages and sandboxing practices are standard production practice, reconstructed from first principles. Worked throughput numbers are illustrative, computed from stated assumptions.
:::

## B.3 SFT data mixtures and rejection-sampling recipes

### B.3.1 What a data mixture is

A %%data mixture%% is the weighted blend of sources that makes up supervised fine-tuning: X% code, Y% math, Z% chat, and so on. The mixture is not a vibe; it is a sampling distribution. Every training batch is drawn from it, so the mixture decides what the model practices, in what proportion, for the whole run.

%%Rejection sampling%% is the filter that upgrades a source: generate k candidates per prompt, keep the ones that pass a checker or judge, discard the rest. It trades compute for quality inside a fixed source.

### B.3.2 Why the mixture matters more than its size

At fixed training compute, tokens are a budget. Every token of low-quality chat you include is a token of high-quality code you exclude. The mixture sets three things:

1. **Skill balance.** The model's competence follows the mixture weights, with a lag. Underweight math and math scores sag two weeks later.
2. **Repetition risk.** Small sources get repeated many times per run. A 5B-token source at 20% weight in a 100B-token run is seen 4 times. Repetition past a few epochs teaches memorization of the source, not the skill.
3. **Behavioral tone.** The chat portion sets refusal style, formatting habits, and personality. It is small in tokens and large in influence, because it is where the model learns how to talk.

The QLoRA paper's finding is the extreme case of the quality argument: fine-tuning on a small high-quality dataset beat larger noisier ones. Tokens are not interchangeable; the mixture is a quality-weighted portfolio.

### B.3.3 How it works under the hood

**Mixture math.** Sources have sizes (tokens available) and weights (target fractions). Per training run of T tokens, source i contributes weight_i x T tokens, which is (weight_i x T / size_i) epochs of that source. The planner below computes exactly this, and flags repetition.

**Temperature and upsampling.** Small, precious sources (expert-written math, carefully reviewed code) are routinely upsampled: given weight beyond their size share, accepting repetition as the price. The rule of thumb is to watch epochs per source and cap the worst offenders, either by lowering their weight or by finding more data.

**The rejection-sampling recipe**, in the order practitioners run it:

1. **Generate k per prompt** (k=8 to 64; temperature moderate-high for diversity).
2. **Filter** with the verifier from B.2 (code, math) or a model judge (open-ended).
3. **Keep at most one or two per prompt.** Keeping all k passes of an easy prompt floods the data with near-duplicates of the same reasoning.
4. **Dedupe** across prompts: exact-match dedupe first, then near-dedupe on normalized text or embeddings.
5. **Re-balance** by difficulty: keep a higher fraction of hard-prompt successes, because easy successes are plentiful and teach less.

```python
# SFT mixture planner: from source sizes and target weights to a training
# plan. The mix is a promise about what the model will practice. This
# planner turns "50% code, 30% math, 20% chat" into tokens per source,
# epochs per source, and total steps, so you can see when a small source
# will be repeated 4x (and start memorizing itself).

def plan_mixture(sources, total_tokens, seq_len, batch_seqs):
    # sources: list of (name, size_tokens, weight). Weights need not sum
    # to 1; they are normalized here, so you can write 5, 3, 2.
    total_w = sum(w for _, _, w in sources)
    plan = []
    for name, size, w in sources:
        frac = w / total_w
        tokens = frac * total_tokens   # tokens drawn from this source
        # Epochs > 2-3 deserve a second look: repetition teaches the
        # source's surface patterns, not the underlying skill.
        epochs = tokens / size
        plan.append((name, frac, tokens, epochs))
    steps = total_tokens / (seq_len * batch_seqs)
    return plan, steps

sources = [
    ("code", 40e9, 0.5),   # 40B tokens available, target 50%
    ("math", 10e9, 0.3),   # 10B tokens available, target 30%
    ("chat",  5e9, 0.2),   #  5B tokens available, target 20%
]
plan, steps = plan_mixture(sources, total_tokens=100e9,
                           seq_len=2048, batch_seqs=512)
for name, frac, tokens, epochs in plan:
    print(f"{name:6s} {frac:4.0%}  {tokens/1e9:5.1f}B tokens  "
          f"{epochs:4.1f} epochs")
print(f"total steps: {steps:,.0f}")
# code    50%   50.0B tokens   1.2 epochs
# math    30%   30.0B tokens   3.0 epochs
# chat    20%   20.0B tokens   4.0 epochs
# total steps: 95,367

# Rejection-sampling recipe: generate k, keep what passes, cap per prompt.
def rejection_sample(prompts, k, generate, judge):
    # generate(prompt) -> candidate string; judge(prompt, candidate)
    # -> True/False (the verifier from B.2, or a model judge).
    # Returns kept (prompt, candidate) pairs, at most one per prompt:
    # the cap is what prevents easy prompts from flooding the dataset
    # with near-duplicate successes.
    kept = []
    for p in prompts:
        for _ in range(k):
            c = generate(p)
            if judge(p, c):
                kept.append((p, c))
                break          # one keep per prompt, then move on
    return kept
```

::: walkthrough
1. **The inputs.** Each source declares what exists (size) and what you want (weight). The planner's job is to show the tension between the two.
2. **The epochs column is the diagnosis.** Code at 1.2 epochs is comfortable. Math at 3.0 and chat at 4.0 are repeated enough to memorize surface patterns. If chat quality matters (it sets tone), 4 epochs of the same 5B tokens is a risk: find more chat data or lower its weight.
3. **Steps from geometry.** 100B tokens at 2048-token sequences x 512 sequences per batch = ~95K steps. Change the batch and the step count moves, but the per-source epochs do not: epochs are a property of the mixture, not the hardware.
4. **The rejection cap.** `break` after the first keep is deliberate. Without it, easy prompts contribute k near-identical successes and the dataset's difficulty distribution collapses toward trivial.
5. **What breaks if you skip dedupe.** The recipe as written does not dedupe across prompts; in production, an exact-match pass and a near-dedupe pass follow. Duplicated reasoning traces teach the model to repeat itself verbatim.
:::

### B.3.4 Common misunderstanding

**"More data is always better; just add every source you can find."** At fixed compute, adding a source means removing tokens from somewhere else, and low-quality tokens actively harm: the model practices the wrong patterns. The QLoRA paper's result is the cleanest demonstration: a small high-quality instruction set beat larger, noisier alternatives. Mixture design is portfolio management under a token budget, and the scarcest resource is not tokens but verified-correct tokens. When in doubt, shrink the mixture and raise the bar.

### B.3.5 Visual: the mixture

![Three ingredient buckets labeled code 50 percent, math 30 percent, chat 20 percent pouring into a single training bowl](../scratch/appendix-07b/img/b3-mixture.webp)

*Figure B3.1. The SFT mixture as a recipe: three sources with target weights blending into one training distribution. The planner turns these percentages into tokens, epochs, and steps. Generated for this volume.*

::: walkthrough
1. **Three buckets, one bowl.** Each bucket's width is its target weight, not its available size: the mixture is about what you want, and the planner reconciles it with what exists.
2. **The tension the planner exposes.** The chat bucket wants 20% of 100B tokens but only 5B exist: the bowl gets 4 epochs of chat. The diagram shows intent; the epochs column shows cost.
3. **What the picture hides.** Quality differences between buckets. Two mixtures with identical percentages but different filtering bars produce different models: the rejection-sampling recipe is the quality control the percentages do not show.
:::

### B.3.6 Lab pointers

::: lab Lab B3.1: Plan your own mixture
Run `plan_mixture` with three sources you invent (sizes and weights). Find the source with the highest epochs. Halve its weight and re-run: how do the other sources' epochs change? Write one sentence on whether the new mixture is healthier.
:::

::: lab Lab B3.2: Difficulty re-balancing (paper exercise)
You rejection-sampled 10K prompts and kept 8K pairs: 7K from easy prompts, 1K from hard ones. Argue in three sentences why training on this raw kept set is worse than a re-balanced 4K/4K split, even though it is smaller. Then describe how you would implement the re-balance in `rejection_sample`.
:::

::: takeaway
- The mixture is a sampling distribution over sources; at fixed compute it is a token budget, and quality beats quantity.
- Plan in tokens and epochs per source, not vibes: the epochs column reveals repetition risk.
- Rejection sampling: generate k, filter, cap keeps per prompt, dedupe, re-balance by difficulty.
- Small clean datasets beat big noisy ones; verified-correct tokens are the scarce resource.
:::

::: provenance
**Last verified: September 2026.** Mixture planning math is standard SFT practice, reconstructed from first principles. The QLoRA paper's small-high-quality-dataset finding is as reported (Dettmers et al., 2023). Worked numbers are illustrative, computed from stated assumptions.
:::

## B.4 Preference-data collection design

### B.4.1 What preference data is

%%Preference data%% is judgments about which of two (or more) model responses is better, for a given prompt. Each item is (prompt, response A, response B, verdict): A wins, B wins, or tie. Reward models learn from these comparisons (Volume 7 Chapter 4); DPO learns from them directly (Volume 7 Chapter 5).

The data looks simple. Collecting it well is an operations discipline: the labels are the ground truth your reward model treats as reality, so every systematic error in collection becomes a systematic error in the model's values.

### B.4.2 Why collection design matters

A reward model cannot exceed its labels. Noisy, biased, or inconsistent preferences train a noisy, biased, inconsistent reward function, and RL then optimizes it faithfully. Three specific risks:

1. **Rubric ambiguity.** If labelers disagree on what "better" means, the dataset encodes the disagreement as noise.
2. **Labeler drift.** Standards shift over weeks of labeling; early and late labels are not drawn from the same distribution.
3. **Spurious correlations.** Labelers prefer longer answers, or more confident tone, or bullet lists. The reward model learns "longer = better" instead of "correct = better", and RL then breeds verbosity (a classic reward-hacking channel).

Collection design is how you fight all three before the data poisons the reward model.

### B.4.3 How it works under the hood

**Labeler instructions.** The rubric is a document, not a paragraph. It defines the criteria in priority order (correctness first, then helpfulness, then clarity). It gives paired examples of each verdict. It lists edge cases: two wrong answers, refusals, disallowed content. It tells labelers when to skip rather than guess. The rubric is versioned; labels carry the rubric version.

**Task design.** Pairwise comparison (A vs B) is the standard unit: it is the easiest judgment for humans to make reliably. Rankings of 3-4 cost more attention per item. Multi-turn comparisons judge whole conversations, which is harder and noisier; many pipelines judge single turns and compose.

**Disagreement handling.** Every item gets 3 labelers. Majority wins. Ties and abstentions go to expert adjudication, never to a coin flip: a coin flip trains the reward model on noise. Items with low agreement across the whole batch trigger rubric repair: the instructions were unclear, so fix the document, not the labelers.

**Agreement metrics.** %%Percent agreement%% (how often labelers pick the same winner) and %%Cohen's kappa%% (agreement corrected for chance) are tracked per batch. Healthy operations live around kappa 0.6-0.8. Below ~0.4, stop labeling and repair the rubric: more labels under a broken rubric just manufacture confident noise.

**Calibration.** Gold items with known-correct verdicts are mixed into every batch. Labelers who drift on gold get retrained. This is how you catch drift before it contaminates thousands of labels.

**AI feedback as a complement.** RLAIF and Constitutional AI (Volume 7 Chapter 7) generate preference labels with a model guided by written principles. It scales the labeling; it does not replace the rubric, the gold items, or the final human evals. The same quality-control math applies: measure agreement between the AI judge and human gold, and repair the principles when it sags.

```python
# Preference-label quality control: agreement metrics and adjudication.
# Three labelers judge each pair; the majority becomes the label, and the
# agreement rate tells you whether the rubric is working. Low agreement
# means the prompt is ambiguous (fix the instructions), not that the
# labelers are bad.

def cohens_kappa(a, b):
    # a, b: two labelers' 0/1 vote lists on the same pairs.
    # kappa = (observed - chance) / (1 - chance). Chance-correction
    # matters: two labelers who always vote 1 agree 100% but know
    # nothing; kappa gives them 0. Healthy ops: 0.6-0.8. Below ~0.4,
    # stop and repair the rubric before labeling another pair.
    n = len(a)
    agree = sum(1 for x, y in zip(a, b) if x == y) / n
    p_a1 = sum(a) / n            # labeler A's rate of voting 1
    p_b1 = sum(b) / n            # labeler B's rate of voting 1
    chance = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    return (agree - chance) / (1 - chance) if chance < 1 else 1.0

def adjudicate(votes):
    # votes: three 0/1 labels, None where a labeler abstained.
    # Majority of the non-abstained wins. A tie or total abstention goes
    # to expert review: never a coin flip, because a coin flip trains the
    # reward model on pure noise disguised as a label.
    clean = [v for v in votes if v is not None]
    if not clean:
        return "expert"          # everyone abstained: genuinely hard pair
    ones = sum(clean)
    zeros = len(clean) - ones
    if ones > zeros:
        return 1
    if zeros > ones:
        return 0
    return "expert"              # tie: a human expert decides

# Demo: 1000 pairs, three labelers each erring 12% independently.
import random
random.seed(11)
truth = [random.choice([0, 1]) for _ in range(1000)]
def labeler(err):
    return [t if random.random() > err else 1 - t for t in truth]
l1, l2, l3 = labeler(0.12), labeler(0.12), labeler(0.12)
print(f"kappa(l1,l2) = {cohens_kappa(l1, l2):.2f}")   # 0.59
print(f"kappa(l1,l3) = {cohens_kappa(l1, l3):.2f}")   # 0.54
majority = [1 if sum(v) >= 2 else 0 for v in zip(l1, l2, l3)]
acc = sum(1 for m, t in zip(majority, truth) if m == t) / 1000
print(f"majority-vote accuracy: {acc:.1%}")           # 96.2%
# Three 88%-accurate labelers vote a 96%-accurate label: redundancy buys
# reliability, which is the economic argument for 3x labeling.
```

::: walkthrough
1. **The error model.** Each labeler is right 88% of the time, independently. That is a decent but imperfect labeler: the regime real operations live in.
2. **Kappa in the healthy band.** Pairwise kappas of 0.54-0.59 sit just below the 0.6-0.8 healthy range: realistic for a rubric with some genuinely ambiguous items. Below 0.4 would mean stop and repair.
3. **The majority dividend.** Three 88% labelers vote a 96.2% label. The 8-point gain is the entire economic argument for triple labeling: redundancy converts mediocre judges into a reliable dataset.
4. **The adjudication rule.** `adjudicate` never flips a coin. Abstentions and ties route to experts. In the demo there are no Nones, so every item resolves by majority; in production, the "expert" queue is where rubric ambiguity becomes visible and repairable.
5. **What breaks if you single-label.** One 88% labeler gives you 88% labels with no way to know which 12% are wrong. The reward model then treats those errors as ground truth. Triple labeling costs 3x and buys the error detection.
:::

### B.4.4 Worked numbers: a labeling operation

10,000 pairs, 3 labelers each: 30,000 judgments. At 2 minutes per judgment (read prompt, read both responses, decide, justify briefly), that is 1,000 labeler-hours. With the demo's agreement profile: ~78% unanimous or 2-1 majority clean, ~8% ties/abstentions to expert review, and kappa tracked per 500-pair batch to catch drift. The expert queue (800 items) is small enough for a senior reviewer and is also the rubric-repair feed: every item that reaches an expert is a candidate rubric clarification.

### B.4.5 Common misunderstanding

**"Labeler disagreement means bad labelers; replace them."** Some prompts are genuinely ambiguous: two flawed answers with different flaws, a question with no clean resolution, criteria that conflict (more helpful vs more cautious). Disagreement on these is signal, not noise: it marks where the rubric is silent. The fix is adjudication plus rubric repair (add the edge case to the instructions with an example), not labeler churn. Operations that punish disagreement get labelers who hide it, and hidden disagreement becomes silent noise in your reward model.

### B.4.6 Visual: the labeling pipeline

![Two answer cards A and B above three judge figures, arrows converging to a majority vote box, with a side branch for expert review](../scratch/appendix-07b/img/b4-labeling.webp)

*Figure B4.1. Preference collection: three labelers judge each pair, majority wins, ties and abstentions go to expert review. Agreement metrics watch the whole operation for rubric decay. Generated for this volume.*

::: walkthrough
1. **Top: the pair.** One prompt, two responses. The labeler's only job is the comparison, the easiest reliable judgment format.
2. **Middle: three judges.** Independent votes, each recorded with the rubric version. Independence matters: labelers who discuss converge socially, not truthfully.
3. **Majority box.** Two-of-three becomes the label. The demo's math (88% to 96.2%) is this box's value proposition.
4. **Side branch: expert review.** Ties, abstentions, and low-agreement items land here. The experts' decisions feed back into the rubric as new edge-case examples: the loop that keeps the instructions honest.
:::

### B.4.7 Lab pointers

::: lab Lab B4.1: Measure agreement on real judgments
Write a 10-item rubric for "helpful code answers" (correctness first, then clarity). Take 20 prompt/response pairs (generate them with any model) and judge them yourself twice, a day apart. Compute your own self-agreement and kappa. The gap between your two passes is your personal ambiguity budget: every item you flipped is a rubric edge case to document.
:::

::: lab Lab B4.2: Find the spurious correlation (paper exercise)
Your reward model prefers longer answers. Design a labeling intervention: how would you change the instructions, the pair construction, or the aggregation to break the length correlation? Write the rubric addition in two sentences, then describe one gold item that would catch a labeler who still votes long.
:::

::: takeaway
- Preference data = (prompt, A, B, verdict); it is the ground truth reward models treat as reality.
- Collection design: versioned rubric with edge cases, pairwise tasks, 3 labelers with majority vote, expert adjudication for ties, gold-item calibration.
- Track kappa per batch (healthy 0.6-0.8; below 0.4 stop and repair); disagreement signals rubric gaps, not bad labelers.
- Triple labeling's dividend: three 88% labelers vote a 96% label.
- AI feedback (RLAIF/CAI) scales labeling but needs the same quality control against human gold.
:::

::: provenance
**Last verified: September 2026.** Agreement metrics (percent agreement, Cohen's kappa) and triple-labeling practice are standard data-operations practice, reconstructed from first principles. RLAIF/CAI labeling as described in Volume 7 Chapter 7 (Constitutional AI paper, Anthropic 2022). Worked numbers are illustrative, computed from stated assumptions; the kappa demo ran with seed 11 as shown.
:::

## Unit 2: Mechanistic interpretability basics

## B.5 What interpretability is for

### B.5.1 What it is

%%Mechanistic interpretability%% is the project of explaining what a neural network does in terms of its internal parts: which features it represents, how those features combine into computations (circuits), and how the computations produce behavior. "Mechanistic" marks the ambition: not just describing what the model did, but how the machinery did it, at a level where you could predict what it will do next.

The toolkit this unit covers: %%probing%% (testing whether information is present in activations), %%sparse autoencoders%% (splitting activations into interpretable features), %%circuit analysis%% (tracing how features connect), and %%steering%% (changing behavior by intervening on the machinery).

### B.5.2 Why it exists

Black-box evaluation tells you what the model did on the tests you ran. It cannot tell you why, and "why" matters in three situations:

1. **Debugging.** The model fails on a slice of inputs. Eval scores say where; interpretability aims at the mechanism, which tells you how to fix it.
2. **Trust under distribution shift.** Evals cover the tested distribution. If you understand the mechanism, you can reason about inputs you have not tested.
3. **Safety claims.** "The model does not do X" is stronger when you can point at the machinery and show the absence of the mechanism, not just the absence of observed failures.

None of these are solved problems. This unit teaches the methods and their limits with equal weight.

### B.5.3 How the field thinks: the ladder and the standard of evidence

The ladder of explanation has three rungs:

- **Behavior:** what the model outputs. Evals live here.
- **Representation:** what information the activations carry. Probing and SAEs live here.
- **Mechanism:** how the parts compute the behavior. Circuits and causal interventions live here.

Each rung up is harder and more valuable. The field's standard of evidence is %%correlation plus intervention%%: first show a feature correlates with a behavior (probe fires, SAE feature activates), then intervene (ablate the feature, steer the activation) and show the behavior changes. Correlation alone is a hypothesis; intervention is the test. This standard is what separates modern interpretability from "this neuron looks like X" storytelling.

The key background idea is %%superposition%% (Elhage et al., 2022): networks represent more features than they have dimensions, packing them as nearly-orthogonal directions in activation space. Consequence: individual neurons are %%polysemantic%% (one neuron fires for many unrelated concepts), because the true features are directions, not neurons. SAEs exist to find those directions.

### B.5.4 Worked numbers: the scale of the enterprise

Anthropic's SAE program scaled from 4,096 features on a small model (2023; features for DNA sequences, legal language, code) to roughly 1M, 4M, and 34M features on Claude 3 Sonnet (2024). The big runs surfaced features for abstract concepts including deception, sycophancy, and bias. Google DeepMind's Gemma Scope released SAEs across Gemma 2's layers. OpenAI trained a 16M-feature SAE on GPT-4 and published scaling laws relating compute, dictionary size, sparsity, and reconstruction quality. The direction is clear: interpretability is becoming a budgeted engineering effort, not a craft project.

### B.5.5 Common misunderstanding

**"Interpretability means attention heatmaps."** Attention weights show where the model looked, not what it computed or why. A high attention weight on a token does not mean that token caused the output, and heads routinely attend broadly while the decisive computation happens in the MLP. Modern interpretability barely uses raw attention visualization; it works on features (SAEs), probes, and causal interventions. If someone shows you a heatmap as an explanation, ask what intervention would change the behavior.

### B.5.6 Visual: the map

![Concept map: central node interpretability connected to probing, sparse autoencoders, circuits, and steering](../scratch/appendix-07b/img/b5-interp-map.webp)

*Figure B5.1. The interpretability toolkit: four methods, one shared standard of evidence (correlation plus intervention). Generated for this volume.*

::: walkthrough
1. **Center: the goal.** Mechanistic explanation: features, computations, behavior, linked causally.
2. **Probing (read-only).** Asks "is the information there?" Cheapest, weakest: presence is not use.
3. **Sparse autoencoders (read-only, finer).** Asks "what are the features?" Splits polysemantic activations into monosemantic directions.
4. **Circuits (structure).** Asks "how do features connect?" Traces computations across layers and heads.
5. **Steering (write).** Asks "does this cause that?" Intervenes on activations and watches behavior move. This is the intervention half of the standard of evidence.
:::

### B.5.7 Lab pointers

::: lab Lab B5.1: Sort claims by rung
Pick a model you use. Write three claims about it: one behavioral ("it refuses X"), one representational ("it tracks Y internally"), one mechanistic ("it does Z by computing W"). For each, write what evidence would support it and what intervention would test it. Notice how fast the evidence gets expensive as you climb.
:::

::: takeaway
- Mechanistic interpretability explains behavior in terms of internal features and computations.
- The ladder: behavior, representation, mechanism. The standard of evidence: correlation plus intervention.
- Superposition (more features than dimensions) causes polysemanticity (neurons with many unrelated jobs); SAEs aim to recover the true feature directions.
- Attention heatmaps are not explanations; features, probes, and interventions are the modern toolkit.
:::

::: provenance
**Last verified: September 2026.** Superposition/polysemanticity framing from Elhage et al., "Toy Models of Superposition" (2022). Anthropic SAE milestones: "Towards Monosemanticity" (Oct 2023), "Scaling Monosemanticity" (May 2024, Claude 3 Sonnet, ~1M/4M/34M features). Gemma Scope (DeepMind, Gemma 2) and OpenAI's 16M-feature GPT-4 SAE with scaling laws as publicly reported. Correlation-plus-intervention as the field's evidence standard, widely articulated in the literature.
:::

## B.6 Sparse autoencoders

### B.6.1 What an SAE is

A %%sparse autoencoder%% (SAE) is a small neural network trained to reconstruct a model's activations through a wide, sparse bottleneck. Input: an activation vector x from some layer (say 2,048 dimensions). The encoder maps it to a much wider feature vector f(x) (say 16,384 dimensions). A sparsity penalty forces most entries of f(x) to zero on any given input. The decoder reconstructs the activation from the few surviving features: x ≈ W_dec f(x) + b.

The bet: the surviving features are the monosemantic directions superposition hid. Instead of one neuron that fires for "DNA sequences, legal text, and Python comments," you get three features that each fire for one thing.

### B.6.2 Why it exists

Neurons are polysemantic because of superposition: the model has more features to represent than dimensions to put them in, so it packs them as overlapping directions. Reading neurons directly gives mush. Dictionary learning (the classical name for what SAEs do) separates the mush into a dictionary of clean directions: an overcomplete set of feature vectors such that each activation is a sparse combination of a few of them.

The 2023 result that launched the modern wave (Cunningham et al.): SAEs trained on language model activations learned features more interpretable and monosemantic than any alternative decomposition, measured by automated interpretability scoring. Ablating those features enabled precise behavior edits (e.g., removing pronoun prediction) with less collateral damage than prior methods.

### B.6.3 How it works under the hood

**Architecture.** Encoder: f(x) = ReLU(W_enc (x - b_dec) + b_enc). Decoder: x_hat = W_dec f(x) + b_dec. The decoder bias doubles as the centering term: activations are centered before encoding, which stabilizes training.

**Loss.** Reconstruction (mean squared error between x and x_hat) plus sparsity (L1 penalty on f(x)). The L1 term is the whole trick: without it, the network learns a wide identity map and every "feature" is mush. With it, features compete to explain the activation and only the winners survive per input.

**The expansion factor** (d_sae / d_model, typically 4x to 64x in practice, up to 256x) sets how many candidate features compete. More features: finer concepts, more compute, harder to inspect them all.

**Anti-cheating.** A subtle failure: the network can shrink feature activations and grow decoder weights to dodge the L1 penalty. The standard fix constrains decoder columns to unit norm, so features cannot cheat on scale.

**Training data.** SAEs train on cached activations: run the model over a large text corpus, save layer activations, train the SAE on those. The SAE never sees the model's weights, only its activity.

**Reading the features.** For each feature, look at the inputs where it activates most strongly (max-activating examples). A feature whose top examples are all database error messages is the "database error" feature. Automated scoring (asking another model to predict the feature's activation from a proposed description) scales this to millions of features.

**Using the features.** Clamp a feature's activation and watch behavior change: Anthropic's famous demo clamped the "Golden Gate Bridge" feature high and the model steered every topic toward the bridge. That is correlation plus intervention, the B.5 standard, applied to a single feature.

```python
# A sparse autoencoder in miniature: the whole idea in ~30 lines.
# x: an activation vector (d_model). The encoder lifts it into a much
# wider feature space (d_sae >> d_model); a sparsity penalty forces most
# features to zero; the decoder rebuilds x from the few survivors.
# NOTE: needs PyTorch; run on any GPU machine (RunPod-ready) or CPU.

import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        # Widening map d_model -> d_sae. The expansion factor
        # (d_sae / d_model, 4x-64x in practice) sets how many candidate
        # features compete to explain each activation.
        self.encoder = nn.Linear(d_model, d_sae)
        # Decoder back to d_model. Real SAEs constrain these columns to
        # unit norm so features cannot dodge the L1 penalty by shrinking
        # activations and growing weights; omitted here for brevity.
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # Center, encode, sparsify with ReLU, reconstruct.
        f = torch.relu(self.encoder(x - self.bias))
        x_hat = self.decoder(f) + self.bias
        # L1 on features: the sparsity pressure. Without this term the
        # network learns a wide identity map and every "feature" is mush.
        sparsity = f.abs().sum(dim=-1).mean()
        recon = ((x - x_hat) ** 2).mean()
        return x_hat, recon + 1e-3 * sparsity, f

# Toy run: 512-dim "activations", 4096 features (8x expansion).
torch.manual_seed(0)
sae = SparseAutoencoder(512, 4096)
opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
x = torch.randn(64, 512)        # stand-in for cached residual-stream acts
x_hat, loss, f = sae(x)
opt.zero_grad()
loss.backward()                 # gradients flow through the sparse ReLU
opt.step()
active = (f > 0).float().sum(dim=-1).mean().item()
print(f"loss={loss.item():.4f}  mean active features={active:.0f}/4096")
# After one step the L1 term is already pushing most features to zero;
# train on real cached activations and the survivors become interpretable.
```

::: walkthrough
1. **The shapes.** 512 in, 4096 features, 512 out. The bottleneck is wide but sparse: capacity lives in the width, interpretability in the sparsity.
2. **The centering.** `x - self.bias` before encoding, `+ self.bias` after decoding. Activations have a large mean component; centering keeps the encoder's job about variation, not offset.
3. **The ReLU.** Nonnegativity plus L1 gives true zeros, not small values. Zeros are what make features countable and inspectable: "47 features active on this token" is a meaningful sentence.
4. **The loss balance.** `recon + 1e-3 * sparsity`: the coefficient trades reconstruction fidelity against feature count. Too high and features die (everything reconstructs from bias); too low and nothing is sparse. Tuning it is the main practical knob.
5. **What the toy omits.** Unit-norm decoder columns (anti-cheating), real cached activations, dead-feature resampling (features that never activate are re-initialized), and the max-activating-examples inspection loop. Each is one paragraph in the SAE literature and one afternoon in the lab.
:::

### B.6.4 Worked numbers

d_model = 2048, expansion 8x: 16,384 features. Typical sparsity: tens to low hundreds active per token (illustrative; the exact count follows from the L1 coefficient). Parameters: encoder 2048x16384 + decoder 16384x2048 = ~67M, small next to the model it explains. Training tokens: billions of cached activations for frontier-scale SAEs. That is why Gemma Scope and the OpenAI 16M-feature SAE were notable compute investments. And it is why OpenAI's scaling laws (compute vs dictionary size vs sparsity vs reconstruction) matter: they turned SAE training from alchemy into budgeting.

### B.6.5 Common misunderstanding

**"Each SAE feature is one clean concept, and SAEs find all of them."** Features are hypotheses, not ground truth. Known issues:
- %%Feature splitting%%: one concept fragmented across several features.
- %%Feature merging%%: one feature covering two related concepts.
- Uninterpretable features: activate on no discernible pattern.
- Incomplete coverage: the SAE explains much of the activation variance but not all; the residual is unexplained machinery.

Treat a feature as a lead to be confirmed by intervention, per the B.5 standard, not as a discovered fact.

### B.6.6 Visual: the SAE

![Diagram: dense activation bar feeding into a wide sparse feature layer with a few highlighted dots, then reconstructing back into an activation bar](../scratch/appendix-07b/img/b6-sae.webp)

*Figure B6.1. The sparse autoencoder: a dense activation enters, a few of many candidate features activate (the highlighted dots), and the decoder rebuilds the activation from those few. Generated for this volume.*

::: walkthrough
1. **Left: the activation.** One dense vector from the model's residual stream: 2,048 numbers, polysemantic, unreadable.
2. **Middle: the feature layer.** 16,384 candidate features; only a handful are lit. Each lit feature is a direction the training found worth keeping: ideally one concept each.
3. **Right: the reconstruction.** The decoder rebuilds the activation from the lit features plus bias. Reconstruction error is the unexplained remainder: machinery the SAE did not capture.
4. **The sparsity is the product.** Without it, the middle layer is just a wide dense layer and the diagram means nothing. The L1 penalty is what makes "a handful lit" true.
:::

### B.6.7 Lab pointers

::: lab Lab B6.1: Train the toy SAE (RunPod-ready)
Run the SAE code on CPU or GPU. Sweep the sparsity coefficient (1e-4, 1e-3, 1e-2) and record mean active features and reconstruction loss after 100 steps. Plot the tradeoff: this curve is the practical face of the sparsity coefficient.
:::

::: lab Lab B6.2: Ablate a feature (paper exercise)
Pick the most active feature on your toy data. Zero it in the forward pass and measure the reconstruction error increase. Then zero a different, weakly-active feature and compare the two increases. The gap is the first feature's causal contribution: correlation (it was active) plus intervention (removing it hurt).
:::

::: takeaway
- SAEs reconstruct activations through a wide sparse bottleneck: x ≈ W_dec f(x) + b, most of f(x) zero.
- The L1 sparsity penalty is the mechanism; without it the SAE learns a wide identity map.
- Expansion 4x-64x; decoder unit-norm stops scale cheating; max-activating examples make features readable.
- Frontier scale: 34M features on Claude 3 Sonnet, Gemma Scope, 16M on GPT-4 with published scaling laws.
- Features are hypotheses: splitting, merging, uninterpretable features, and incomplete coverage are all real.
:::

::: provenance
**Last verified: September 2026.** SAE architecture/loss as in Cunningham et al., "Sparse Autoencoders Find Highly Interpretable Features in Language Models" (arXiv 2309.08600, 2023), including the IOI causal analysis and the pronoun-prediction ablation result. Anthropic milestones ("Towards Monosemanticity" Oct 2023; "Scaling Monosemanticity" May 2024) and the Golden Gate Bridge steering demo as publicly reported. Gemma Scope and OpenAI 16M-feature SAE as publicly reported.
:::

## B.7 Probing vs steering

### B.7.1 What they are

%%Probing%% is read-only: freeze the model, train a tiny classifier (usually linear) on its activations to predict some property, and check the accuracy. High accuracy means the information is present in the activations. The classic form is the %%linear probe%% (Alain and Bengio, 2016): a linear classifier per layer, asking what each layer knows.

%%Steering%% is write: modify activations during the forward pass to change behavior. The canonical recipe is %%contrastive activation addition%% (CAA, Rimsky et al., 2023). Collect contrast pairs: prompts showing the behavior vs its opposite. Take the mean difference of residual-stream activations as a steering vector v. Add alpha x v at token positions during generation. Positive alpha pushes toward the behavior, negative alpha away.

### B.7.2 Why the pair matters

Probing answers "does the model represent X?" Steering answers "does X drive behavior?" The pair is the B.5 standard of evidence in miniature: probe for correlation, steer for causation. Either alone misleads:

- A probe can be 95% accurate on information the model never uses (the probe is a stronger reader than the model's own downstream layers).
- Steering without a probe is blind: you are pushing a direction you have not verified means anything.

Together they are the cheapest interpretability loop that clears the correlation-plus-intervention bar.

### B.7.3 How they work under the hood

**Linear probes.** Take activations at layer L for a labeled dataset, train logistic regression (or least squares), report accuracy vs a majority-class baseline. Sweep layers: the layer where accuracy peaks is where the information is most linearly available. Keep the probe weak on purpose: a strong probe finding the signal proves nothing, because a strong probe can memorize. A linear probe winning is honest evidence the information is linearly present.

**ITI (inference-time intervention, Li et al., 2023).** The bridge between the two: train linear probes to find truthfulness-correlated attention heads, then shift activations along the probe direction at inference. Probing selects the direction; steering applies it.

**CAA mechanics.** For each contrast pair (same question, "honest" vs "sycophantic" completion), record the residual-stream activation at the answer position for both, take the difference, average over hundreds of pairs. The averaging cancels everything except the behavior direction: that is why contrast pairs, not single examples, are used. At inference, add the vector at all token positions after the prompt, scaled by alpha. Reported effects on Llama 2 Chat: steered behaviors (sycophancy, hallucination, refusal, corrigibility) moved substantially with small capability loss, and the steering composed with fine-tuning and system prompts rather than replacing them.

**Choosing alpha.** Sweep it. Too small: no visible change. Too large: the model degrades into repeating the steered concept (the Golden Gate Bridge demo is alpha set to maximum, on purpose, as a spectacle). The usable range is where the target behavior shifts and everything else stays put: measure both.

```python
# Probing vs steering on a toy concept. Runs anywhere (numpy only).
# Probe (read-only): a least-squares linear classifier on activations.
# High accuracy = the information is present.
# Steering (write): a mean-difference vector from contrast pairs, added
# during the "forward pass". A behavior shift = the direction is causal.

import numpy as np

rng = np.random.default_rng(0)

def make_activations(n, truthful):
    # Toy world: 64-dim activations; dimension 0 secretly encodes
    # "truthfulness" as +2/-2 plus noise. The probe must discover it;
    # steering must move it.
    X = rng.normal(0, 1, (n, 64))
    X[:, 0] += np.where(truthful, 2.0, -2.0)
    return X

X_true = make_activations(500, True)
X_false = make_activations(500, False)
X = np.vstack([X_true, X_false])
y = np.array([1] * 500 + [0] * 500)

# --- Probe: least-squares linear classifier, closed form, no libraries.
# Deliberately weak (linear): if THIS finds the signal, the information
# is honestly, linearly present. A strong probe winning proves nothing.
Xc = X - X.mean(0)
w = np.linalg.solve(Xc.T @ Xc + 1e-3 * np.eye(64), Xc.T @ (y - 0.5))
pred = (Xc @ w > 0).astype(int)
print(f"probe accuracy: {(pred == y).mean():.1%}  (chance: 50%)")
print(f"weight on dim 0: {w[0]:.2f} vs mean |w|: {np.abs(w).mean():.3f}")
# probe accuracy: 98.2%  (chance: 50%)
# weight on dim 0: 0.20 vs mean |w|: 0.010
# The probe put 20x the average weight on the true dimension: it found it.

# --- Steering: mean-difference vector, CAA-style.
# v = mean(truthful acts) - mean(false acts), over contrast pairs.
# Add alpha * v to FRESH false-labeled activations and watch the probe's
# verdict move: correlation (probe) becomes intervention (steering).
v = X_true.mean(0) - X_false.mean(0)
X_new = make_activations(200, False)      # fresh items, labeled false
for alpha in [0.0, 0.5, 1.0]:
    steered = X_new + alpha * v
    verdict = ((steered - X.mean(0)) @ w > 0).mean()
    print(f"alpha={alpha}: probe says truthful {verdict:.0%} of the time")
# alpha=0.0: probe says truthful 2% of the time
# alpha=0.5: probe says truthful 52% of the time
# alpha=1.0: probe says truthful 96% of the time
```

::: walkthrough
1. **The toy world.** Dimension 0 carries the concept; the other 63 are noise. This is the ground truth the methods must rediscover: a clean room for checking the logic.
2. **The probe.** Closed-form least squares, linear, no tricks. 98.2% accuracy with 20x weight on dimension 0: the information is linearly present, and the probe says exactly where.
3. **The steering vector.** Mean difference over 500 contrast pairs. Averaging is the noise control: any single pair's difference is full of irrelevant variation; the mean keeps only what is consistent across pairs.
4. **The intervention.** Fresh false-labeled items, never seen during vector construction, get alpha x v added. At alpha=1.0 the probe flips 96% of them to "truthful": the direction is causal in this toy, not just correlated.
5. **The honest caveat.** The toy's concept lives on one clean dimension. Real concepts are distributed and entangled: the same code on real activations gives messier, smaller, still meaningful shifts. The logic transfers; the magnitudes do not.
:::

### B.7.4 Common misunderstanding

**"A 95%-accurate probe means the model uses that information."** No. A probe is a new classifier trained on the activations; it can learn to read information that the model's own downstream layers ignore. The classic control: probe a randomized or untrained network and watch probes still score above chance on some tasks (they fit the probe to spurious patterns). Presence is not use. The claim "the model uses X" requires intervention: steer or ablate along the probe direction and show the behavior moves. Probe proposes, steering disposes.

### B.7.5 Visual: probe vs steer

![Two panels: probing with an arrow reading out from a network layer to a magnifier, steering with an arrow injecting into the layer](../scratch/appendix-07b/img/b7-probe-steer.webp)

*Figure B7.1. Probing reads (left): a classifier on frozen activations asks whether information is present. Steering writes (right): a vector added during the forward pass asks whether the direction is causal. Generated for this volume.*

::: walkthrough
1. **Left panel: the probe.** The layer is frozen; the arrow points out to a magnifier (the classifier). Nothing about the model changes. The question is presence.
2. **Right panel: the steering vector.** The arrow points into the layer, added at every post-prompt token position, scaled by alpha. The model runs differently. The question is causation.
3. **The loop between them.** Probe first to find candidate directions cheaply; steer to test the winners. ITI is literally this loop: probes select truthfulness heads, intervention shifts them.
4. **What the picture omits.** Alpha selection (sweep it), layer selection (steering works best at middle-to-late layers in practice), and the capability check (verify everything else stayed put).
:::

### B.7.6 Lab pointers

::: lab Lab B7.1: Run the probe-steer loop
Run the numpy script. Change the concept strength (2.0 to 0.5) and watch probe accuracy and steering effectiveness fall together. Then add a second concept on dimension 1 and steer only dimension 0's vector: check whether the dimension-1 probe verdict moves (it should not, much). That non-movement is specificity: the beginning of a real steering evaluation.
:::

::: lab Lab B7.2: The randomized control (paper exercise)
Train the same probe on activations from a randomly initialized network of the same shape. If it scores above chance, write two sentences on what that means for interpreting probe accuracies on trained models. (This control is standard in the probing literature.)
:::

::: takeaway
- Probing (read-only, linear, weak-on-purpose) tests whether information is present; steering (write, CAA mean-difference vectors, alpha-scaled) tests whether a direction is causal.
- The pair is correlation plus intervention: probe proposes, steering disposes.
- CAA: average activation differences over contrast pairs, add at post-prompt positions with ±alpha; reported to steer high-level behaviors in Llama 2 Chat with small capability cost.
- A high probe score alone never proves the model uses the information; run the intervention.
:::

::: provenance
**Last verified: September 2026.** Linear classifier probes: Alain and Bengio, "Understanding intermediate layers using linear classifier probes" (2016). CAA: Rimsky et al., "Steering Llama 2 via Contrastive Activation Addition" (arXiv 2312.06681, 2023): mean-difference vectors, post-prompt positions, ±coefficient, Llama 2 Chat evaluations as reported. ITI: Li et al. (2023) truthfulness probes on attention heads. Demo numbers ran as shown (numpy, seed 0).
:::

## B.8 Limits and open questions

### B.8.1 What this chapter is

An honest boundary map. Interpretability is a young field with real results and real limits, and the limits matter more than the results for anyone making decisions with these methods. Five limits, then the open questions.

### B.8.2 The limits

**1. Correlation is not causation, and most results are correlation.** The bulk of published interpretability is "feature X activates when the model does Y." The intervention half of the standard is rarer because it is harder. Ablating or steering cleanly, measuring the behavior change, ruling out side effects: each is real work. Treat un-intervened findings as leads.

**2. Coverage is partial.** SAEs explain much of activation variance but not all; the residual is unexplained machinery. A safety argument of the form "we checked all the features and found no deception circuit" fails at the first step: you did not check all the machinery, only the part your SAE captured.

**3. Steering is brittle.** It works on broad behavioral tendencies (more/less sycophantic, more/less refusing). It degrades on precise structured tasks. Steering a model to produce valid JSON for a schema, for example, has been reported to fail where the same steering moves open-ended style easily. The direction is real but coarse; fine control is an unsolved problem.

**4. Explanations need evaluation too.** "This feature means X" is itself a claim requiring evidence: predictive scoring on held-out activations, intervention tests, human or automated agreement. Automated interpretability scoring exists but is noisy. An explanation method without an evaluation method is storytelling.

**5. Scale costs.** Frontier SAEs are large compute investments (billions of cached activations, published scaling laws). Interpretability at the frontier is budgeted like training: which means most practitioners will use released SAEs (Gemma Scope) rather than train their own, inheriting the releaser's layer choices, sparsity settings, and blind spots.

### B.8.3 Open questions

- **Faithful circuits at scale.** Small circuits (induction heads) are understood; full behavioral circuits in frontier models are not. The gap between "a feature" and "the computation" is the field's main open problem.
- **Prediction before deployment.** Can interpretability find failure modes evals miss, before they matter? A few demonstrations exist; a reliable practice does not.
- **Composable steering.** Steering vectors for two behaviors often interfere. Reliable composition (steer A and B independently) is unsolved.
- **Theory of superposition.** Why do networks pack features the way they do, and can architectures be designed to be interpretable by construction rather than by post-hoc dissection?

```ascii
Claim about the model
        |
        v
What rung is it? ---- Behavior ----> Evals: run the tests
        |
        +--- Representation ---> Probe or SAE (correlation)
        |                             |
        |                             v
        |                       Intervene: steer or ablate
        |                             |
        |                    +--------+--------+
        |                    v                 v
        |              Behavior moved?    Behavior moved?
        |               YES  |              NO  |
        |                    v                 v
        |          Mechanistic evidence   Spurious correlation:
        |                                 discard
        |
        +--- Mechanism ------> Circuit tracing + interventions
                                        |
                                        v
                              Mechanistic evidence
```

*Figure B8.1. The evidence ladder as a decision procedure: every interpretability claim climbs until it passes an intervention test or is discarded.*

::: walkthrough
1. **Start with the claim.** Every investigation begins as a sentence about the model ("it tracks user intent in layer 20").
2. **Name the rung.** Behavioral claims need evals; representational claims need probes or SAEs; mechanistic claims need circuits.
3. **The intervention gate.** Representational findings must pass through steering or ablation. "Behavior moved" promotes the finding to evidence; "no movement" discards it as spurious correlation.
4. **What the chart enforces.** No finding reaches "mechanistic evidence" without an intervention. That single rule, applied consistently, is most of what separates this field's strong work from its storytelling.
:::

### B.8.4 Common misunderstanding

**"Interpretability will let us prove a model is safe."** It will not, not in its current state, and possibly not ever in the strong sense. What it offers is evidence: features, probes, interventions that raise or lower confidence in specific claims about specific behaviors. Treat interpretability outputs the way you treat eval outputs: as measurements with error bars, blind spots, and failure modes, not as proofs. The honest use is comparative and diagnostic: this checkpoint vs that one, this behavior's mechanism vs unknown.

### B.8.5 Lab pointers

::: lab Lab B8.1: Audit an interpretability claim
Find any public claim of the form "we found the X feature/circuit in model Y." Score it against the ladder: did it report an intervention? A control? Coverage of the residual? Write the strongest version of the skeptic's case in three sentences, then the strongest version of the supporter's case. Notice which one was easier to write: that asymmetry is the field's current state.
:::

::: takeaway
- Limits: most results are correlational; coverage is partial; steering is coarse and brittle on structured tasks; explanations need their own evaluation; frontier SAEs are expensive.
- Open: faithful circuits at scale, pre-deployment failure prediction, composable steering, theory of superposition.
- The decision rule: no finding counts as mechanistic evidence without passing an intervention test.
- Interpretability produces evidence with error bars, not safety proofs.
:::

::: provenance
**Last verified: September 2026.** Limits summarized from the field's published state (SAE coverage caveats, steering brittleness reports, scaling-law publications). The evidence-ladder framing is pedagogical, built for this volume from the correlation-plus-intervention standard.
:::

## B.9 Patch: LoRA and QLoRA with GPU-memory accounting drills

### Why this patch exists

Volume 7's main chapters teach full-parameter post-training: every weight moves. In practice, much fine-tuning moves a tiny fraction of the weights instead. This patch covers %%LoRA%% (Low-Rank Adaptation) and %%QLoRA%% (quantized LoRA): the parameter-efficient methods that made fine-tuning 7B-65B models possible on single GPUs. Volume 7 does not cover them; the drills below close the gap.

### B.9.1 What LoRA is

LoRA freezes the pretrained weights W and learns a low-rank update instead: W' = W + B x A. B is (d_out x r), A is (r x d_in), and the rank r is tiny (8, 16, 64) next to dimensions like 4096 or 8192. Only A and B train; W never moves.

The forward pass: h = Wx + (alpha/r) x B(Ax). B starts at zero, so training begins exactly at the base model (delta = 0) and the adapter grows from nothing. At inference, B x A can be merged back into W (W_merged = W + BA), so there is zero latency cost: the deployed model is just a dense matrix again.

Applied to a 7B model with r=8 on the attention projections, LoRA trains roughly 4M parameters out of 7B: about 0.06%. The bet: fine-tuning lives in a low-rank subspace, so a small update captures most of the adaptation.

### B.9.2 What QLoRA adds

QLoRA (Dettmers et al., 2023) keeps the LoRA idea and quantizes the frozen base to 4-bit, with three innovations:

1. **NF4 (4-bit NormalFloat):** a 4-bit data type described as information-theoretically optimal for normally distributed weights (which pretrained weights approximately are). More precision where the weight mass is.
2. **Double quantization:** quantize the quantization constants themselves, saving ~0.4 bits per parameter on top.
3. **Paged optimizers:** optimizer state pages spilled to CPU memory during spikes, so transient memory peaks do not OOM the run.

Headline result, as reported: fine-tune a 65B model on a single 48GB GPU while preserving full 16-bit fine-tuning task performance. Their Guanaco models reached 99.3% of ChatGPT's score on the Vicuna benchmark, with 24 hours of fine-tuning on one GPU. The paper's analysis also notes full 16-bit fine-tuning of 65B needs over 780 GB: the gap QLoRA closes.

### B.9.3 How it works under the hood

**Rank and the update.** The adapter's capacity is r x (d_in + d_out) parameters per targeted matrix. Rank 8 on a 4096x4096 matrix: 8 x 8192 = 65K params vs 16.8M for the full matrix. Double r, double the adapter. The alpha/r scaling keeps the update magnitude stable as you sweep r: without it, larger r would mean larger initial steps for no good reason.

**Target modules.** LoRA is usually applied to attention projections (q_proj, v_proj, sometimes k_proj, o_proj) and optionally the MLP. More targets = more capacity = more memory. The choice is a knob, not a law: attention-only is the cheap default; adding MLP targets helps when the adaptation needs it.

**Where QLoRA's memory goes.** The 4-bit base is dequantized to bf16 on the fly for the forward/backward compute (compute stays in bf16; only storage is 4-bit). Gradients flow through the frozen quantized weights into the bf16 adapters. So the memory is: 0.5 bytes/param for the base + full bf16 training state for the ~0.1% of params that are adapters.

**The honest limit.** LoRA is excellent at style, format, and instruction-following adaptation: teaching the model how to talk. It is weaker at adding genuinely new knowledge or rewiring deep behaviors: the low-rank update can only nudge what the base model already represents. If the base model cannot do X at all, LoRA rarely teaches X from scratch; that needs pre-training-scale updates or a stronger base.

```python
# LoRA in miniature: the weight surgery and the memory accounting drill.
# Frozen base W; trainable low-rank update B @ A. Forward:
# h = Wx + (alpha/r) * B(Ax). B starts at zero so training begins exactly
# at the base model. NOTE: needs PyTorch; RunPod-ready.

import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False   # the base never moves: this is the
                                      # entire memory trick
        d_out, d_in = base.weight.shape
        # A: random init (the "directions" to learn in). B: zeros, so the
        # initial update is exactly zero and training starts at the base
        # model. If B were random too, step 0 would jump away from the
        # pretrained weights: the zero-init is load-bearing.
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.02)
        self.B = nn.Parameter(torch.zeros(d_out, r))
        # alpha/r keeps the update scale stable as you sweep r: without
        # it, doubling r would double step sizes for no reason.
        self.scaling = alpha / r

    def forward(self, x):
        # Base path (frozen) plus adapter path (trainable). At inference
        # this merges to a single matrix: W + scaling * B @ A.
        return self.base(x) + self.scaling * (x @ self.A.T @ self.B.T)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def finetune_memory_gb(params_b, mode):
    # The accounting drill, in GB for weights + optimizer states.
    # Full fine-tuning (bf16): weights 2 B/param + grads 2 B/param +
    #   Adam m and v in fp32 (4+4 B/param) = 12 B/param = 6x weight GB.
    # LoRA: frozen bf16 weights (2 B/param) + tiny adapter states.
    # QLoRA: 4-bit base (0.5 B/param) + tiny adapter states.
    # Activations are not modeled here: they add on top for all modes.
    w_bytes = {"full": 2.0, "lora": 2.0, "qlora": 0.5}[mode]
    weights = params_b * w_bytes        # billions x bytes = GB
    if mode == "full":
        return weights * 6             # 12 bytes/param total
    # Adapters are ~0.1% of params: their grads + Adam states fit in
    # ~0.5 GB for 7B at r=8. The base dominates; quantization shrinks it.
    return weights + 0.5

for mode in ["full", "lora", "qlora"]:
    mem = finetune_memory_gb(7, mode)
    print(f"7B {mode:5s}: {mem:5.1f} GB (weights + optimizer states)")
# 7B full :  84.0 GB (weights + optimizer states)
# 7B lora :  14.5 GB (weights + optimizer states)
# 7B qlora:   4.0 GB (weights + optimizer states)
# Full fine-tuning needs multi-GPU or CPU offload; LoRA fits a 24 GB
# card; QLoRA fits a 16 GB card. Same model, three budgets.
```

::: walkthrough
1. **The surgery.** `LoRALinear` wraps a frozen linear layer. `A` holds learned input directions, `B` maps them back to output space. The product BA is the low-rank update; the rank r is its bottleneck.
2. **The zero-init.** `B` starts at zeros, so BA = 0 and the model behaves exactly as the base at step 0. This is what makes LoRA safe to attach to a working model: nothing changes until training moves B.
3. **The memory trick.** `requires_grad = False` on the base means no gradients and no optimizer states for 99.9% of params. The drill's numbers: full fine-tuning stores 12 bytes/param (weights + grads + Adam m,v); LoRA stores 2 bytes/param for the frozen base plus crumbs for adapters.
4. **QLoRA's extra squeeze.** The base drops from 2 to 0.5 bytes/param (NF4). Compute still happens in bf16 (dequantize on the fly); only storage is 4-bit. That is how 65B fits 48 GB: ~33 GB of weights plus adapters, optimizer, and activations.
5. **What the drill omits.** Activations (add on top for all modes; gradient checkpointing trades compute for them), the double-quantization saving (~0.4 bits/param), and paged-optimizer spill. The drill is the skeleton; those are the flesh.
:::

### B.9.4 The memory drill, worked

```
7B model, bf16, full fine-tuning
============================================================
weights      14.0 GB   (7B x 2 bytes)
gradients    14.0 GB   (7B x 2 bytes)
Adam m        28.0 GB   (7B x 4 bytes fp32)
Adam v        28.0 GB   (7B x 4 bytes fp32)
------------------------------------------------------------
total        84.0 GB   -> no single 40/48 GB card; needs 80 GB
                           + offload, or FSDP across cards

7B model, LoRA r=8 (attention projections), bf16 base
============================================================
frozen base  14.0 GB   (no grads, no optimizer: the trick)
adapters      ~0.1 GB   (4M params x 12 bytes)
------------------------------------------------------------
total        ~14.5 GB   -> fits a 24 GB card with activations

65B model, QLoRA (NF4 base), bf16 adapters
============================================================
frozen base  ~32.5 GB   (65B x 0.5 bytes)
adapters+opt  ~2.0 GB
activations    ~8.0 GB   (sequence-length dependent)
------------------------------------------------------------
total        ~42.5 GB   -> fits a single 48 GB card (as reported)
```

Read the three blocks as one lesson: the optimizer states are the hidden giant in full fine-tuning (56 of 84 GB), and LoRA deletes them by freezing the base. QLoRA then compresses the base itself. Each method removes one term of the sum.

### B.9.5 Common misunderstanding

**"LoRA matches full fine-tuning everywhere, so full fine-tuning is obsolete."** LoRA matches it on adaptation tasks: style, format, instruction following, domain tone. It lags where the task needs new knowledge or deep behavioral change, because a rank-8 update cannot rewrite what the base model never learned. The honest rule: LoRA for teaching how to talk, full-parameter (or a stronger base) for teaching new things. Rank and target-module choice move the boundary but do not erase it.

### B.9.6 Lab pointers

::: lab Lab B9.1: Count the adapters (paper exercise)
A 7B model: 32 layers, hidden size 4096, attention projections q/k/v/o each 4096x4096. LoRA r=8 on q_proj and v_proj only. Compute trainable params: per matrix r x (4096+4096) = 65,536; x2 matrices x32 layers = 4,194,304 (~4.2M, 0.06% of 7B). Now redo with r=64 on all four projections: how many params, what fraction? Write both.
:::

::: lab Lab B9.2: Run the memory drill (RunPod-ready)
Run `finetune_memory_gb` for 13B and 70B in all three modes. For each, name the smallest single GPU that fits weights+optimizer (40/48/80 GB), then add a rough activation estimate (batch 4 x 2048 tokens x layers x hidden x 2 bytes x small factor) and check whether your answer changes. The cases where it changes are where activation memory, not the optimizer, is the binding constraint.
:::

::: takeaway
- LoRA: freeze W, learn low-rank BA; B zero-init keeps step 0 at the base model; merges to zero-cost inference.
- QLoRA: 4-bit NF4 base + double quantization + paged optimizers; reported 65B on one 48GB GPU at full 16-bit task performance (Guanaco 99.3% of ChatGPT on Vicuna).
- The drill: full fine-tuning = 12 bytes/param (optimizer states dominate); LoRA deletes grads+optimizer for the base; QLoRA compresses the base 4x.
- 7B: 84 GB full / 14.5 GB LoRA / 4 GB QLoRA (weights+optimizer; activations extra).
- LoRA teaches how to talk; new knowledge needs more. Rank and target modules set the boundary.
:::

::: provenance
**Last verified: September 2026.** LoRA (Hu et al., 2021): W + BA formulation, zero-init, alpha/r scaling, merge-at-inference as in the paper. QLoRA (Dettmers et al., NeurIPS 2023): NF4/double quantization/paged optimizers, 65B on single 48GB GPU, >780 GB for full 16-bit 65B, Guanaco 99.3% of ChatGPT on Vicuna, 24h single-GPU fine-tune, all as reported in the paper. Drill numbers computed from stated shapes, illustrative (activations excluded unless noted).
:::

---

## Closing: how the appendices connect

- **A.1-A.3** gave you the RL systems stack: budget the step, generate with inference engines, decouple with async, size the memory.
- **B.1-B.4** gave you the data stack: synthesize and verify, mix and filter, collect judgments without fooling yourself.
- **B.5-B.8** gave you the interpretability toolkit and its honest limits.
- **B.9** gave you the efficient fine-tuning methods Volume 7 skipped, with the memory math to choose between them.

Volume 8 picks up the aligned model and serves it: the inference engines from A.2 return as the serving stack, and the memory math from A.3 and B.9 becomes the capacity plan.
