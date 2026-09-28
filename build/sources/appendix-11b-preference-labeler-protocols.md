---
title: Preference-Labeler Protocols
eyebrow: Appendix 11B
---

# Preference-Labeler Protocols

Volume 7 covers the other half of preference data: how to write rubrics, why agreement sits around 70 percent, and the classic pitfalls (verbosity bias, sycophancy, guideline drift, demographic skew). This appendix covers the operational half. How do you train labelers? How do you handle their disagreements? How do you design the labeler pool so it does not silently bias your reward model?

A preference label is a measurement made by a human instrument. This appendix is the instrument's operating manual.

## Chapter 1. From rubric to reliable labels

### 1.1 The instruction packet

Volume 7's rubric says what "better" means. The instruction packet teaches a new labeler to apply that rubric the same way a senior labeler would. The packet has four parts.

First, the task definition in one paragraph. What is the labeler comparing, and what does a label mean? "Pick the response you would rather receive" is not the same task as "pick the more factually correct response." Labelers will silently substitute one for the other. The packet must pin down which task they are doing.

Second, worked examples of right and wrong calls. Not one example; ten or more, including the close calls where reasonable people disagree. Each example carries the reasoning, not just the verdict. Labelers copy the reasoning pattern, not the verdict.

Third, an edge-case playbook. What to do when both answers are wrong, when both are correct, when one is correct but rude, when the answer refuses. Every unlabeled edge case becomes a coin flip, and coin flips become noise your reward model then learns.

Fourth, when to skip or flag. Some items are unlabelable: the question needs expertise the labeler lacks, or the rubric does not cover the case. A "flag for review" button is cheaper than a confident wrong label. Flagged items feed the adjudication queue in section 1.3.

```
Rubric (Volume 7: what "better" means)
        |
        v
Instruction packet: task definition, worked examples,
edge-case playbook, skip and flag rules
        |
        v
Qualification: gold-item test batch
        |
   +----+----+
   |         |
 passes     fails
   |         |
   v         v
Production  Retrain or
labeling    do not hire
   |
   v
Ongoing monitoring: gold items in every batch
```

### 1.2 Qualification: the gold-item test

Before a labeler touches production data, they label a test batch of 30 to 50 items where the correct answer is already known. These are gold items: written and adjudicated by senior labelers, covering the easy calls and the known-hard edge cases.

The bar is set in advance. A common bar is 80% agreement with the gold labels, with a hard floor on the easy items. Anyone who misses the obvious ones is not having a bad day. They misunderstand the task. Labelers who fail are retrained once, then not hired. This sounds harsh until you price the alternative: a mislabeled preference pair poisons the reward model, and the policy then optimizes toward the poison.

Qualification is not a one-time event. Seed gold items into every production batch, invisibly, at around 5%. If a labeler's gold accuracy drifts below the bar, they are paused and retrained. This is how you catch guideline drift, the slow reinterpretation of the rubric over weeks that Volume 7 names as a classic pitfall.

### 1.3 Disagreement handling: the adjudication ladder

Even good labelers disagree. Around 30% of preference pairs draw split verdicts in practice. The protocol decides what happens to those items, in four rungs.

Rung one is independent double labeling. Every production item goes to two labelers who cannot see each other. This is the cheapest way to find the items that matter: the agreements are your data, the disagreements are your signal about where the rubric is weak.

Rung two is majority vote. Disputed items go to a third independent labeler, and the majority wins. Majority vote is not magic; it is arithmetic. If each labeler is right 75% of the time independently, the majority of three is right about 84% of the time. The math: the majority is correct when at least 2 of 3 are correct. That is 3 times 0.75 squared times 0.25, plus 0.75 cubed, or 0.84375. The improvement only holds if the errors are independent, which is why the three labelers must work blind to each other.

Rung three is adjudication. Items that survive majority vote with patterns that look systematic, all the borderline cases of one type failing the same way, go to a senior labeler. The adjudicator does not just pick a winner; they write down why, and that write-up goes back into the instruction packet's edge-case playbook. Adjudication is how the protocol learns.

Rung four is escalation. If disagreement on a whole category stays high after adjudication, the rubric itself is ambiguous and no amount of voting fixes it. The item type is pulled from production, the rubric is rewritten, and previously labeled items of that type are relabeled. This is expensive and it is the right call. A reward model trained on coin flips learns to flip coins.

```
Two independent labels
        |
   +----+----+
   |         |
  agree    disagree
   |         |
   v         v
Label     Third labeler:
accepted  majority vote
              |
         +----+----+
         |         |
    clear majority  systematic
         |          pattern
         v           |
   Label accepted    v
               Senior adjudicator:
               decides + writes reasoning
                       |
                  +----+----+
                  |         |
            Reasoning feeds  Whole category
            back into the    disputed
            instruction      |
            packet           v
                       Escalate: rewrite
                       the rubric, relabel
```

Here is the majority-vote arithmetic as code, so you can feel how the numbers move.

```python
import math

def majority_vote_accuracy(p, n=3):
    # p: probability one labeler is right on one item.
    # n: odd number of independent labelers.
    # Returns P(majority is right) = P(at least (n+1)/2 labelers are right).
    # "Independent" is the load-bearing assumption: labelers must work blind
    # to each other, or their errors correlate and this number lies.
    if n % 2 == 0:
        raise ValueError("majority vote needs an odd number of labelers")
    needed = n // 2 + 1  # 2 of 3, 3 of 5, and so on.
    total = 0.0
    for k in range(needed, n + 1):
        # Binomial term: choose(n,k) ways for exactly k to be right,
        # times p^k * (1-p)^(n-k).
        total += math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    return total


# The worked example: 75% individual accuracy, 3 labelers.
print(f"3 labelers at 75%: majority = {majority_vote_accuracy(0.75, 3):.3f}")

# Diminishing returns: going from 3 to 5 labelers buys less than you hope.
print(f"5 labelers at 75%: majority = {majority_vote_accuracy(0.75, 5):.3f}")

# The floor matters most: weak labelers do not vote their way to quality.
print(f"3 labelers at 60%: majority = {majority_vote_accuracy(0.60, 3):.3f}")
```

::: walkthrough
1. The function sums binomial terms: for each k from the majority threshold up to n, it adds the probability that exactly k labelers are right.
2. With 75% individual accuracy, 3 labelers give 0.844. Five give 0.896. The jump from 1 to 3 is worth more than the jump from 3 to 5: diminishing returns.
3. With 60% individual accuracy, the majority only reaches 0.648. Majority vote amplifies a decent signal; it does not rescue a weak one. This is why qualification (section 1.2) comes first.
4. The independence assumption is doing the heavy lifting. If all three labelers share the same misunderstanding, they agree confidently and wrongly. Blind labeling plus gold items is what keeps the assumption honest.
:::

### 1.4 Designing the pool: diversity as a protocol

Volume 7 names demographic skew as a classic pitfall: a small, homogeneous labeler pool teaches the reward model one group's tastes as if they were universal. Diversity of the labeler pool is not a slogan; it is a design input with three concrete levers.

The first lever is coverage. Map the deployment population (languages, regions, age bands, domains of expertise) and staff the pool to cover it. If the model will serve medical questions, some labelers need medical literacy. If it will serve three languages, each language needs native-fluent labelers, not one bilingual hero doing triple duty.

The second lever is overlap. Assign the same items across demographic slices on purpose, then measure agreement by slice. If one slice systematically prefers shorter answers and another prefers thorough ones, you have found a real preference difference, not noise. The rubric then needs an explicit rule, or the product needs a per-audience decision. Either way, you found it in the data instead of shipping it as a bug.

The third lever is rotation. Labelers who label the same task for months develop house styles and shared shortcuts. Rotate people across task types, and retire anyone whose agreement with fresh labelers collapses. A pool that never turns over is a pool that has stopped measuring the world and started measuring itself.

::: takeaway
- Train labelers with an instruction packet: task definition, worked examples, edge-case playbook, skip and flag rules.
- Qualify on gold items before production, and keep seeding golds into every batch to catch drift.
- Handle disagreement on a ladder: double label, majority vote, adjudication with written reasoning, escalation to rubric rewrite.
- Design the pool for coverage across your deployment population, measure agreement by slice, and rotate people before house styles calcify.
:::

::: provenance
**Last verified: September 2026.** Majority-vote binomial arithmetic: standard probability. The four-rung adjudication ladder: double label, majority vote, senior adjudication with written reasoning, rubric escalation. Gold-item seeding at ~5% with an 80% bar. Both are documented preference-data operations practice from RLHF practitioner writeups. InstructGPT's reported ~70% labeler agreement and the named pitfalls (verbosity bias, guideline drift, demographic skew): Volume 7's coverage, consistent with the InstructGPT paper's reported methodology. **UNVERIFIED:** the specific constants (30 to 50 gold items, 5% seeding, 80% bar) are practitioner rules of thumb from operations writeups, not from one citable paper.
:::
