# Markdown-first build pipeline

## Layout

```
build/
  build.py        # MD -> self-contained HTML (run from this dir)
  ds.css          # ONE canonical stylesheet (inlined at build time)
  ds.js           # ONE canonical script (inlined at build time)
  sources/        # your .md volume sources go here (one per volume)
  README.md       # this file
```

## Build

```bash
cd ~/workspace/your_files/research-engineer-curriculum/build
python3 build.py sources/04-llm-internals.md        # one volume
python3 build.py --all                               # every .md in sources/
python3 build.py --check ../04-llm-internals.html    # QA gates on built HTML
```

Output HTML lands next to the .md source (or pass an explicit out path by editing
build_one call). The build inlines ds.css + ds.js fresh on every run, so a design
fix in the canonical files propagates to all volumes with one rebuild.

## Front matter (optional, top of file)

```markdown
---
title: LLM Internals
eyebrow: Volume 4
---
```

## Learning-component directives

Fenced `:::` blocks. Content inside is Markdown.

```markdown
::: takeaway
- point one
- point two
:::

::: ob-board
Open question for the reader...
:::

::: ob-board Custom board title
...
:::

::: lab Lab 4.1: Title here
Lab body. Code fences work inside.
:::

::: callout warn
Warning text...
:::
```

`callout` kinds: `warn`, `interview`, or plain (`::: callout`).

```markdown
::: walkthrough
1. Start at the left box.
2. Follow the arrow.
:::

::: provenance
**Last verified: September 2026.** Live-verified: ... **UNVERIFIED:** ...
:::

::: pq
**Q1.** Question stem?
A. first option
B. second option
::: answer
**Answer: B.** Why B is right.
- **A, wrong:** reason
- **B, right:** reason
:::
:::
```

First-use terms: `%%gradient descent%%` renders as `<dfn>gradient descent</dfn>`.
(Not processed inside code fences.)

## Images

- `![alt](relative/path.png)` with a **local file** is embedded as a base64 data
  URI at build time (offline-safe, renders from file://). Missing files warn.
- Remote `https://` image URLs are left as-is but WARN (they break offline use);
  prefer downloading the image into sources/ and referencing it locally.
- Internet-sourced images: download, inspect, keep a source credit line in the
  caption or nearby prose.
- Mermaid: ` ```mermaid ` blocks render to inline SVG via the local mmdc install
  (`~/workspace/tools/mermaid`, puppeteer config in `build/puppeteer.json`
  pointing at the VM's Chromium); if a render ever fails the block stays a
  readable code block and the build warns.
- ASCII diagrams in fenced code blocks are the guaranteed fallback.

## Fonts

Body stack: `"Atkinson Hyperlegible", "Inter",` then system fallbacks. The fonts
are installed on this VM, so Chromium renders and print-to-PDF embeds the real
font; on machines without them the system fallback applies (no webfont fetch,
no render dependency). Never add `@import`/webfont links.

## QA gates (run `--check` on every built file)

- Zero em dashes anywhere in shipped content, zero CSS gradients, zero external stylesheet/script/img
- Humanizer gate (from `~/workspace/skills/humanizer/SKILL.md`, user standing order): hard FAIL on banned AI tells — delve, furthermore, moreover, additionally/overall as sentence starters, tapestry, seamless, game-changer, supercharge, "deep dive", "it's worth noting", "it's important to note", "in conclusion", "as an AI", elevate, "in today's". WARN (human judges) on robust / leverage-as-verb / unlock, throat-clearing openers ("in this section we will…", "let's talk about…"), and sentences over ~30 words (count + samples reported)
- Code standard (user hard rule): ALL code samples are Python — hard FAIL on any fenced block tagged with another programming language; WARN on low comment density in Python samples over 15 lines (<8% comment lines) and on suspicious untagged blocks. Manual checklist per volume: every code sample has what/why/what-breaks comments and complexity notes where relevant; non-trivial samples get a plain-English walkthrough; key flows (attention, backprop, collectives, RL loops, inference batching, RAG/agent flows) pair code with a visual (generated image / Mermaid / ASCII) accurate to the code
- Balanced tags, unique ids, no TODO/FIXME/lorem, no personal identifiers
- Then screenshot-verify in headless Chromium at 1440 / 768 / 390 px and fix
  every visual defect (sidebar, drawer, tables/pre internal scroll only, images).

## Token efficiency

Bulk mechanical work (conversions, QA scripts, re-skins) goes to the cheapest
capable runner available on this VM (wrappers: `zclade`, `clade-mini`,
`codex-mini`, `zcodexx` — see ~/.bashrc). Never sacrifice quality for tokens.
