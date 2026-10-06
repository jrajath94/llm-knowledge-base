# Build Log — Research-Engineer Curriculum

## 2026-09-28 ~13:35 EDT — Coordinator setup
- Skill-universe extraction from /tmp/roles.json (180 roles): top demands Python, distributed training, experiments, evals, safety, JAX/PyTorch, pre-training, RL, inference optimization, retrieval, profiling, TPU/GPU, system design. 17 Research Engineer rows identified.
- Wrote BRIEF.md (crew contract), CURRICULUM-MAP.md (spine), index.html, 00-how-to-use.html.
- Spawned 9 writer workers (depth 2).

## Worker completions
- [x] Vol 7 post-training-rl (worker 2/9, completed 13:38 EDT): 07-post-training-rl.html, 3.6 MB, 10 chapters, 13 figures (2 internet-sourced credited, 11 generated meta/muse-image, ASCII traces), capstone SFT→DPO lab (marked reconstructed), QA passed (0 em dashes, 0 gradients, 0 emojis, balanced HTML, unique ids, all imgs base64 decode-verified, wm_clean Layer A clean, no jumb false positive).
- [ ] Vol 1 math (worker 1/9) — running
- [ ] Vols 2+3 ML/DL bridges (worker 3/9) — running
- [ ] Vols 4+5 LLM internals + pre-training (worker 4/9) — running
- [ ] Vol 6 distributed training (worker 5/9) — running
- [ ] Vols 8+9+10 inference bridge + RRK guide + MLOps (worker 6/9) — running
- [ ] Vol 12 paper spine (worker 7/9) — running
- [ ] Vols 11+13+14 research methods + system design + behavioral (worker 8/9) — running
- [ ] DSA track 300 (worker 9/9) — running

## Delivery plan (updated per user direction)
- Assembly order when all 9 workers complete: final QA sweep -> wm_clean.py Layer A over all HTML -> README.md (curriculum description + study order) -> BUILD-LOG final -> zip FIRST: ~/workspace/your_files/research-engineer-curriculum.zip (entire output dir, excluding __pycache__/*.pyc) -> then push to PRIVATE GitHub repo `research-engineer-curriculum` via ~/workspace/skills/github/bin/gh_publish.py (check, create-repo --private, push-dir via REST Contents API; auth is custom.github Secure Vault, verified against jrajath94). Never a public repo. Report: zip path + byte size + push status.

## 2026-09-28 ~13:41 EDT — User quality gate propagated
- User mandate: DO NOT HALLUCINATE; all info current as of Sept 2026. Model names, paper results, library APIs, hardware specs, pricing verified against live sources (official docs, arXiv, vendor pages); prefer 2026 sources; never invent API signatures/findings/benchmarks; UNVERIFIED markers where unverifiable; "Last verified: September 2026" provenance box per volume.
- Sent to all 8 running workers (Vols 1, 2+3, 4+5, 6, 8+9+10, 11+13+14, 12, DSA track). Paper-spine worker got an extra-strict variant (verify every arXiv ID/title/author/year on live arXiv).
- Vol 7 completed BEFORE the gate: coordinator will add the "Last verified" provenance box and run a verification sweep over its claims during assembly (its worker already used "as reported"/"illustrative" markers).

## 2026-09-28 ~13:47 EDT — Design system update propagated
- User mandate: pitch-black dark mode, exact tokens (--bg:#000000; --surface:#0d0d0d; --border:#262626; --text:#ececec; --muted:#a8a8a8; --accent:#f0b429; --link:#6cb2ff; --code-bg:#111111), system fonts only, 17px/1.75, justified paragraphs, 72ch, sticky sidebar (scroll-spy, search filter, localStorage checkmarks, progress bar, mobile drawer, keyboard nav), learning components, quality gates (mobile, a11y, print stylesheet, watermark clean).
- Built shared assets: design-system.css (node --check passed on JS; 0 gradients; 0 em dashes) and design-system.js (auto sidebar from h2/h3, scroll-spy, filter, checkmarks, progress, prev/next, copy buttons, .pq widgets, drawer, keyboard).
- Wrote DESIGN-APPLY.md (retrofit instructions: inline CSS+JS, sidebar skeleton, component classes).
- Sent retrofit order to all 8 running workers. Vol 7 (worker completed pre-update): coordinator retrofits it during assembly.

## 2026-09-28 ~13:55 EDT — Shared-JS bugfix
- DSA worker found a real bug: chapter-nav insertion did h2.parentNode.insertBefore(bar, nextH2), which throws NotFoundError when h2s are wrapped in <section> elements. Fixed to insert via nextH2.parentNode. Also removed one em dash from a JS comment (keeps grep-based QA at zero).
- Broadcast to all 7 running workers: re-inline design-system.js from source if already inlined.
- Completed: 2/9 (Vol 7 post-training-rl, DSA track-300 with retrofit). DSA worker QA: 300/300 unique problems, 251 atlas anchors resolve, 19/19 images decode, wm_clean Layer A clean.

## 2026-09-28 ~14:03 EDT — Vol 6 complete (worker 1/9)
- 06-distributed-training.html, 3.2 MB, 13 chapters, 11 figures (8 internet-sourced, 3 generated, all with walkthroughs).
- Quality-gate verification: corrected 1F1B bubble formula (was wrong: bubble = (p-1)/(m+p-1), same as GPipe; win is activation memory); fixed A100 ridge 156 -> 153 FLOP/byte; fixed nonexistent StateDictType.SHARDED -> StateDictOptions; labs tested on CPU or honestly marked; GPU-needing lab marked not executed, code py_compile-clean, APIs torch-2.14.0-verified; prices marked illustrative.
- Design retrofit applied, fresh patched JS inlined (byte-match confirmed), <section> wrappers removed, 27 first-use <dfn>, QA all pass, wm_clean Layer A clean.
- Completed: 3/9 (Vol 6, Vol 7, DSA track). Remaining: Vols 1, 2+3, 4+5, 8+9+10, 11+13+14, 12.

## 2026-09-28 ~14:04 EDT — Vols 2+3 complete (worker 3/9)
- 02-ml-foundations-bridge.html (3.1 MB, 5 ch: experiment loop, splits, metrics/calibration, leakage, significance) and 03-deep-learning-for-researchers.html (3.8 MB, 6 ch: normalization, init, optimizers, mixed precision, checkpointing, bridge).
- 24 embedded images (3 internet-sourced, 12 generated via OpenRouter meta/muse-image, 9 matplotlib) + 9 ASCII diagrams, all with walkthroughs.
- 11 labs pure NumPy+matplotlib, all run 2026-09-28 with stdout quoted verbatim; required leakage + numerics labs included.
- Live verification pass: AdamW/RMSNorm/BatchNorm/LayerNorm/Xavier/He/checkpointing/BF16/FP16/temperature scaling/AMP APIs all verified against arXiv; provenance "Last verified: September 2026", UNVERIFIED: none.
- QA false-positive notes: base64 PNG data contains coincidental TODO/FIXME letter sequences and "jumb"-family matches; strip data URIs before regex checks. <title> tags had 2 em dashes (real, fixed).
- Design retrofit + fresh patched JS inlined; wm_clean Layer A clean.
- Completed: 4/9 (Vols 2, 3, 6, 7, DSA track = 5 volumes + DSA). Remaining: Vols 1, 4+5, 8+9+10, 11+13+14, 12.

## 2026-09-28 ~14:04 EDT — Vol 1 complete (worker 8/9)
- 01-math-for-ml.html, 18.4 MB, 8 chapters, 30 images all decode.
- Live jsdom DOM test of the patched design-system.js against the real file: 8 nav groups, 41 items, 8 chapter footers inserted with no errors (confirms the NotFoundError fix in the section-wrapper scenario), copy buttons, pq widgets, search filter, localStorage, drawer. All pass.
- Live verification: Adam/AdamW/LoRA/double descent/linear scaling/perplexity/softmax-CE verified on arXiv; illustrative numbers marked UNVERIFIED inline; provenance box "Last verified: September 2026".
- wm_clean note: full 18.4MB file exceeds the 300s subprocess cap on base64 payloads; worker ran Layer A on text with data-URIs stripped (clean) + real PNG chunk parse (no JUMB/uuid/C2PA). Lesson for assembly: same approach for large files.
- Completed: 5/9 (Vols 1, 2, 3, 6, 7, DSA track = 6 volumes + DSA). Remaining: Vols 4+5, 8+9+10, 11+13+14, 12.

## 2026-09-28 ~14:05 EDT — Vols 8+9+10 complete (worker 9/9)
- 08-inference-serving-bridge.html (3.7 MB), 09-agents-rag-guide.html (2.9 MB), 10-productionizing-mlops.html (2.9 MB).
- Live verification (Sept 2026 sources): Splitwise, DistServe, vLLM PagedAttention (16-token blocks), Orca (OSDI 2022, credit added), GPTQ, AWQ (scaling correction applied), GGUF Q4_K_M, A100 80GB HBM2e ~2 TB/s, speculative decoding 1/(1-a) bound, ScaNN, gemini-embedding-001 GA. Fixed: canary math (1% vs 2% needs ~2-5k samples), KV-cache example labeled hypothetical.
- Retrofit: old light theme -> design-system.css, patched JS inlined fresh, sidebar skeleton, .dfn terms (36/20/13), provenance boxes with UNVERIFIED lists. QA all pass, wm_clean Layer A clean (deliverables are the .cleaned.html copies).
- Assembly note: dedupe .cleaned.html vs source copies; remove helper scripts (volkit.py, assemble_*.py, gen_vol1_images.py etc.) from the shipped dir.
- Completed: 6/9 (Vols 1, 2, 3, 6, 7, 8, 9, 10, DSA track). Remaining: Vols 4+5, 11+13+14, 12.

## 2026-09-28 ~14:10 EDT — Vol 12 complete (worker 7/9)
- 12-paper-spine.html, 6.38 MB, 18 chapters (14 papers + synthesis), 21 data-URI images, 16 labs, 98 <dfn> terms.
- All 14 papers verified against live arXiv (2026-09-28); 7 factual fixes: InstructGPT SFT 33k -> ~13k prompts (Table 6), 85±3% with error bar; CAI replaced unverifiable agreement claim with paper's actual >90% binary accuracy statements; FlashAttention "up to 20x smaller memory footprint" exact phrasing. Confirmed correct: Transformer 41.8 BLEU, GPT-2 63.24% LAMBADA, GPT-3 86.4%/3.14E+23 FLOPs, Kaplan/Chinchilla constants, DPO b=0.1, Mixtral 8x2, RAG 21,015,324 docs. UNVERIFIED: none.
- Fixed pre-existing bug: DeepSeek-R1 chapter was missing its h2.
- Retrofit done, patched JS byte-identical, QA all pass, wm_clean Layer A clean.
- Completed: 7/9. Remaining: Vols 4+5, Vols 11+13+14.

## 2026-09-28 14:17 EDT — Delivery-quality update (from CCAR-P track)
- New user-mandated gates: (1) ALL CSS/JS inlined per HTML file, zero external deps, must render perfectly from file://. (2) Screenshot-verify EVERY page in headless Chromium (Playwright) at 1440/768/390px, fix visual defects, world-class at every width.
- Sent to the 2 remaining workers (Vols 4+5, Vols 11+13+14) as a prevention requirement.
- Assembly plan: coordinator re-verifies all completed volumes against both gates (grep for external deps + Playwright screenshot sweep of every file at 3 widths), including Vol 7 retrofit and DSA JS re-inline.

## 2026-09-28 ~14:35 EDT — Markdown-first pipeline live (user pivot)
- build/build.py: MD -> self-contained HTML. Inlines ONE canonical ds.css + ds.js fresh every run. Directives: ::: takeaway/ob-board/lab/callout/walkthrough/provenance/pq+answer, %%term%% -> <dfn>. Local images embedded as base64; mermaid -> inline SVG via mmdc (fallback: code block + warn); --check QA gates (no em dash/gradient/external dep/TODO, no personal identifiers).
- build/ds.css: canonical stylesheet, font stack now "Atkinson Hyperlegible", "Inter", system fallbacks. build/ds.js: canonical script (patched).
- build/README.md: directive reference + worker instructions (cheap runners zclade/clade-mini for mechanical work).
- Fonts installed on VM (~/.fonts, fc-cache): Atkinson Hyperlegible (4 styles, OFL) + Inter variable (OFL). Chromium renders real font; print-to-PDF embeds it. No webfont links anywhere (offline-safe).
- mermaid-cli (mmdc) installing in background (proc_b89c6350f10c).
- Both remaining workers (Vols 4+5, Vols 11+13+14) switched to MD-first for all new content; already-written HTML chapters: convert via cheap runner or keep + pass gates, their call. Screenshot-verify at 1440/768/390px mandated.
- Pipeline smoke test: build/test-smoke.md -> test-smoke.html, all directives verified rendering.

## 2026-09-28 ~15:05 EDT — Mermaid→SVG path verified
- mmdc installed locally at ~/workspace/tools/mermaid (global npm tree was wiped; /usr/bin/chrome entries were phantom; real browser is /opt/meta-chromium/chrome v152).
- build/puppeteer.json points mmdc at the real Chrome (--no-sandbox). build.py find_mmdc() checks PATH + local install; renders wrapped in <figure class="diagram"> with inline SVG.
- Smoke test: mermaid block -> inline SVG (212 KB page), labels verified, no fallback. Graceful code-block fallback retained if render ever fails.

## 2026-09-28 ~15:15 EDT — MASTER DIRECTIVE (user loves the preview)

Deep build with an army of bots. Strict order:
1. BASE curriculum finished completely (gap-checked against real RE job postings + interview reports, first principles, russian-doll). Must cover: LLM RE generalist, post-training/alignment (RLHF/RL), inference & serving, distributed training systems, evals & safety, agent systems.
2. ROLE TRACKS derived from base (target role + base reading order + role-specific deep dives).
3. CRASH COURSE versions (20% that answers 80% of real questions).
4. INTERVIEW PREP packs last.

REPO DISCIPLINE: git repo = professional KNOWLEDGE BASE only, work-safe. Zero "interview" framing in filenames, titles, READMEs, content. Repo name: llm-knowledge-base (public). Interview prep NEVER enters the repo: ships only as separate zips, outside git. Two streams: public knowledge-base repo vs private interview-prep zips.
IMAGES: media-namespace generation first (until rate limits) -> OpenRouter muse-image CLI -> internet-sourced -> Mermaid SVG -> ASCII -> manim/hyperframe. Every text-heavy section gets a visual; verify rendering.
Standing gates: markdown-first, one canonical ds.css/ds.js, 3-width screenshots, humanizer --check, zero PII, combined hyperlinked PDF per stream.

Spawned 2026-09-28 ~15:15: researcher A (role-asks.md from 180 roles + fresh 2026 postings), researcher B (interview-asks.md from real candidate reports). Gap analysis gates all writer waves.
Still running: vols 4+5 writer (744a1610).

## 2026-09-28 ~15:20 EDT — WAVE 1 (base gap-fill): 7 writers spawned
Both research reports in (build/research/role-asks.md, build/research/interview-asks.md). /tmp/roles.json confirmed gone (tmpfs wipe); role-asks built from skill-universe map + 7 fresh Sept 2026 searches — accepted.
Workers (all MD-first, media-images-first, humanizer --check, 3-width screenshots, work-safe repo framing, scratch under build/scratch/ since /tmp is full):
- W1 new Vol 15: 15-gpu-kernels.md (GPU execution model, Triton kernel lab, FP8, torch.compile, 2:4 sparsity)
- W2 appendix-06a-reliability.md (async checkpointing, loss-spike rollback, NCCL/RDMA, stragglers)
- W3 appendix-06b-parallelism-xla.md (context/expert parallelism depth, TPU/XLA comparative)
- W4 appendix-07a-rl-systems.md + appendix-07b-posttraining-data.md (RL systems, post-training data pipelines, interpretability basics; verify CAI/LoRA drills)
- W5 appendix-08a-inference-economics.md (routing/cascades, shadow traffic, disaggregation economics; verify FP8/KV-quant/MoE-serving/LoRA)
- W6 appendix-09a-agent-evals.md + appendix-09b-agent-security.md (agentic eval engineering, capability security, red-teaming; verify MCP/A2A/LangGraph/harness/CI-CD)
- W7 verify-and-patch sweep (V11 eval stats, V13 classic distributed systems, V5 mid-training, V4 long-context, V11 labeler protocols)
Appendices build standalone; assembly merges into parent volumes. Still running: vols 4+5 writer.

## 2026-09-28 ~15:25 EDT — CODE STANDARD (user hard rule)
ALL code samples must be Python, no other languages anywhere. Every sample: thorough comments (what/why/what-breaks + complexity notes). Non-trivial samples get plain-English walkthroughs. Key flows pair code with a visual (media-generated image / Mermaid / ASCII, accurate to the code).
Implemented in build.py: code_standard_check_md() — hard FAIL on fenced blocks tagged with any non-Python language (rust/cpp/cuda/js/go/java/...); WARN on Python samples >15 lines with <8% comment lines; WARN on suspicious untagged blocks (heuristic signatures); bash/text/ascii/mermaid tags exempt. Wired into build_one() and --check (both .md and .html). README QA checklist updated. Existing volumes scanned: zero non-Python <pre> blocks found. Broadcast to all 8 running writers (7 Wave-1 + vols 4+5).

## 2026-09-28 ~19:35 EDT — hamburger/eyebrow overlap fix (canonical ds.css)
Vols 4+5 worker flagged a known overlap: at <=56rem with the nav drawer open, the fixed hamburger (.ds-menu-btn, z-index 55) sat ABOVE the drawer (.ds-sidebar, z-index 50) and overlapped the sidebar eyebrow text. Fix: body.ds-nav-open .ds-sidebar now z-index 56. Screenshot-verified via Playwright at 768px (drawer open: eyebrow fully readable; closed: button visible, no overlap) and 390/1440 closed. All .md-based builds inherit the fix automatically (CSS inlined fresh each run). Rebuilt top-level preview index.html + 00-how-to-use.html with the fix (code + QA PASS). Note: the preview zip handed off at 19:06 now differs by this one CSS line; older inlined HTML volumes (vols 1-14, dsa) will pick the fix up at final assembly rebuild.

## 2026-09-28 ~19:45 EDT — W5 (Vol 8 supplement) landed; bare-img overflow fixed canonically
appendix-08a-inference-economics.md (55.6 KB) + built HTML (1.8 MB, standalone). 8 chapters: cost-per-outcome, routing/cascades, shadow traffic, PD disaggregation economics + patches for FP8, KV-quant, MoE serving, multi-tenant LoRA. Gate PASS, 3-width screenshots, zero JS errors. Code standard honored natively (8 python blocks, ~30% comment density, 16 walkthroughs).
Worker found a real pipeline defect: md emits bare <img> in <p> but ds.css only constrained figure img, so all 8 images overflowed at 1440px. Fixed canonically in ds.css (.ds-content img{max-width:100%;height:auto}); worker's wrap-fig-08a.py workaround no longer needed. Note: 08a HTML was built with its workaround; at assembly rebuild the canonical CSS makes it redundant (harmless).

## 2026-09-28 ~19:55 EDT — W1 (Vol 15 GPU kernels) landed + watermark hygiene applied
15-gpu-kernels.md (62 KB, 7 chapters + 6 practice Qs + provenance) + built HTML (1.8 MB standalone). 5 generated images (img15/) + 3 Mermaid flowcharts as inline SVG. All gates PASS zero warnings. Code standard native: 10 python blocks (Triton), 34-61% comment density, ASCII diagrams in text tags. 7 CPU labs executed and pass (softmax max abs err 1.49e-08; FP8 per-tensor 100.0% erased vs block-128 3.2%; 2:4 50% nonzero valid). Triton GPU kernels not executed (no GPU on box), noted in volume.
Worker shared-pipeline changes: build/mermaid-config.json added (dark theme, wrappingWidth 380) + render_mermaid_src --size 1400 (fixes squeezed 168px nodes; strict improvement for all volumes). Reviewed: sane, no regression risk.
Watermark hygiene (standing order): wm_clean.py Layer A over .md (clean, no changes) and 5 PNGs (metadata stripped; swapped cleaned versions in as canonical, rebuilt HTML with them embedded, --check PASS).

## 2026-09-28 ~20:00 EDT — W4 (Vol 7 appendices) + W3 (Vol 6 appendix B) landed; hygiene applied
appendix-07a-rl-systems.md (36 KB) + HTML (527 KB): rollout throughput, inference engines as rollout workers, async RL + worker memory.
appendix-07b-posttraining-data.md (89 KB) + HTML (723 KB): synthetic data, verifier/reward pipelines, SFT mixtures + rejection sampling, preference-data collection, interpretability overview, SAEs, probing vs steering, limits, B.9 LoRA/QLoRA patch with memory drills (7B: 84GB full / 14.5GB LoRA / 4GB QLoRA).
appendix-06b-parallelism-xla.md (46.7 KB) + HTML (913 KB): context parallelism depth, expert parallelism, all-to-all byte-math, TPU topology/pods, XLA vs eager + JAX sharding. 4 webp images + 5 inline Mermaid SVGs.
All gates PASS (md incl. code-standard gate, html). Screenshots 1440/768/390, zero JS errors. Code standard native: Python-only, comment density met, walkthroughs on non-trivial samples.
Verification findings: Constitutional AI fully covered in Vol 7 Ch.7 (no patch needed); LoRA/QLoRA had zero real mentions (base64 noise only) -> B.9 written. TPU v6e ICI bandwidth sources conflict (1,600 vs 3,200 Gb/s) -> used Google "~2x v5e" only, marked UNVERIFIED.
Watermark hygiene: Layer A over 3 .md (clean, no changes) + 14 images (metadata stripped, swapped in as canonical); all 3 HTMLs rebuilt with cleaned images embedded, --check PASS.
Worker env note: chrome --screenshot/--dump-dom CLI flags silently produce nothing in this container; use puppeteer over CDP; TMPDIR=/home/hatch/.ctmp needed (socket path limit).

## 2026-09-28 ~20:05 EDT — W6 (Vol 9 supplements) landed; shared QA fixes reviewed; hygiene applied
appendix-09a-agent-evals.md (-> 270 KB HTML): task harnesses, graders/rubrics, judge calibration vs humans, lucky-pass mitigation (pass@k + trajectory checks + auto-generated evals), long-horizon evals & drift patch, agent CI/CD with eval gates patch.
appendix-09b-agent-security.md (-> 502 KB HTML): capability-level security, red-teaming practice (Garak/PyRIT/OWASP LLM Top 10 2025), PII handling, MCP server-building mechanics patch, A2A patterns patch, LangGraph depth patch.
Facts web-verified Sept 2026 (MCP spec 2025-11-25, stdio + Streamable HTTP, HTTP+SSE deprecated 2025-03-26; A2A agent card + JSON-RPC lifecycle; OWASP LLM Top 10 2025; Garak NVIDIA / PyRIT Microsoft; pass@k Chen et al. 2021; LangGraph StateGraph/reducers/checkpointers/interrupt). UNVERIFIED markers kept: Garak probe counts, PyRIT internals, LangGraph minor APIs, CI gate delta/n starting points.
Gates PASS all four files (md incl. code gate, html). 14 python + 12 mermaid blocks, heavy commenting, walkthrough beside every non-trivial pair. Screenshots 1440/768/390, no defects (1618 scrollWidth at 768/390 is off-canvas drawer only).
Worker shared-pipeline fixes (reviewed, kept): (1) qa_check strips <svg> before identifier scan (mermaid path coords false-positived the phone regex); (2) fenced code stripped for .md identifier scan (attacker@evil.com example data false-positive). Both sane, smoke test passes.
Watermark hygiene: Layer A over 2 .md (clean, identical) + 2 hero webp (metadata stripped, swapped in); both HTMLs rebuilt with cleaned images embedded, --check PASS.

## 2026-09-28 ~20:10 EDT — W2 (Vol 6 appendix A, reliability) landed; hygiene applied
appendix-06a-reliability.md -> 764 KB standalone HTML. 4 chapters: checkpointing + Daly cadence math (worked 1000-GPU example); failure modes (loss-spike detector + rollback, NaN playbook, collective-hang debugging); stragglers (MAD detection, mitigation, autoretry stats); networking (NCCL recap, ring 2(N-1)/N, GPUDirect RDMA, NVLink-vs-IB math). All code Python-only, heavily commented, walkthroughs on non-trivial samples; 9 CPU labs tested working (numpy/matplotlib). Visuals: 2 generated images (Daly U-curve, scale-up vs scale-out fabric), 2 Mermaid flowcharts (hang-debug, ring all-reduce), 1 ASCII timeline. Provenance box with honest UNVERIFIED markers.
Gates: --check PASS zero warnings (fixed "robust", split 12 long sentences). Screenshots 1440/768/390, no defects, no overflow.
Worker note: mermaid gantt/timeline SVGs false-positive the phone heuristic — already fixed canonically by W6 (qa_check strips <svg> before identifier scan); worker used ASCII timeline instead, fine. Confirmed ds.css img fix (mine) resolved their overflow.
Watermark hygiene: Layer A over .md (identical, clean) + 2 webp (metadata stripped, swapped in); HTML rebuilt with cleaned images, --check PASS.

## 2026-09-28 ~20:15 EDT — W7 (verify-and-patch sweep) landed; WAVE 1 COMPLETE
Verification verdicts (evidence-based): Vol 11 eval stats PARTIAL (paired tests/bootstrap/p-values/CIs/SE covered; missing A/B design for non-deterministic outputs + autorater calibration); Vol 13 DS primitives THIN (no consistent hashing, exactly-once, backpressure, fair scheduling, idempotency, ordered messaging); Vol 5 mid-training THIN (no QA pre-mix, contamination, decontamination); Vol 4 long-context THIN (RoPE covered; YaRN passing mention; NTK-aware/H2O/StreamingLLM/sliding window/KV eviction absent, Vol 8 same gap); labeler protocols PARTIAL (rubrics/70% agreement/pitfalls covered; missing disagreement handling + pool diversity).
5 patch files (all .md -> .html, gates PASS, screenshots 1440/768/390 no defects, all-Python ~40% comments, walkthroughs):
- appendix-11a-eval-statistics: A/B design for non-deterministic outputs (3 noise layers, paired design, 3-5 repeats, MDE, Wilson interval, Bonferroni, cluster bootstrap, paired-bootstrap Python); autorater calibration (6-step loop, Cohen's kappa worked 100-item example k=0.375, Landis-Koch, human ceiling, bias traps, kappa Python).
- appendix-11b-preference-labeler-protocols: instruction packet, gold-item qualification (30-50 items, 80% bar, 5% seeding), 4-rung adjudication ladder, majority-vote binomial Python (0.75 -> 0.844 majority-of-3), pool diversity levers.
- appendix-13a-distributed-systems: consistent hashing (ring, virtual nodes, 1M-key/100-node numbers, Python ring); delivery semantics + idempotency (keys, sequence numbers, Kafka transactions, Python idempotent consumer); backpressure (bounded queues, Little's Law sizing, demand signaling, Python bounded queue); fair multi-tenant scheduling + ordered messaging (WRR, DRR, DRF worked 8 CPU/32 GB, per-partition ordering, Python smooth WRR).
- appendix-05a-mid-training: definition, lab-side vs practitioner-side variants, annealing insight + microanneal, QA pre-mix -> RL readiness, 3 post-decontamination contamination doors, mixture budget + n-gram overlap Python.
- appendix-04a-long-context-engineering: RoPE extension (PI, NTK-aware b'=b*s^(D/(D-2)), YaRN NTK-by-parts a=1/b=32 + temp 1/t=0.1*ln s+1, Python); KV eviction (attention sinks, sliding window, StreamingLLM, H2O, KV-memory Python + H2O simulation). Key numbers verified in Python: NTK-aware theta (b=10000,D=128): s=2->20221.3, s=4->40889.9, s=8->82685, s=16->167198.7; YaRN 1/t: s=8->1.2079, s=16->1.2773, s=32->1.3466; KV: 32 layers x d_model 4096 fp16 = 0.5MB/token -> 64GB at 128k.
Facts grounded via web search Sept 28 (YaRN arXiv:2309.00071; NTK-aware bloc97; H2O NeurIPS 2023; StreamingLLM 2023; consistent hashing Karger 1997; Cohen/Landis-Koch; DRF; OLMo 2 microanneal). UNVERIFIED markers on practitioner rules-of-thumb.
Worker fragility warning: mermaid SVGs whose coords match phone regex false-positive -- ALREADY fixed canonically (qa_check strips <svg> before identifier scan); worker used ASCII for 4 blocks in 11a/11b, fine.
Watermark hygiene: Layer A over 5 .md (all identical, clean). No images in these files (ASCII + mermaid only).
Flags for assembly: "interview" matches in appendix-06b + 09a (+1 banned-tell each), 1 banned-tell in 06a -> repo-discipline flavor.py pass.
WAVE 1 COMPLETE: all 18 children done.

## 2026-09-28 ~20:10 EDT — Wave 2 Phase B prep: flavor.py built + tested
Worker built build/flavor.py: deterministic interview-framing -> knowledge-base transform (visible text only; never touches pre/code/svg/comments/URLs; idempotent; self-enforces humanizer gate on replacement strings). Handles: "On the board" box labels -> "Design prompt", "[INTERVIEW]" tags -> "[FIELD NOTE]", renames 14-behavioral-research.html -> 14-communicating-research.html with link fixups, exits nonzero on residual visible "interview". Tested: appendix HTMLs (1 replacement, clean), synthetic edge cases (all 9 rule types), rename machinery, --all --dry-run on top-level volumes (214 replacements projected, matches Wave 1 flags exactly). Manual-review list (291 visible leftovers) at /tmp/flavor-test/manual-review-list.txt.
APPROVED mappings for Phase B worker: "INTERVIEW RELEVANCE" tag -> "FIELD RELEVANCE"; "Interview line." -> "Key line."; "Interview answer." -> "Worked answer."; "interview trap" -> "classic trap"; blanket "interviewers" -> "reviewers" (case-preserving); "the interviewer is ..." -> per-sentence rephrase to reviewer framing; "interview day"/"interview-ready"/"interview version"/"interview ROI" -> per-instance rephrase; product names ("RRK Interview Q&A Bank v2", "FDE interview materials", "Interview sentence bank") KEEP as proper nouns; Vol 14 "Behavioral rounds" -> reframe as communication rounds; dsa-track-300.html EXCLUDED from flavor.py (separate track, keep as-is).
Pipeline fix applied: build.py COMPONENT_DEFAULTS ob-board default title "On the board" -> "Design prompt" so future .md rebuilds emit work-safe labels (smoke test PASS).

## 2026-09-28 ~21:30 EDT — Design standard update: Charis SIL + site navigation
User directive ("Charles SSIL" = Charis SIL, SIL International OFL font). Two canonical changes applied to build/ds.css + build/build.py + all 39 existing HTML files:
1. FONT: Charis SIL (400/400italic/700/700italic, latin + latin-ext subsets) downloaded from Google Fonts as woff2, bundled in fonts/ (8 files, ~370KB total), wired via @font-face with font-display:swap, fully offline-safe (no CDN). Set as --font-body for body text AND headings; new --font-ui (system sans stack) for tiny nav labels only (sidebar, buttons, breadcrumbs, sitemap).
2. NAVIGATION: every page now has a sticky site header (LLM Knowledge Base home link, prev/next across the canonical 39-page order, breadcrumbs Home/Section/Page, collapsible "Browse all volumes and tracks" details nav) and a footer with the full sitemap grouped by section (Start here, Base volumes, Role tracks, DSA track, Crash packs). Zero orphans: index links to all pages within one click; every page links back via header + footer.
Implementation: build.py gained SITE_MAP + site_header_html()/site_footer_html(); SIDEBAR_SKELETON emits header/footer per page. build/apply-site-update.py applied the CSS swap + chrome injection idempotently to all existing HTMLs. Rendering fix: re-added p/li/td/th code overflow-wrap (was a vol06-local fix lost in the CSS swap; caught a 5px mobile overflow). Mobile fix: site header clears the fixed hamburger button.
Verification: all 39 files PASS build.py --check (humanizer, python-only, no-external, no-gradient, no-identifiers); 90 page-widths (30 files x 1440/768/390) screenshot-verified: zero overflow, Charis SIL confirmed loaded via document.fonts, zero JS errors. Pushed to jrajath94/llm-knowledge-base (39 HTML + 8 woff2 + 3 build files).
Note for in-flight crash workers: build.py template now emits the new chrome automatically; crash-track-01/04 already rebuilt with it. crash-base.html (not yet built) will get it on build.

## 2026-09-28 ~21:45 EDT — Crash packs complete (all 11) + design update applied
All 8 remaining crash workers delivered (tracks 01, 03, 04, 06, 07, 08, 09 + crash-base). Design update verified on all 11 crash packs (33 page-widths: zero overflow, header/footer present, Charis SIL loaded, zero JS errors). crash-base.html needed the chrome injection (had new CSS but old template build); fixed via apply-site-update.py. Added "Crash packs" section to index.html linking all 11 packs (gates PASS). Pushed to jrajath94/llm-knowledge-base: 11 crash HTMLs + updated index.html (0 failed). All 32 Wave 2 workers now complete.
