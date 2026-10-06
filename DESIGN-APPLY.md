# Design System Retrofit — Instructions for Writer Workers

A user-mandated design system now applies to ALL HTML volumes. Two shared files are the source of truth (do not improvise your own theme):

- `~/workspace/your_files/research-engineer-curriculum/design-system.css`
- `~/workspace/your_files/research-engineer-curriculum/design-system.js`

## What you must do to your volume(s) before reporting done

1. **Inline the CSS**: copy the entire contents of `design-system.css` into a `<style>` block in your `<head>`, REPLACING your old `<style>` block. Delete your old theme CSS. Keep any volume-specific CSS only if it does not conflict (it must use the tokens; no new colors, no gradients).
2. **Inline the JS**: copy the entire contents of `design-system.js` into a `<script>` block at the end of `<body>`. Do not modify it.
3. **Restructure the body** to this skeleton (keep all your content, just wrap it):
   ```html
   <body>
     <div class="ds-progress" id="dsProgress"></div>
     <button class="ds-menu-btn" id="dsMenuBtn" aria-label="Open navigation">☰</button>
     <aside class="ds-sidebar" id="dsSidebar" aria-label="Volume navigation">
       <div class="ds-side-head">
         <p class="ds-side-eyebrow">Research-Engineer Curriculum</p>
         <p class="ds-side-title">[YOUR VOLUME TITLE]</p>
         <input class="ds-side-search" id="dsNavSearch" type="search" placeholder="Filter sections..." aria-label="Filter sections">
       </div>
       <nav class="ds-nav" id="dsNav" aria-label="Sections"></nav>
       <div class="ds-side-foot"><span id="dsCompleteCount"></span></div>
     </aside>
     <div class="ds-scrim" id="dsScrim"></div>
     <main class="ds-content" id="dsContent">
       [ALL YOUR EXISTING CONTENT HERE]
     </main>
     <script>/* design-system.js inlined */</script>
   </body>
   ```
   The JS auto-builds the sidebar nav from your `h2` (chapter groups) and `h3` (items), auto-generates missing heading ids, and adds scroll-spy, search filter, localStorage completion checkmarks, progress bar, chapter prev/next footers, copy buttons, and keyboard nav. You do not hand-build the nav.
4. **First-use terms**: wrap first-use terms in `<dfn>` (gets the accent dotted-underline definition style).
5. **Practice questions**: where you have Q&A, use the `.pq` widget markup so answers are click-to-reveal with why-each-option-is-right/wrong:
   ```html
   <div class="pq">
     <p class="pq-q"><strong>Q1.</strong> Question text</p>
     <button class="pq-reveal" aria-expanded="false">Reveal answer</button>
     <div class="pq-a" hidden>
       <p><strong>Answer: B.</strong> Explanation.</p>
       <ul><li><span class="pq-opt-wrong"><strong>A, wrong:</strong></span> why</li>
       <li><span class="pq-opt-right"><strong>B, right:</strong></span> why</li></ul>
     </div>
   </div>
   ```
6. **"On the board" boxes**: keep them, they now render in the amber `.ob-board` style. If yours used a different class, switch to `class="ob-board"` with an `<h4>On the board</h4>`.
7. **Key takeaways**: use `class="takeaway"` with an `<h4>Key takeaways</h4>`.
8. **"Last verified" provenance box** (from the quality gate): use `class="provenance"`, e.g.
   ```html
   <div class="provenance"><strong>Last verified: September 2026.</strong> Live-verified: [list]. <span class="uv">UNVERIFIED:</span> [list, or "none"].</div>
   ```
9. **Labs**: wrap in `class="lab"` with an `<h4>` title.

## What you must NOT do

- Do not invent your own colors, fonts, or theme. Tokens are exact: `--bg:#000000; --surface:#0d0d0d; --border:#262626; --text:#ececec; --muted:#a8a8a8; --accent:#f0b429; --link:#6cb2ff; --code-bg:#111111`.
- Do not add gradients, decorative filler, webfonts, or external dependencies. System font stack only.
- Do not rewrite your content for the retrofit; only re-skin and re-structure. Your chapters, depth, images, and labs stay as they are.

## New quality gates (add to your QA checklist)

- [ ] Mobile: sidebar becomes a drawer, no page-level horizontal scrolling at 360px width.
- [ ] Accessibility: visible focus states (built in), semantic heading order (h1 once, then h2/h3), contrast from the tokens.
- [ ] Print stylesheet present (built into the CSS; verify print preview hides nav).
- [ ] Sidebar nav builds correctly (open the file, check groups/items/checkmarks/search).
- [ ] wm_clean.py Layer A still clean after the retrofit.
