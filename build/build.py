#!/usr/bin/env python3
"""Markdown-first curriculum build: MD -> self-contained HTML.

Usage:
    python3 build.py volume-04.md volume-05.md
    python3 build.py --all            # build every *.md in sources/
    python3 build.py --check out.html # QA checks only

Pipeline: .md sources (one per volume) -> build.py -> single self-contained
.html with ONE canonical ds.css + ds.js inlined at build time. Zero external
dependencies: local images are embedded as base64 data URIs; mermaid blocks
are rendered to inline SVG when mmdc is available.

Learning components are Markdown directives (fenced ::: blocks):

    ::: takeaway
    - point one
    :::

    ::: ob-board Optional title
    Open question for the reader...
    :::

    ::: lab Lab 4.1: Title here
    Lab body...
    :::

    ::: callout warn
    Warning text...
    :::
    (callout kinds: warn, interview, or plain)

    ::: walkthrough
    Numbered steps for reading the figure above...
    :::

    ::: provenance
    **Last verified: September 2026.** ...
    :::

    ::: pq
    **Q1.** Question stem?
    A. option
    B. option
    ::: answer
    **Answer: B.** Why.
    - **A, wrong:** reason
    - **B, right:** reason
    :::
    :::

Inline first-use terms: %%gradient descent%% -> <dfn>gradient descent</dfn>.

Front matter (optional, at top of file):
    ---
    title: LLM Internals
    eyebrow: Volume 4
    ---
"""
import base64
import html as htmlmod
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DS_CSS = os.path.join(HERE, "ds.css")
DS_JS = os.path.join(HERE, "ds.js")
SOURCES = os.path.join(HERE, "sources")

try:
    import markdown as md_lib
except ImportError:
    sys.exit("ERROR: pip install markdown  (pip install --break-system-packages markdown)")

# ---------------------------------------------------------------- directives

COMPONENT_DEFAULTS = {
    "takeaway": ("takeaway", "Key takeaways"),
    "ob-board": ("ob-board", "Design prompt"),
    "walkthrough": ("walkthrough", "How to read this diagram"),
    "lab": ("lab", "Lab"),
    "provenance": ("provenance", None),
    "callout": ("callout", None),
}

DFN_TOKEN = "\ue000dfn%d\ue001"


class Node:
    def __init__(self, kind, title=None):
        self.kind = kind          # directive name or "root"/"answer"
        self.title = title        # optional custom title
        # ordered flow: ("text", [lines]) and ("node", Node) in source order
        self.flow = []
        self._buf = []

    def _flush(self):
        if self._buf:
            self.flow.append(("text", self._buf))
            self._buf = []


def parse_directives(lines):
    """Stack-based ::: fenced-block parser. Returns root Node.

    Directives keep their source position: flow interleaves text segments
    and directive nodes in document order.
    """
    root = Node("root")
    stack = [root]
    for ln in lines:
        if re.match(r"^\s*:::\s*answer\s*$", ln):
            stack[-1]._flush()
            node = Node("answer")
            stack[-1].flow.append(("node", node))
            stack.append(node)
            continue
        m = re.match(r"^(\s*):::\s*(\w[\w-]*)\s*(.*)$", ln)
        if m:
            stack[-1]._flush()
            node = Node(m.group(2), m.group(3).strip() or None)
            stack[-1].flow.append(("node", node))
            stack.append(node)
            continue
        if re.match(r"^\s*:::\s*$", ln):
            if len(stack) > 1:
                stack[-1]._flush()
                stack.pop()
            else:
                stack[-1]._buf.append(ln)  # stray closer, keep literally
            continue
        stack[-1]._buf.append(ln)
    for n in stack:
        n._flush()
    return root


def md_to_html(text):
    return md_lib.markdown(text, extensions=["fenced_code", "tables", "toc"])


def render_flow(node):
    """Render a node's flow (text segments + directive nodes) in source order."""
    parts = []
    for typ, item in node.flow:
        if typ == "text":
            parts.append(md_to_html("\n".join(item).strip("\n")))
        else:
            parts.append(render_node(item))
    return "".join(parts)


def render_node(node):
    if node.kind == "root":
        return render_flow(node)
    q_parts = []
    ans_html = ""
    for typ, item in node.flow:
        if typ == "text":
            q_parts.append(md_to_html("\n".join(item).strip("\n")))
        elif item.kind == "answer":
            a_inner = render_flow(item)
            ans_html += '<div class="pq-a" hidden>' + a_inner + "</div>"
        else:
            q_parts.append(render_node(item))
    body = "".join(q_parts)

    if node.kind == "pq":
        return ('<div class="pq"><div class="pq-q">' + body + "</div>"
                '<button class="pq-reveal" aria-expanded="false">Reveal answer</button>'
                + ans_html + "</div>")
    if node.kind in COMPONENT_DEFAULTS:
        cls, default_title = COMPONENT_DEFAULTS[node.kind]
        if node.kind == "callout" and node.title:
            cls = "callout " + node.title
            title_html = ""
        else:
            title = node.title or default_title
            title_html = "<h4>" + htmlmod.escape(title) + "</h4>" if title else ""
        return '<div class="' + cls + '">' + title_html + body + "</div>"
    # unknown directive: render as plain div with its name as class
    return '<div class="' + htmlmod.escape(node.kind) + '">' + body + "</div>"


# ---------------------------------------------------------------- preprocess

def extract_fenced_blocks(text):
    """Pull out ``` code blocks; return (text_with_placeholders, [blocks])."""
    blocks = []

    def repl(m):
        blocks.append(m.group(0))
        return "\ue010code%d\ue011" % (len(blocks) - 1)

    text = re.sub(r"```.*?```", repl, text, flags=re.DOTALL)
    return text, blocks


def restore_fenced_blocks(html_text, blocks):
    for i, b in enumerate(blocks):
        if b is None:
            continue  # mermaid blocks use their own token path
        m = re.match(r"```(\w*)\n?(.*?)```$", b, flags=re.DOTALL)
        lang, code = (m.group(1), m.group(2)) if m else ("", b)
        code_html = htmlmod.escape(code)
        cls = ' class="language-%s"' % lang if lang else ""
        html_text = html_text.replace(
            "\ue010code%d\ue011" % i, "<pre><code%s>%s</code></pre>" % (cls, code_html))
    return html_text


def embed_local_images(text, src_dir, warnings):
    """Embed local ![alt](path) images as base64 data URIs."""

    def repl(m):
        alt, path = m.group(1), m.group(2).strip()
        if path.startswith(("http://", "https://", "data:")):
            if path.startswith("http"):
                warnings.append("remote image left as-is (breaks offline): %s" % path[:80])
            return m.group(0)
        full = os.path.normpath(os.path.join(src_dir, path))
        if not os.path.exists(full):
            warnings.append("MISSING image file: %s" % path)
            return m.group(0)
        mime, _ = mimetypes.guess_type(full)
        data = base64.b64encode(open(full, "rb").read()).decode()
        return "![%s](data:%s;base64,%s)" % (alt, mime or "image/png", data)

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, text)


def apply_dfn_placeholders(text):
    """%%term%% -> placeholder tokens (restored to <dfn> after md conversion)."""
    terms = []

    def repl(m):
        terms.append(m.group(1))
        return DFN_TOKEN % (len(terms) - 1)

    return re.sub(r"%%([^%\n]+)%%", repl, text), terms


def restore_dfn(html_text, terms):
    for i, t in enumerate(terms):
        html_text = html_text.replace(
            DFN_TOKEN % i, "<dfn>" + htmlmod.escape(t) + "</dfn>")
    return html_text



# ---------------------------------------------------------------- site map
# Canonical site order for header nav, footer sitemap, breadcrumbs, prev/next.
# (filename, nav title, section)
SITE_MAP = [
    ("index.html", "Home", "Start here"),
    ("00-how-to-use.html", "How to Use This Curriculum", "Start here"),
    ("01-math-for-ml.html", "Math for ML", "Base volumes"),
    ("02-ml-foundations-bridge.html", "ML Foundations", "Base volumes"),
    ("03-deep-learning-for-researchers.html", "Deep Learning for Researchers", "Base volumes"),
    ("04-llm-internals.html", "LLM Internals", "Base volumes"),
    ("05-pretraining.html", "Pre-training", "Base volumes"),
    ("06-distributed-training.html", "Distributed Training", "Base volumes"),
    ("07-post-training-rl.html", "Post-training and RL", "Base volumes"),
    ("08-inference-serving-bridge.html", "Inference and Serving", "Base volumes"),
    ("09-agents-rag-guide.html", "Agents and RAG", "Base volumes"),
    ("10-productionizing-mlops.html", "Productionizing and MLOps", "Base volumes"),
    ("11-research-methods.html", "Research Methods", "Base volumes"),
    ("12-paper-spine.html", "Guided Paper Spine", "Base volumes"),
    ("13-ml-system-design.html", "ML System Design", "Base volumes"),
    ("14-communicating-research.html", "Communicating Research", "Base volumes"),
    ("15-gpu-kernels.html", "GPU Kernels", "Base volumes"),
    ("track-01-llm-research-engineer.html", "LLM Research Engineer", "Role tracks"),
    ("track-02-post-training-alignment.html", "Post-training / Alignment", "Role tracks"),
    ("track-03-inference-serving.html", "Inference and Serving", "Role tracks"),
    ("track-04-distributed-training-systems.html", "Distributed Training Systems", "Role tracks"),
    ("track-05-evals-safety.html", "Evals and Safety", "Role tracks"),
    ("track-06-agent-systems.html", "Agent Systems", "Role tracks"),
    ("track-07-forward-deployed-engineer.html", "Forward Deployed Engineer", "Role tracks"),
    ("track-08-ai-engineer.html", "AI Engineer", "Role tracks"),
    ("track-09-ml-engineer.html", "ML Engineer", "Role tracks"),
    ("track-10-software-engineer.html", "SWE / Backend Systems", "Role tracks"),
    ("dsa-track-300.html", "DSA Track: Top-300", "DSA track"),
    ("crash-base.html", "Crash Pack: Base Curriculum", "Crash packs"),
    ("crash-track-01.html", "Crash: LLM Research Engineer", "Crash packs"),
    ("crash-track-02.html", "Crash: Post-training / Alignment", "Crash packs"),
    ("crash-track-03.html", "Crash: Inference and Serving", "Crash packs"),
    ("crash-track-04.html", "Crash: Distributed Training", "Crash packs"),
    ("crash-track-05.html", "Crash: Evals and Safety", "Crash packs"),
    ("crash-track-06.html", "Crash: Agent Systems", "Crash packs"),
    ("crash-track-07.html", "Crash: Forward Deployed Eng.", "Crash packs"),
    ("crash-track-08.html", "Crash: AI Engineer", "Crash packs"),
    ("crash-track-09.html", "Crash: ML Engineer", "Crash packs"),
    ("crash-track-10.html", "Crash: SWE / Backend", "Crash packs"),
]
SITE_SECTIONS = ["Start here", "Base volumes", "Role tracks", "DSA track", "Crash packs"]

_FILE_IDX = {f: n for n, (f, _, _) in enumerate(SITE_MAP)}

# Hand-drawn chevrons (inline SVG, no emoji, no icon fonts).
_CHEV_L = ('<svg class="chev chev-l" viewBox="0 0 12 12" aria-hidden="true" focusable="false">'
           '<path d="M7.6 2.4 4 6l3.6 3.6" fill="none" stroke="currentColor" '
           'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>')
_CHEV_R = ('<svg class="chev chev-r" viewBox="0 0 12 12" aria-hidden="true" focusable="false">'
           '<path d="M4.4 2.4 8 6l-3.6 3.6" fill="none" stroke="currentColor" '
           'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>')
_CHEV_D = ('<svg class="chev chev-d" viewBox="0 0 12 12" aria-hidden="true" focusable="false">'
           '<path d="M2.4 4.4 6 8l3.6-3.6" fill="none" stroke="currentColor" '
           'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>')

def _site_nav_groups(current):
    """Numbered volume list grouped by section, for the header drawer/panel."""
    parts = []
    for sec in SITE_SECTIONS:
        items = [(f, t) for (f, t, s) in SITE_MAP if s == sec]
        if not items:
            continue
        lis = []
        for f, t in items:
            cls = ' class="site-nav-current"' if f == current else ""
            num = _FILE_IDX[f] + 1
            lis.append('<li><a href="%s"%s><span class="site-nav-num">%02d</span>'
                       '<span class="site-nav-title">%s</span></a></li>'
                       % (f, cls, num, htmlmod.escape(t)))
        parts.append("<div><h4>%s</h4><ul>%s</ul></div>" % (sec, "".join(lis)))
    return "".join(parts)

def _foot_index_groups(current):
    """Grouped sitemap with real hierarchy for the footer."""
    parts = []
    for sec in SITE_SECTIONS:
        items = [(f, t) for (f, t, s) in SITE_MAP if s == sec]
        if not items:
            continue
        lis = []
        for f, t in items:
            cls = ' class="foot-current"' if f == current else ""
            lis.append('<li><a href="%s"%s>%s</a></li>' % (f, cls, htmlmod.escape(t)))
        parts.append('<div class="foot-group"><h4>%s</h4><ul>%s</ul></div>'
                     % (sec, "".join(lis)))
    return "".join(parts)

def site_header_html(current, section, page_title):
    """N6 newspaper-masthead header: issue line, wordmark + volume sequence,
    breadcrumb row + volume-nav toggle, collapsible panel (drawer on mobile),
    double rule below."""
    idx = _FILE_IDX.get(current, -1)
    if idx >= 0:
        total = len(SITE_MAP)
        issue = "No. %02d of %d" % (idx + 1, total)
        if idx > 0:
            pf, pt = SITE_MAP[idx - 1][0], SITE_MAP[idx - 1][1]
            prev_html = ('<div class="site-prevnext-group"><span class="site-prevnext-kicker">Previous volume</span>'
                         '<a class="site-prevnext-link" href="%s" aria-label="Previous volume: %s">%s'
                         '<span class="site-prevnext-title">%s</span></a></div>'
                         % (pf, htmlmod.escape(pt), _CHEV_L, htmlmod.escape(pt)))
        else:
            prev_html = ('<div class="site-prevnext-group"><span class="site-prevnext-kicker">Previous volume</span>'
                         '<span class="site-prevnext-empty">Start</span></div>')
        nxt = SITE_MAP[idx + 1] if idx + 1 < total else None
        if nxt:
            next_html = ('<div class="site-prevnext-group site-prevnext-next"><span class="site-prevnext-kicker">Next volume</span>'
                         '<a class="site-prevnext-link" href="%s" aria-label="Next volume: %s">'
                         '<span class="site-prevnext-title">%s</span>%s</a></div>'
                         % (nxt[0], htmlmod.escape(nxt[1]), htmlmod.escape(nxt[1]), _CHEV_R))
        else:
            next_html = ('<div class="site-prevnext-group site-prevnext-next"><span class="site-prevnext-kicker">Next volume</span>'
                         '<span class="site-prevnext-empty">End</span></div>')
        crumbs = ('<a href="index.html">Home</a><span class="bc-sep" aria-hidden="true">/</span>'
                  '<span>%s</span><span class="bc-sep" aria-hidden="true">/</span>'
                  '<span class="bc-current" aria-current="page">%s</span>'
                  % (htmlmod.escape(section), htmlmod.escape(page_title)))
    else:
        issue = "Curriculum"
        prev_html, next_html = "", ""
        crumbs = '<a href="index.html">Home</a>'
    return ("""<header class="site-header">"""
            """<p class="mast-line"><span>Research-Engineer Curriculum</span>"""
            """<span class="mast-line-right">%s &middot; Sept 2026 edition</span></p>"""
            """<div class="mast-row">"""
            """<a class="mast-name" href="index.html">LLM Knowledge Base</a>"""
            """<nav class="site-prevnext" aria-label="Volume sequence">%s%s</nav></div>"""
            """<div class="mast-sub">"""
            """<nav class="breadcrumbs" aria-label="Breadcrumb">%s</nav>"""
            """<button class="site-nav-toggle" id="siteNavToggle" aria-expanded="false" """
            """aria-controls="siteNavPanel">%s<span>Browse all volumes</span></button></div>"""
            """<div class="site-nav-panel" id="siteNavPanel"><div class="site-nav-panel-inner">"""
            """<div class="site-nav-drawer-head"><span>All volumes</span>"""
            """<button class="site-nav-close" id="siteNavClose">Close</button></div>"""
            """<div class="site-nav-groups">%s</div></div></div>"""
            """<hr class="mast-rule-double" aria-hidden="true">"""
            """<div class="site-nav-scrim" id="siteNavScrim"></div></header>"""
            % (issue, prev_html, next_html, crumbs, _CHEV_D, _site_nav_groups(current)))

def site_footer_html(current):
    """Ft1 mast-headed footer: wordmark band + grouped index + colophon."""
    return ("""<footer class="site-footer">"""
            """<div class="foot-mast">"""
            """<a class="foot-name" href="index.html">LLM Knowledge Base</a>"""
            """<p class="foot-tag">A research-engineer curriculum in 39 self-contained volumes. """
            """Every page works offline from file:// and links back here. No personal information.</p></div>"""
            """<nav class="foot-index" aria-label="Site index">%s</nav>"""
            """<p class="foot-colophon"><span>LLM Knowledge Base</span>"""
            """<span class="foot-sep" aria-hidden="true">/</span>"""
            """<span>Sept 2026 edition</span>"""
            """<span class="foot-sep" aria-hidden="true">/</span>"""
            """<span>Set in Charis SIL</span></p></footer>"""
            % _foot_index_groups(current))

# ---------------------------------------------------------------- page build

SIDEBAR_SKELETON = """<body>
<div class="ds-progress" id="dsProgress"></div>
<button class="ds-menu-btn" id="dsMenuBtn" aria-label="Open navigation">&#9776;</button>
<aside class="ds-sidebar" id="dsSidebar" aria-label="Volume navigation">
<div class="ds-side-head">
<p class="ds-side-eyebrow">Research-Engineer Curriculum</p>
<p class="ds-side-title">{side_title}</p>
<input class="ds-side-search" id="dsNavSearch" type="search" placeholder="Filter sections..." aria-label="Filter sections">
</div>
<nav class="ds-nav" id="dsNav" aria-label="Sections"></nav>
<div class="ds-side-foot"><span id="dsCompleteCount"></span></div>
</aside>
<div class="ds-scrim" id="dsScrim"></div>
<main class="ds-content" id="dsContent">
{site_header}
{content}
{site_footer}
</main>
<script>
{js}
</script>
</body>"""

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{css}
</style>
</head>
{body}
</html>"""


def parse_front_matter(text):
    meta = {}
    m = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        text = text[m.end():]
    return meta, text


def build_one(md_path, out_path=None):
    warnings = []
    src_dir = os.path.dirname(os.path.abspath(md_path))
    text = open(md_path, encoding="utf-8").read()

    # code standard gate on the source (Python-only, commented)
    cs_issues, cs_warnings = code_standard_check_md(text)
    warnings.extend("code: " + w for w in cs_warnings)
    code_failures = cs_issues

    meta, text = parse_front_matter(text)
    title = meta.get("title") or os.path.splitext(os.path.basename(md_path))[0]
    side_title = meta.get("eyebrow") or title

    # 1. protect fenced code; mermaid blocks get their own token path
    text, code_blocks = extract_fenced_blocks(text)
    mermaid_html = {}
    for i, b in enumerate(code_blocks):
        m = re.match(r"```mermaid\n(.*?)```$", b, flags=re.DOTALL)
        if m:
            token = "\ue030mermaid%d\ue031" % i
            mermaid_html[token] = render_mermaid_src(m.group(1), warnings)
            text = text.replace("\ue010code%d\ue011" % i, token)
            code_blocks[i] = None  # do not restore as code

    # 2. embed local images
    text = embed_local_images(text, src_dir, warnings)
    # 3. dfn placeholders
    text, dfn_terms = apply_dfn_placeholders(text)
    # 4. directives -> component HTML, in source order
    root = parse_directives(text.splitlines())
    body_html = render_node(root)
    # 5. restore dfn, mermaid svg, then code blocks
    body_html = restore_dfn(body_html, dfn_terms)
    for token, svg in mermaid_html.items():
        body_html = body_html.replace(token, svg)
    body_html = restore_fenced_blocks(body_html, code_blocks)

    css = open(DS_CSS, encoding="utf-8").read()
    js = open(DS_JS, encoding="utf-8").read()
    out_name = os.path.basename(out_path) if out_path else os.path.splitext(os.path.basename(md_path))[0] + ".html"
    _sec = next((s for (f, _, s) in SITE_MAP if f == out_name), "")
    _pt = next((t for (f, t, _) in SITE_MAP if f == out_name), title)
    site_header = site_header_html(out_name, _sec, _pt)
    site_footer = site_footer_html(out_name)
    body = SIDEBAR_SKELETON.format(side_title=htmlmod.escape(side_title),
                                   content=body_html, js=js,
                                   site_header=site_header, site_footer=site_footer)
    page = PAGE.format(title=htmlmod.escape(title), css=css, body=body)

    if out_path is None:
        out_path = os.path.splitext(md_path)[0] + ".html"
    open(out_path, "w", encoding="utf-8").write(page)
    return out_path, warnings, code_failures


def find_mmdc():
    for cand in (shutil.which("mmdc"),
                 os.path.expanduser("~/workspace/tools/mermaid/node_modules/.bin/mmdc")):
        if cand and os.path.exists(cand):
            return cand
    return None


def render_mermaid_src(src, warnings):
    mmdc = find_mmdc()
    if not mmdc:
        warnings.append("mermaid: mmdc not installed; kept as code block")
        return "<pre><code>" + htmlmod.escape(src) + "</code></pre>"
    puppeteer_cfg = os.path.join(HERE, "puppeteer.json")
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "d.mmd")
        outp = os.path.join(td, "d.svg")
        open(inp, "w").write(src)
        cmd = [mmdc, "-i", inp, "-o", outp, "-b", "transparent",
               "--size", "1400"]
        mmd_cfg = os.path.join(HERE, "mermaid-config.json")
        if os.path.exists(mmd_cfg):
            cmd[1:1] = ["-c", mmd_cfg]
        if os.path.exists(puppeteer_cfg):
            cmd[1:1] = ["-p", puppeteer_cfg]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode == 0 and os.path.exists(outp):
            svg = open(outp).read()
            return '<figure class="diagram">' + re.sub(r"<\?xml.*?\?>", "", svg).strip() + "</figure>"
    warnings.append("mermaid: render failed; kept as code block")
    return "<pre><code>" + htmlmod.escape(src) + "</code></pre>"


# ---------------------------------------------------------------- QA checks

# ---------------------------------------------------------------------------
# Humanizer gate (user standing order, from ~/workspace/skills/humanizer/SKILL.md)
# Hard FAIL: unambiguous AI tells + em dash anywhere in shipped content.
# WARN (not fail): context-dependent words that can be legitimate technical
# usage (robust statistics, etc.) — reported with context for human judgment.
# ---------------------------------------------------------------------------
HUMANIZER_FAIL_PATTERNS = [
    (r"\bdelv\w+", "banned AI tell: delve/delves/delving"),
    (r"\bfurthermore\b", "banned AI tell: furthermore"),
    (r"\bmoreover\b", "banned AI tell: moreover"),
    (r"(?:^|[.!?:;]\s+)additionally\b", "banned AI tell: 'additionally' as sentence starter"),
    (r"(?:^|[.!?:;]\s+)overall\b", "banned AI tell: 'overall' as sentence starter"),
    (r"\btapestry\b", "banned AI tell: tapestry"),
    (r"\bseamless(?:ly)?\b", "banned AI tell: seamless"),
    (r"\bgame[- ]changer\b", "banned AI tell: game-changer"),
    (r"\bsupercharg\w+", "banned AI tell: supercharge"),
    (r"\bdeep dive\b", "banned AI tell: 'deep dive'"),
    (r"\bit'?s worth noting\b", "banned AI tell: \"it's worth noting\""),
    (r"\bit'?s important to note\b", "banned AI tell: \"it's important to note\""),
    (r"\bin conclusion\b", "banned AI tell: 'in conclusion'"),
    (r"\bas an AI\b", "banned AI tell: 'as an AI'"),
    (r"\belevat\w+\b", "banned AI tell: elevate"),
    (r"\bin today'?s\b", "banned AI tell: \"in today's\""),
]
HUMANIZER_WARN_PATTERNS = [
    (r"\brobust(?:ly|ness)?\b", "'robust' — ok only as a technical term (robust statistics); otherwise an AI tell"),
    (r"(?<![-\w])leverag(?:e|es|ed|ing)\b", "'leverage' as a verb is an AI tell; prefer 'use'"),
    (r"\bunlock(?:s|ed|ing)?\b", "'unlock' is an AI tell; check usage"),
]
THROAT_CLEARING = [
    r"in this section we will",
    r"in this chapter we will",
    r"let'?s talk about",
    r"now,? let'?s understand",
    r"we will explore",
    r"let'?s dive into",
    r"in this volume we",
]


def _prose_text(raw, is_html):
    """Reader-facing text: strip data URIs, code, scripts, styles, markup."""
    t = re.sub(r"data:[^\"'\)]*", "", raw)
    if is_html:
        t = re.sub(r"<script.*?</script>", " ", t, flags=re.DOTALL | re.IGNORECASE)
        t = re.sub(r"<style.*?</style>", " ", t, flags=re.DOTALL | re.IGNORECASE)
        t = re.sub(r"<pre.*?</pre>", " ", t, flags=re.DOTALL | re.IGNORECASE)
        t = re.sub(r"<code.*?</code>", " ", t, flags=re.DOTALL | re.IGNORECASE)
        t = re.sub(r"<[^>]+>", " ", t)
    else:
        t = re.sub(r"```.*?```", " ", t, flags=re.DOTALL)
        t = re.sub(r"`[^`]*`", " ", t)
        t = re.sub(r"^#{1,6}\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"[ \t]+", " ", t)
    return t


def humanizer_check(raw, is_html):
    """Returns (fail_issues, warn_issues) per the humanizer skill rules."""
    fails, warns = [], []
    prose = _prose_text(raw, is_html)
    # 1. em dash: zero anywhere in shipped content (scripts/styles/data
    #    URIs are not reader content and are excluded).
    content = re.sub(r"data:[^\"'\)]*", "", raw)
    content = re.sub(r"<script.*?</script>", " ", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"<style.*?</style>", " ", content, flags=re.DOTALL | re.IGNORECASE)
    if "\u2014" in content:
        m = re.search(r".{40}\u2014.{40}", content.replace("\n", " "))
        fails.append("em dash found in shipped content" +
                     (" near: ..." + m.group(0) + "..." if m else ""))
    # 2. banned AI tells (hard fail). Quoted spans are excluded: a "tell"
    # inside double quotes is quoted material (paper titles like "Delving
    # Deep into Rectifiers", quoted speech), not the author's own prose
    # voice, so it must not fail the gate.
    unquoted = re.sub(r'"[^"\n]{0,400}"', ' ', prose)
    for pat, label in HUMANIZER_FAIL_PATTERNS:
        ms = list(re.finditer(pat, unquoted, re.IGNORECASE | re.MULTILINE))
        if ms:
            sample = ms[0].group(0)
            ctx = prose[max(0, ms[0].start() - 45):ms[0].end() + 45].replace("\n", " ")
            fails.append("%s (%d×) e.g. ...%s..." % (label, len(ms), ctx.strip()))
    # 3. context-dependent tells (warn only)
    for pat, label in HUMANIZER_WARN_PATTERNS:
        ms = list(re.finditer(pat, prose, re.IGNORECASE))
        if ms:
            warns.append("%s (%d×)" % (label, len(ms)))
    # 4. throat-clearing openers (warn)
    for pat in THROAT_CLEARING:
        ms = list(re.finditer(pat, prose, re.IGNORECASE))
        if ms:
            warns.append("throat-clearing: '%s' (%d×)" % (ms[0].group(0), len(ms)))
    # 5. long sentences (warn): flag sentences over ~30 words with samples.
    # Tables are excluded (cell text is fragments); block boundaries are
    # turned into sentence breaks so list items / paragraphs don't merge.
    if is_html:
        block = re.sub(r"</(p|li|h[1-6]|td|th|tr|div|figcaption|caption)>",
                       ". ", raw, flags=re.IGNORECASE)
        block = re.sub(r"<(br|hr)[^>]*>", ". ", block, flags=re.IGNORECASE)
        block = re.sub(r"<table.*?</table>", " ", block,
                       flags=re.DOTALL | re.IGNORECASE)
        sent_text = _prose_text(block, True)
    else:
        # markdown: drop pipe-table rows (cell text is fragments)
        sent_text = re.sub(r"^\s*\|.*$", " ", prose, flags=re.MULTILINE)
    sentences = re.split(r"(?<=[.!?])\s+", sent_text)
    long = []
    for s in sentences:
        words = re.findall(r"[A-Za-z0-9']+", s)
        if len(words) > 30:
            long.append((len(words), " ".join(words[:14]) + "..."))
    if long:
        long.sort(reverse=True)
        warns.append("%d sentences over 30 words (longest %d words): %s" %
                     (len(long), long[0][0],
                      " | ".join(x[1] for x in long[:2])))
    return fails, warns


# ---------------------------------------------------------------------------
# Code standard (user hard rule): ALL code samples are Python — no other
# programming languages anywhere. Python samples must be thoroughly
# commented; non-trivial samples get a plain-English walkthrough; key
# flows (attention, backprop, collectives, RL loops, inference batching,
# RAG/agent flows) pair code with a visual (generated image / Mermaid / ASCII).
# ---------------------------------------------------------------------------
NON_PYTHON_LANGS = {
    "rust", "cpp", "c++", "c", "cuda", "js", "javascript", "ts", "typescript",
    "go", "java", "kotlin", "swift", "ruby", "r", "matlab", "julia", "scala",
    "haskell", "php",
}
# tags that are not code samples (diagrams, transcripts, plain text)
NON_CODE_TAGS = {"", "text", "ascii", "mermaid", "bash", "sh", "shell",
                 "console", "diff", "json", "yaml", "toml"}
# strong non-Python signatures (WARN-only heuristics)
NON_PY_SIGS = [r"#include\s*<", r"\bstd::", r"\bfn\s+\w+\s*\(",
               r"public\s+(static\s+)?(void|class)\b", r"console\.log\(",
               r"\bfunc\s+\w+\s*\(", r"=>\s*\{", r"System\.out\.print"]


def code_standard_check_md(md_text):
    """Enforce the Python-only code standard on a Markdown source."""
    issues, warnings = [], []
    for m in re.finditer(r"```(\w*)\n(.*?)```", md_text, re.DOTALL):
        lang = m.group(1).lower()
        body = m.group(2)
        nlines = len([l for l in body.splitlines() if l.strip()])
        if lang in NON_PYTHON_LANGS:
            issues.append("non-Python code block tagged '%s' (%d lines) — "
                          "all code samples must be Python" % (lang, nlines))
            continue
        if lang in ("python", "py"):
            if nlines > 15:
                comments = len([l for l in body.splitlines()
                                if l.strip().startswith("#")])
                if comments / max(nlines, 1) < 0.08:
                    warnings.append(
                        "low comment density in Python sample (%d lines, "
                        "%d%% comments) — every block needs what/why/"
                        "what-breaks comments" % (nlines,
                        int(100 * comments / nlines)))
        elif lang == "" or lang not in NON_CODE_TAGS:
            # untagged or unknown block: heuristic scan for non-Python
            for pat in NON_PY_SIGS:
                if re.search(pat, body):
                    warnings.append("possible non-Python code in '%s' block "
                                    "(%d lines) — verify it is Python"
                                    % (lang or "untagged", nlines))
                    break
    return issues, warnings


def code_standard_check_html(raw):
    """Heuristic-only scan of built HTML <pre> blocks (WARN, can't be sure)."""
    warnings = []
    for m in re.finditer(r"<pre[^>]*>(.*?)</pre>", raw,
                         re.DOTALL | re.IGNORECASE):
        body = re.sub(r"<[^>]+>", "", m.group(1))
        for pat in NON_PY_SIGS:
            if re.search(pat, body):
                warnings.append("possible non-Python code in a <pre> block — "
                                "verify all samples are Python")
                break
    return warnings


def qa_check(path):
    issues = []
    warnings = []
    raw = open(path, encoding="utf-8").read()
    is_html = path.endswith(".html")
    # strip data URIs + code blocks for prose checks
    prose = re.sub(r"data:[^\"'\)]*", "", raw)
    prose = re.sub(r"<pre>.*?</pre>", "", prose, flags=re.DOTALL)
    prose = re.sub(r"<code>.*?</code>", "", prose, flags=re.DOTALL)
    # humanizer gate: em dashes, banned AI tells, throat-clearing, long sentences
    h_fails, h_warns = humanizer_check(raw, is_html)
    issues.extend(h_fails)
    warnings.extend(h_warns)
    # code standard: Python-only (heuristic scan on built HTML; the fence-
    # language gate runs on .md sources in build_one / --check)
    for w in code_standard_check_html(raw):
        warnings.append(w)
    if "<link" in raw and "stylesheet" in raw:
        issues.append("external stylesheet link present")
    if re.search(r"<script\s+src=", raw):
        issues.append("external script src present")
    if re.search(r"<img[^>]+src=\"http", raw):
        issues.append("remote img src present (breaks offline/file://)")
    css_m = re.search(r"<style>(.*?)</style>", raw, flags=re.DOTALL)
    if css_m and "gradient" in css_m.group(1):
        issues.append("CSS gradient found")
    if re.search(r"(TODO|FIXME|lorem ipsum)", prose, re.IGNORECASE):
        issues.append("TODO/FIXME/lorem marker found")
    # personal identifiers (generic-content rule). SVG diagram markup is not
    # prose: mermaid path coordinates (e.g. "125 588.0625") false-positive
    # the phone pattern, so strip svg before scanning. For .md sources also
    # strip fenced code blocks: fake example data inside a sample (e.g. an
    # attacker's address in a prober payload) is not a personal identifier.
    id_prose = re.sub(r"<svg.*?</svg>", " ", prose, flags=re.DOTALL)
    if not is_html:
        id_prose = re.sub(r"```.*?```", " ", id_prose, flags=re.DOTALL)
    for pat in [r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"]:
        if re.search(pat, id_prose):
            issues.append("possible personal identifier: " + pat[:20])
            break
    return issues, warnings


def main(argv):
    if "--check" in argv:
        argv.remove("--check")
        ok = True
        for p in argv[1:]:
            issues, warnings = qa_check(p)
            if p.endswith(".md"):
                # fence-language + comment-density gate on the source
                src = open(p, encoding="utf-8").read()
                ci, cw = code_standard_check_md(src)
                issues.extend(ci)
                warnings.extend("code: " + w for w in cw)
            print(("PASS " if not issues else "FAIL ") + p)
            for i in issues:
                print("   - " + i)
                ok = False
            for w in warnings:
                print("   ~ " + w)
        sys.exit(0 if ok else 1)

    targets = []
    if "--all" in argv:
        if not os.path.isdir(SOURCES):
            sys.exit("no sources/ dir; pass .md files explicitly")
        targets = [os.path.join(SOURCES, f) for f in sorted(os.listdir(SOURCES))
                   if f.endswith(".md")]
    else:
        targets = [a for a in argv[1:] if a.endswith(".md")]
    if not targets:
        sys.exit("usage: build.py [--all | volume.md ...] [--check out.html ...]")

    for t in targets:
        out, warnings, code_issues = build_one(t)
        size = os.path.getsize(out)
        print("built %s (%d KB)" % (out, size // 1024))
        issues, qa_warnings = qa_check(out)
        issues.extend(code_issues)
        for w in warnings:
            print("   warn: " + w)
        for w in qa_warnings:
            print("   ~ " + w)
        for i in issues:
            print("   QA FAIL: " + i)


if __name__ == "__main__":
    main(sys.argv)
