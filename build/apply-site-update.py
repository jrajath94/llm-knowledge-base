#!/usr/bin/env python3
"""Apply the design-standard update to existing curriculum HTML files.

1. Replace the inlined canonical ds.css with the new version (Charis SIL + site chrome styles).
2. Inject the site header (home link, prev/next, breadcrumbs, volume nav) as the
   first child of <main>, and the site footer (full sitemap) as the last child.
Idempotent: safe to re-run.
"""
import os, re, sys, html as htmlmod

CUR = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import site_header_html, site_footer_html, SITE_MAP  # noqa

NEW_CSS = open(os.path.join(CUR, "build", "ds.css"), encoding="utf-8").read()
MARKER = "shared design system"
FILES = sorted(f for f in os.listdir(CUR) if f.endswith(".html"))

SEC_OF = {f: s for (f, _, s) in SITE_MAP}
TITLE_OF = {f: t for (f, t, _) in SITE_MAP}

def page_title(raw, fname):
    m = re.search(r"<title>(.*?)</title>", raw, re.S)
    if m:
        return htmlmod.unescape(m.group(1).strip())
    return TITLE_OF.get(fname, fname)

def update(path):
    raw = open(path, encoding="utf-8").read()
    changed = []
    # 1. swap canonical CSS block
    def css_repl(m):
        block = m.group(0)
        if MARKER in block and 'font-family:"Charis SIL"' not in block:
            changed.append("css")
            return "<style>\n" + NEW_CSS + "\n</style>"
        return block
    raw = re.sub(r"<style>.*?</style>", css_repl, raw, flags=re.DOTALL)
    # 2/3. inject site header + footer (once)
    if 'class="site-header"' not in raw:
        fname = os.path.basename(path)
        title = page_title(raw, fname)
        sec = SEC_OF.get(fname, "")
        nav_title = TITLE_OF.get(fname, title)
        header = site_header_html(fname, sec, nav_title)
        footer = site_footer_html(fname)
        main_open = '<main class="ds-content" id="dsContent">'
        assert main_open in raw, "no main in " + path
        raw = raw.replace(main_open, main_open + "\n" + header, 1)
        # footer goes right before </main>
        assert "</main>" in raw
        raw = raw.replace("</main>", footer + "\n</main>", 1)
        changed.append("chrome")
    if changed:
        open(path, "w", encoding="utf-8").write(raw)
    return changed

if __name__ == "__main__":
    only = sys.argv[1:] or FILES
    for f in only:
        p = os.path.join(CUR, f) if not os.path.isabs(f) else f
        if not os.path.exists(p):
            print("SKIP (missing)", f)
            continue
        try:
            ch = update(p)
            print(("UPDATED " if ch else "ok ") + f + (" [" + ",".join(ch) + "]" if ch else ""))
        except Exception as e:
            print("ERROR " + f + ": " + e)
