#!/usr/bin/env python3
"""Chrome-redesign patcher (hallmark redesign, chrome-only).

Replaces, in place, for each of the 39 SITE_MAP pages:
  1. the inlined canonical ds.css block ("shared design system" marker)
  2. the inlined ds.js block ("shared behavior" marker)
  3. <header class="site-header">...</header> with the new masthead header
  4. <footer class="site-footer">...</footer> with the new masthead footer
Page content is never touched. Idempotent: re-running regenerates identical chrome.
"""
import os, re, sys, html as htmlmod

CUR = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import site_header_html, site_footer_html, SITE_MAP  # noqa

NEW_CSS = open(os.path.join(CUR, "build", "ds.css"), encoding="utf-8").read()
NEW_JS = open(os.path.join(CUR, "build", "ds.js"), encoding="utf-8").read()
CSS_MARKER = "shared design system"
JS_MARKER = "shared behavior"
FILES = [f for (f, _, _) in SITE_MAP]
SEC_OF = {f: s for (f, _, s) in SITE_MAP}
TITLE_OF = {f: t for (f, t, _) in SITE_MAP}

HEADER_RE = re.compile(r'<header class="site-header">.*?</header>', re.DOTALL)
FOOTER_RE = re.compile(r'<footer class="site-footer">.*?</footer>', re.DOTALL)
STYLE_RE = re.compile(r"<style>.*?</style>", re.DOTALL)
SCRIPT_RE = re.compile(r"<script>.*?</script>", re.DOTALL)
SCRIM_RE = re.compile(r'<div class="site-nav-scrim"[^>]*></div>')

def patch(path):
    fname = os.path.basename(path)
    raw = open(path, encoding="utf-8").read()
    changed = []

    # stray scrims first (they are re-issued inside the new header)
    raw, n_scrim = SCRIM_RE.subn("", raw)
    if n_scrim:
        changed.append("scrim-x%d" % n_scrim)

    def css_repl(m):
        if CSS_MARKER in m.group(0):
            changed.append("css")
            return "<style>\n" + NEW_CSS + "\n</style>"
        return m.group(0)
    raw = STYLE_RE.sub(css_repl, raw)

    def js_repl(m):
        if JS_MARKER in m.group(0):
            changed.append("js")
            return "<script>\n" + NEW_JS + "\n</script>"
        return m.group(0)
    raw = SCRIPT_RE.sub(js_repl, raw)

    new_header = site_header_html(fname, SEC_OF.get(fname, ""), TITLE_OF.get(fname, fname))
    raw, n = HEADER_RE.subn(lambda m: new_header, raw)
    if n != 1:
        raise RuntimeError("header replace count=%d in %s" % (n, fname))
    changed.append("header")

    new_footer = site_footer_html(fname)
    raw, n = FOOTER_RE.subn(lambda m: new_footer, raw)
    if n != 1:
        raise RuntimeError("footer replace count=%d in %s" % (n, fname))
    changed.append("footer")

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
            ch = patch(p)
            print("UPDATED " + f + " [" + ",".join(ch) + "]")
        except Exception as e:
            print("ERROR " + f + ": " + e)
