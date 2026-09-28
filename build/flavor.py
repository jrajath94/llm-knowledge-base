#!/usr/bin/env python3
"""flavor.py -- repo-discipline transform for the research-engineer curriculum.

Rewrites interview framing into work-safe knowledge-base language in BUILT HTML
files (the git repo must read as a professional knowledge base at work).

Usage:
    python3 flavor.py path/to/volume.html [more.html ...]
    python3 flavor.py --all            # every top-level volume HTML (root NN-*.html + index.html)
    python3 flavor.py --all --dry-run  # report counts, change nothing

What it does, per file:
  1. Applies deterministic, humanizer-clean replacements (visible text only).
  2. Renames 14-behavioral-research.html -> 14-communicating-research.html
     (when that file is among the inputs) and fixes its links in index.html
     and any other processed file.
  3. Scans the result for any remaining case-insensitive "interview"
     occurrences, prints each with file + line + context, and exits NONZERO
     if any human-visible occurrence remains (so a human reviews leftovers).

Never altered: <pre>, <code>, <svg> blocks, HTML comments, href/src URLs,
numbers, facts, UNVERIFIED markers. CSS class names are kept as-is (only
human-visible text changes).

NOTE (build mechanics, for Phase B): build.py renders the ::: ob-board
directive from COMPONENT_DEFAULTS["ob-board"] = ("ob-board", "On the board"),
so any future rebuild from .md reintroduces the old label unless
COMPONENT_DEFAULTS is updated there too. This script only fixes built HTML.
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CURRICULUM_ROOT = os.path.dirname(HERE)  # build/ lives one level under the root

# ---------------------------------------------------------------------------
# Replacement rules, applied in order. Each entry: (name, compiled regex, replacement).
# Replacements are humanizer-clean by construction: no em dashes, no banned
# AI tells (delve / leverage / furthermore / robust / seamless / utilize / ...).
# Case is preserved: ALL CAPS -> ALL CAPS, Title Case -> Title Case, else lower.
# ---------------------------------------------------------------------------

RULES = [
    ("interview-tag-label",
     re.compile(r"\[INTERVIEW\]"),
     "[FIELD NOTE]"),

    ("board-box-label",
     re.compile(r"\bOn the board\b"),
     "Design prompt"),

    ("behavioral-interview",
     re.compile(r"\bbehavioral interviews?\b", re.IGNORECASE),
     "research communication"),

    ("in-interviews",
     re.compile(r"\bin interviews\b", re.IGNORECASE),
     "in design reviews"),

    ("interviewers-ask",
     re.compile(r"\binterviewers ask\b", re.IGNORECASE),
     "reviewers ask"),

    ("interviewer-asks",
     re.compile(r"\binterviewer asks\b", re.IGNORECASE),
     "reviewer asks"),

    ("interview-question",
     re.compile(r"\binterview questions?\b", re.IGNORECASE),
     "design question"),

    ("for-the-interview",
     re.compile(r"\bfor the interview\b", re.IGNORECASE),
     "for the design review"),

    ("interview-prep",
     re.compile(r"\binterview prep\b", re.IGNORECASE),
     "deep study"),

    ("behavioral-research-title",
     re.compile(r"\bBehavioral Research\b"),
     "Communicating Research"),
]

BANNED_TELLS = [
    "delve", "leverage", "furthermore", "robust", "seamless", "utilize",
    "utilise", "moreover", "additionally", "game-changer", "cutting-edge",
    "in today's fast-paced", "tapestry", "landscape", "realm",
]
EM_DASH = "\u2014"


def _check_humanizer_clean():
    """Fail fast if any replacement string violates the humanizer gate."""
    for name, _rx, repl in RULES:
        if EM_DASH in repl:
            raise SystemExit("humanizer gate: em dash in replacement %r (%s)" % (repl, name))
        low = repl.lower()
        for tell in BANNED_TELLS:
            if tell in low:
                raise SystemExit(
                    "humanizer gate: banned tell %r in replacement %r (%s)"
                    % (tell, repl, name))
        # singular/plural sanity: a rule ending in a bare noun must not
        # accidentally pluralize via the case-preserver.


def _preserve_case(match_text, replacement):
    if match_text.isupper():
        return replacement.upper()
    if match_text[0].isupper():
        # Title-case the first word only ("in design reviews" -> "In design reviews")
        first, _, rest = replacement.partition(" ")
        return first.capitalize() + (" " + rest if rest else "")
    return replacement


def apply_rules(text, counts):
    """Apply all RULES to `text`; bump counts[name] per substitution."""
    for name, rx, repl in RULES:

        def _sub(m, _repl=repl, _name=name):
            counts[_name] = counts.get(_name, 0) + 1
            return _preserve_case(m.group(0), _repl)

        text = rx.sub(_sub, text)
    return text


# ---------------------------------------------------------------------------
# Protection: never touch <pre>, <code>, <svg>, HTML comments, or href/src URLs.
# ---------------------------------------------------------------------------

PROTECTED_BLOCK = re.compile(
    r"(?P<blk><!--.*?-->|<pre\b.*?</pre>|<code\b.*?</code>|<svg\b.*?</svg>)",
    re.DOTALL | re.IGNORECASE,
)
TAG_SPLIT = re.compile(r"(<[^>]+>)")
ATTR_TEXT = re.compile(
    r'\b(alt|title|aria-label)\s*=\s*(?P<q>["\'])(?P<val>.*?)(?P=q)',
    re.DOTALL | re.IGNORECASE,
)
URL_ATTR = re.compile(
    r'\b(href|src|srcset|data-src|action|cite)\s*=\s*(["\']).*?\2',
    re.DOTALL | re.IGNORECASE,
)


def transform_html(html):
    """Return (new_html, counts). Replacements apply to visible text and to
    alt/title/aria-label attribute values; tags, comments, code, svg, and
    href/src URLs are untouched."""
    counts = {}
    out = []
    for chunk in PROTECTED_BLOCK.split(html):
        if not chunk:
            continue
        if PROTECTED_BLOCK.fullmatch(chunk):
            out.append(chunk)  # protected: verbatim
            continue
        for piece in TAG_SPLIT.split(chunk):
            if not piece:
                continue
            if piece.startswith("<"):
                # tag: only touch human-readable attribute values
                # (alt/title/aria-label). href/src/class/id stay verbatim.
                piece = ATTR_TEXT.sub(
                    lambda m: '%s=%s%s%s' % (
                        m.group(0).split("=")[0],
                        m.group("q"),
                        apply_rules(m.group("val"), counts),
                        m.group("q"),
                    ),
                    piece,
                )
                out.append(piece)
            else:
                out.append(apply_rules(piece, counts))
    return "".join(out), counts


# ---------------------------------------------------------------------------
# Residual scan: any "interview" left? Human-visible ones fail the run.
# ---------------------------------------------------------------------------

INTERVIEW_RX = re.compile(r"interview", re.IGNORECASE)
TAG_SPAN_RX = re.compile(r"<[^>]*>")


def _attr_at(line, start, end, attr):
    """True if the match lies inside attr="..." on this line."""
    for m in re.finditer(r'%s\s*=\s*(["\'])(.*?)\1' % attr, line, re.IGNORECASE):
        if m.start(2) <= start and end <= m.end(2):
            return True
    return False


def scan_residual(path, html):
    """Return (human_visible_hits, markup_only_hits, url_hits).
    Each hit: (lineno, context). Markup-only hits (class=/id= attributes)
    and URL hits (href/src/... attributes, which we must never rewrite)
    are reported but do not fail the run; only human-visible prose fails."""
    visible, markup, urls = [], [], []
    for lineno, line in enumerate(html.splitlines(), 1):
        for m in INTERVIEW_RX.finditer(line):
            s, e = m.start(), m.end()
            in_tag = any(ts <= s and e <= te
                         for ts, te in
                         ((t.start(), t.end()) for t in TAG_SPAN_RX.finditer(line)))
            ctx = line[max(0, s - 70): e + 70].strip()
            hit = (lineno, ctx)
            if in_tag and (_attr_at(line, s, e, "class")
                           or _attr_at(line, s, e, "id")):
                markup.append(hit)
            elif in_tag and any(_attr_at(line, s, e, a) for a in
                                ("href", "src", "srcset", "data-src",
                                 "action", "cite", "content")):
                urls.append(hit)
            else:
                visible.append(hit)
    return visible, markup, urls


# ---------------------------------------------------------------------------
# File selection / rename / link fixup
# ---------------------------------------------------------------------------

RENAME_FROM = "14-behavioral-research.html"
RENAME_TO = "14-communicating-research.html"


def all_volume_files():
    """Top-level volume HTMLs: root NN-*.html (00-14) plus index.html.
    Excludes .cleaned.html twins? No -- includes them (idempotent).
    Excludes dsa-track-300.html (separate track, parent decides) and
    vol*.src.html (source files, not built volumes)."""
    files = []
    for name in sorted(os.listdir(CURRICULUM_ROOT)):
        p = os.path.join(CURRICULUM_ROOT, name)
        if not os.path.isfile(p) or not name.endswith(".html"):
            continue
        if name == "index.html" or re.match(r"^[0-9]{2}-.+\.html$", name):
            if name == "dsa-track-300.html":
                print("  skip (separate track): %s" % name)
                continue
            if re.match(r"^vol\d+\.src\.html$", name):
                continue
            files.append(p)
    return files


def process_file(path, dry_run=False):
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    new_html, counts = transform_html(html)
    if not dry_run and new_html != html:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_html)
    return counts, new_html


def fixup_links(paths, dry_run=False):
    """After the 14- rename, point links at the new filename."""
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        if RENAME_FROM in html:
            html = html.replace(RENAME_FROM, RENAME_TO)
            if not dry_run:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(html)
            print("  link fixup: %s" % os.path.basename(path))


def main(argv=None):
    _check_humanizer_clean()

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", help="HTML file(s) to transform in place")
    ap.add_argument("--all", action="store_true",
                    help="transform every top-level volume HTML")
    ap.add_argument("--dry-run", action="store_true",
                    help="report counts, write nothing")
    args = ap.parse_args(argv)

    if args.all:
        targets = all_volume_files()
    elif args.files:
        targets = [os.path.abspath(f) for f in args.files]
    else:
        ap.error("give HTML file(s) or --all")

    missing = [t for t in targets if not os.path.isfile(t)]
    if missing:
        raise SystemExit("not found: %s" % ", ".join(missing))

    print("flavor.py: %d file(s)%s" % (len(targets), " (dry run)" if args.dry_run else ""))
    total_counts = {}
    new_htmls = {}
    renamed = False

    for path in targets:
        counts, new_html = process_file(path, dry_run=args.dry_run)
        new_htmls[path] = new_html
        for k, v in counts.items():
            total_counts[k] = total_counts.get(k, 0) + v
        n = sum(counts.values())
        print("  %-45s %3d replacement(s) %s" % (
            os.path.basename(path), n,
            ("{" + ", ".join("%s=%d" % kv for kv in sorted(counts.items())) + "}")
            if counts else ""))

    # Vol 14 rename + link fixup (only when the file itself was processed)
    from_path = os.path.join(CURRICULUM_ROOT, RENAME_FROM)
    to_path = os.path.join(CURRICULUM_ROOT, RENAME_TO)
    if from_path in [os.path.abspath(t) for t in targets]:
        if not args.dry_run:
            if os.path.exists(from_path):
                os.rename(from_path, to_path)
                renamed = True
                print("  renamed: %s -> %s" % (RENAME_FROM, RENAME_TO))
        else:
            print("  (dry run) would rename: %s -> %s" % (RENAME_FROM, RENAME_TO))
        fixup_targets = [t for t in targets
                         if os.path.basename(os.path.abspath(t)) != RENAME_FROM]
        # always include index.html for the link fixup even if not in targets
        idx = os.path.join(CURRICULUM_ROOT, "index.html")
        if idx not in fixup_targets and os.path.isfile(idx):
            fixup_targets.append(idx)
        fixup_links([t for t in fixup_targets if os.path.isfile(t)],
                    dry_run=args.dry_run)

    print("\nTotal replacements by pattern:")
    for k, v in sorted(total_counts.items(), key=lambda kv: -kv[1]):
        print("  %-28s %d" % (k, v))
    if not total_counts:
        print("  (none)")

    # Residual scan -- the human-review gate
    print("\nResidual 'interview' scan:")
    any_visible = False
    for path in targets:
        use_path = path
        if renamed and os.path.abspath(path) == os.path.abspath(from_path):
            use_path = to_path
            new_htmls[to_path] = new_htmls.pop(path)
        html = new_htmls.get(os.path.abspath(path), new_htmls.get(path))
        if html is None:
            with open(use_path, "r", encoding="utf-8") as f:
                html = f.read()
        visible, markup, urls = scan_residual(use_path, html)
        label = os.path.basename(use_path)
        for lineno, ctx in visible:
            print("  VISIBLE %s:%d: ...%s..." % (label, lineno, ctx))
        for lineno, ctx in markup:
            print("  markup-only %s:%d: ...%s..." % (label, lineno, ctx))
        for lineno, ctx in urls:
            print("  url (protected) %s:%d: ...%s..." % (label, lineno, ctx))
        if visible:
            any_visible = True
    if any_visible:
        print("\nFAIL: human-visible 'interview' occurrences remain -- manual review needed.")
        return 1
    print("\nOK: no human-visible 'interview' remains.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
